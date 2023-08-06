from AlgorithmImports import *
from QuantConnect.Data.Custom.Tiingo import *
import os 
import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as tb
import yfinance as yf
import ta 
from sklearn.linear_model import LinearRegression
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR
import cv2
from scipy import stats
import datetime
import warnings
from sklearn.covariance import LedoitWolf
warnings.filterwarnings('ignore')

from data_process import data_process
from fitting_model import fitting_model
from indicator_generator import indicator_generator
from weight_allocator import weight_allocator 
from weight_allocator_bear import weight_allocator_bear 

class GroupFinalProject(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2022, 1, 1)  
        self.SetEndDate(2022, 11, 1) 
        self.test_start_date = datetime.date(2017, 1, 1)
        self.SetCash(1000000)
        self.SetWarmup(50)
        # self.symbols =[]
        self.all_etf = ['QUAL','USMV','SIZE','MTUM','SPY', 'BND',
                    'IWF','RSP', 'IWD','IVE',
                    'DVY','DBA','KBE','IBB','ITB','ICLN','XLP','XLY',
                    'ITA','XLE','PBS','XLF','IYK','XLV','PEJ','KIE','QQQ','XLI','XLB',
                    'XME','IHE', 'XRT','SMH','SLX', 'XLK', 'IYT', 'XLU', 'TIP']


        self.lookback=252 #make sure lookback period longer than duration required for signal calculation.
        self.symbols=[]
        for etf_ticker in self.all_etf: 
            security=self.AddEquity(etf_ticker, Resolution.Daily)
            self.symbols.append(self.Symbol(etf_ticker))
            security.SetFeeModel(CustomFeeModel(self))
            security.SetSlippageModel(CustomSlippageModel(self))

        
        self.yesterday = None
        self.weight=None

        # liquidate when the following threshold are met 
        self.isTrailing = True 
        self.initialised = False
        self.portfolioHigh = 0
        self.maximumDrawdownPercent = -0.10

        # use RSI to determine our weight allocation 
        # SPY as a general market epitome 
        self.EnableAutomaticIndicatorWarmUp = True
        self.rsi = self.RSI("SPY", 40, Resolution.Daily)
        self.ema_long =self.EMA("SPY",50, resolution=Resolution.Daily)
        self.ema_short =self.EMA("SPY",30, resolution=Resolution.Daily)
        self.long_prev = None 
        self.short_prev = None 
        self.DOWN=False 
        self.bear = False 
        self.bull = False 
        self.bull_bear_list =[]
        self.rsi_min =25
        self.rsi_max = 50 #not really max, but to put an early stop on bear

        # Tiingo Initialization
        self.newsData = {}
        self.positive= set()
        self.negative= set()
        self.wordScores = {
            "bad": -0.5, "good": 0.5, "negative": -0.5, 
            "great": 0.5, "growth": 0.5, "fail": -0.5, 
            "failed": -0.5, "success": 0.5, "nailed": 0.5,
            "beat": 0.5, "missed": -0.5, "profitable": 0.5,
            "beneficial": 0.5, "right": 0.5, "positive": 0.5, 
            "large":0.5, "attractive": 0.5, "sound": 0.5, 
            "excellent": 0.5, "wrong": -0.5, "unproductive": -0.5, 
            "lose": -0.5, "missing": -0.5, "mishandled": -0.5, 
            "un_lucrative": -0.5, "up": 0.5, "down": -0.5,
            "unproductive": -0.5, "poor": -0.5, "wrong": -0.5,
            "worthwhile": 0.5, "lucrative": 0.5, "solid": 0.5
        } 
        
        self.UniverseSettings.Resolution = Resolution.Daily
        self.Schedule.On(self.DateRules.EveryDay("IWF"),
            self.TimeRules.AfterMarketOpen("IWF",-15), #Run 15 mins before market opens
            self.generate_weight)
        
        # Risk management initialization 
        self.pnl_history = pd.Series([]) # list of daily PnL
        self.maxDrawdown_duration=0
        self.trade_today = True
        self.today_date = None  

        self.Schedule.On(self.DateRules.EveryDay(),
                 self.TimeRules.AfterMarketOpen("IWF", 15), #Run 15 mins after market opens
                 self.riskManagement)

        self.Schedule.On(self.DateRules.EveryDay("IWF"),
                self.TimeRules.AfterMarketOpen("IWF", 20), #Run 20 mins after market opens
                self.trade)


    def trade(self):
        if self.IsWarmingUp: return 
        if self.Time.date()< self.test_start_date: return 

        self.Debug(self.Time)
        
        if self.trade_today: 
            self.weight = self.adjust_weight_sentiment(self.weight)
            self.weight = self.reduce_fee(self.weight)

            if self.yesterday is None:  #only frist period
                self.yesterday = self.weight

            for s in self.symbols:
                if not np.isnan(self.weight.loc[s]):
                    self.SetHoldings(s,self.weight.loc[s])
                    self.yesterday.loc[s] = self.weight.loc[s]

        self.positive= set()
        self.negative= set()


    def generate_weight(self): 
        if self.IsWarmingUp: return 
        if self.Time.date()< self.test_start_date: return 
        self.df = self.History(self.symbols, self.lookback) 
        self.open = self.df["open"].unstack(level=0)
        self.close = self.df["close"].unstack(level=0)
        self.high = self.df["high"].unstack(level=0)
        self.low = self.df["low"].unstack(level=0)
        
        ind_generator = indicator_generator(self.close, self.open, self.high, self.low)
        signal_dict = ind_generator.run_all()

        f = fitting_model(signal_dict,self.close)
        f.generate_reg_panel_df()
        pred_signal = f.aggregate_signal_by_max_ICIR()

        if len(self.bull_bear_list)==0:
            w = weight_allocator(pred_signal)
            self.bear = False 
            self.bull = True 
        else:
            date, ind = self.bull_bear_list[-1]
            use_bear = (ind == -1) #or self.DOWN 
            if use_bear:
                self.Debug(f"{self.Time}: bearing; VaR {self.VaR}")
                w = weight_allocator_bear(pred_signal)
                self.bear = True 
                self.bull = False 
            else:
                self.Debug(f"{self.Time}: normal; VaR {self.VaR}")
                w = weight_allocator(pred_signal)
                self.bull = True 
                self.bear = False 

        trading_weight = w.linear_weight()
        self.weight = trading_weight.iloc[-1,:]
        

    def OnData(self, data: Slice): #activate whenever news come 
        if self.IsWarmingUp: return 
        if self.Time.date()< self.test_start_date: return 

        news = data.Get(TiingoNews) 

        for article in news.Values:
            words = article.Description.lower().split(" ")
            score = sum([self.wordScores[word] for word in words
                if word in self.wordScores])
            
            #1. Get the underlying symbol and save to the variable symbol
            symbol = article.Symbol.Underlying
            
            #2. Add scores to the rolling window associated with its newsData symbol
            self.newsData[symbol].Window.Add(score)
            
            #3. Sum the rolling window scores, save to sentiment
            # If sentiment aggregate score for the time period is greater than 5, emit an up insight
            sentiment = sum(self.newsData[symbol].Window)
            if sentiment > 5:
                self.positive.add(symbol.ID.ToString())
            if sentiment <-5:
                self.negative.add(symbol.ID.ToString())

        if self.today_date is None:
            self.bull_bear_ind()
            self.today_date = self.Time.date()
        else:
            if self.Time.date()> self.today_date:
                self.bull_bear_ind()
                self.today_date = self.Time.date()
                 

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            newsAsset = self.AddData(TiingoNews, symbol,resolution=Resolution.Daily)
            self.newsData[symbol] = NewsData(newsAsset.Symbol)

        for security in changes.RemovedSecurities:
            newsData = self.newsData.pop(security.Symbol, None)
            if newsData is not None:
                self.RemoveSecurity(newsData.Symbol)
    
    def bull_bear_ind(self):
        if self.IsWarmingUp: return
        if self.Time.date()< self.test_start_date: return 
        rsi = self.rsi.Current.Value
        self.Debug(rsi)
        if rsi>self.rsi_max:
            self.bull_bear_list.append((self.Time.date(), 1))
        if rsi<self.rsi_min:
            self.Debug("bear market start")
            self.bull_bear_list.append((self.Time.date(), -1))

    
        self.long_now = self.ema_long.Current.Value
        self.short_now = self.ema_short.Current.Value

        if self.long_prev is None or self.short_prev is None:
            self.long_prev = self.long_now 
            self.short_prev =self.short_now
        else: 
            if (self.short_prev>self.long_prev) and (self.long_now>self.short_now):
                self.DOWN=True 
            if (self.short_prev<self.long_prev) and (self.long_now<self.short_now):
                self.DOWN=False 
            self.long_prev = self.long_now 
            self.short_prev =self.short_now

 

    def riskManagement(self): 
        if self.IsWarmingUp: return
        if self.Time.date()< self.test_start_date: return 
        equity = self.Portfolio.TotalPortfolioValue
        self.trade_today = True
        drawdown_safe = self.ManageDrawdon()

        # Control VaR: volatility at risk 
        self.pnl_history = self.pnl_history.append(pd.Series(equity))
        
        return_series = self.pnl_history.pct_change().dropna()
        self.VaR =return_series.std() *(-2.33)
        varIndicator = (self.VaR  < -0.03)
        if varIndicator:
            self.Liquidate()
            self.Debug('VAR liquidation !!!!!!!!')
        if varIndicator or (not drawdown_safe):
            self.trade_today = False 
            self.yesterday=None 
        

    def adjust_weight_sentiment(self, weight):
        if self.yesterday is not None:
            positive = self.positive-self.positive.union(self.negative)
            negative = self.negative-self.positive.union(self.negative)
            for s in positive:
                if weight.loc[s]<0:
                    if self.yesterday.loc[s]<0:
                        weight[s]= weight.loc[s]/2
                    else:
                        weight[s]=None #do nothing 
            for s in negative:
                if weight.loc[s]>0:
                    if self.yesterday[s]>0:
                        weight[s]= weight.loc[s]/2
                    else:
                        weight[s]=None #do nothing 
        return weight

    def reduce_fee(self,weight):
        if self.yesterday is not None: 
            for symbol in self.symbols: 
                if self.bull and (not self.bear):
                    if (abs(weight.loc[symbol])<0.01):
                        weight[symbol]=None
                    if (not np.isnan(self.yesterday.loc[symbol])) and (not np.isnan(weight.loc[symbol])) : 
                        if abs((weight.loc[symbol]-self.yesterday.loc[symbol])/self.yesterday.loc[symbol])<=0.1:
                            weight[symbol] = None
                if self.bear and (not self.bull):
                    if (not np.isnan(self.yesterday.loc[symbol])) and (not np.isnan(weight.loc[symbol])) : 
                        if abs((weight.loc[symbol]-self.yesterday.loc[symbol])/self.yesterday.loc[symbol])<=0.1:
                            weight[symbol] = None
                      
        else:
            for symbol in self.symbols:
                if self.bull and (not self.bear):
                    if (abs(weight.loc[symbol])<0.01):
                        weight.loc[symbol]= None
        return weight

    def ManageDrawdon(self):
        '''
        True if drawdown under control
        FALSE if liquidated due to exceeding maximum drawdown or maximum duration
        '''
        currentValue = self.Portfolio.TotalPortfolioValue
        self.Debug(self.maxDrawdown_duration)
        if not self.initialised:
            self.portfolioHigh = currentValue   # Set initial portfolio value
            self.initialised = True

        # Update trailing high value if in trailing mode
        if self.isTrailing and self.portfolioHigh < currentValue:
            self.portfolioHigh = currentValue
            self.maxDrawdown_duration = 0 
            return True
        else: 
            self.maxDrawdown_duration +=1 

        pnl = self.GetTotalDrawdownPercent(currentValue)
        if (pnl < self.maximumDrawdownPercent) or (self.maxDrawdown_duration > 120):
            self.initialised = False # reset the trailing high value for restart investing on next rebalcing period
            self.maxDrawdown_duration = 0 
            self.Liquidate()
            return False 
        return True

    def GetTotalDrawdownPercent(self, currentValue):
        return (float(currentValue) / float(self.portfolioHigh)) - 1.0


class NewsData():
    def __init__(self, symbol):
        self.Symbol = symbol
        self.Window = RollingWindow[float](100)  # a rolling windows of last 100 articles.



class CustomFeeModel(FeeModel):
    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def GetOrderFee(self, parameters):
        # custom fee math
        fee = max(1, parameters.Security.Price
                  * parameters.Order.AbsoluteQuantity
                  * 0.00001)
        # self.algorithm.Log(f"CustomFeeModel: {fee}")
        return OrderFee(CashAmount(fee, "USD"))

class CustomSlippageModel:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def GetSlippageApproximation(self, asset, order):
        # custom slippage math
        slippage = asset.Price * 0.0001 * np.log10(2*float(order.AbsoluteQuantity))
        # self.algorithm.Log(f"CustomSlippageModel: {slippage}")
        return slippage
