#region imports
from AlgorithmImports import *
#endregion
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

class indicator_generator():
    def __init__(self, close, open_p, high, low):
        self.close = close 
        self.open = open_p
        self.high = high
        self.low = low 

        self.factor_return = self.close.pct_change().dropna()
        self.factor_return_backward_1d = self.factor_return.shift(1).dropna()  
        self.factor_return_backward_1d_cum = np.cumprod(1 + self.factor_return_backward_1d)
    
        self.open_return = self.open.pct_change().dropna()
        self.open_return_backward_1d = self.open_return.shift(1).dropna()  
        self.open_return_backward_1d_cum = np.cumprod(1 + self.open_return_backward_1d) 

        self.high_return = self.high.pct_change().dropna()
        self.high_return_backward_1d = self.high_return.shift(1).dropna() 
        self.high_return_backward_1d_cum = np.cumprod(1 + self.high_return_backward_1d) 

        self.low_return = self.low.pct_change().dropna()
        self.low_return_backward_1d = self.low_return.shift(1).dropna()  
        self.low_return_backward_1d_cum = np.cumprod(1 + self.low_return_backward_1d) 

        self.duration_dict={
            'w_percent': 30
            ,'plrc': 20
            ,'auto_correlation':30
            ,'var':80
            ,'inverse_kurt_skew_combination':[134,122]
            ,'sharpe_ratio':130
            ,'cmo':130
            ,'aroon':90
            ,'ema':14
            ,'coppock':[14,13,13]
            ,'hurst':50
            ,'mass_index':[3,20]
            ,'ulcer_index':16
            ,'dbcd':[3,11,11,5] #pick mm
            ,'cci':100
            ,'kdj':[6,10,10] #pick kdj_d
        }

        self.signal_df_dict={}
    
    def calc_w_percent(self):
        '''
        winning percent = close-min/max-min
        the higher the better 
        '''
        d = self.duration_dict['w_percent']
        factor_return_cum = self.factor_return_backward_1d_cum.copy()
        signal_df = factor_return_cum.rolling(window=d).apply(
            lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)))
        
        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['w_percent'] = signal_df
        return (signal_df)
    
    def calc_cci(self):
        d = self.duration_dict['cci']
        factor_return_cum = self.factor_return_backward_1d_cum.copy()
        signal_df = factor_return_cum.apply(lambda x: tb.CCI(x ,x ,x,timeperiod=d),axis=0)
        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['cci'] = signal_df
        return (signal_df)

    def calc_kdj(self):
        '''
        https://zhuanlan.zhihu.com/p/380551571
        d: periods to find KDJ 
        m1: periods for finding K 
        m2: periods for finding D
        Note: only using close price but should have use the min of lowest price and max of highest price 
        '''
        d,m1,m2 = self.duration_dict['kdj']
        factor_return_cum = self.factor_return_backward_1d_cum.copy()
        rsv = factor_return_cum.rolling(window=d).apply(
            lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)))
        K = rsv.apply(lambda x: tb.SMA(x, timeperiod=m1))
        D = K.apply(lambda x: tb.SMA(x,timeperiod=m2))
        J = 3*K-2*D
        K = data_process.signal_df_process(K)
        D = data_process.signal_df_process(D)
        J = data_process.signal_df_process(J)
        # self.signal_df_dict['kdj_k'] = K
        self.signal_df_dict['kdj_d'] = D
        # self.signal_df_dict['kdj_j'] = J
        return (K,D,J)
    
    def calc_dbcd(self):
        '''
        BIAS:=(CLOSE-MA(CLOSE,N))/MA(CLOSE,N);
        DIF:=(BIAS-REF(BIAS,M));
        DBCD:SMA(DIF,T,1);
        MM:MA(DBCD,5);
        '''
        d,m1,m2,m3 = self.duration_dict['dbcd']
        factor_return_cum = self.factor_return_backward_1d_cum.copy() 
        ma_close = factor_return_cum.apply(lambda x: tb.SMA(x,timeperiod = d), axis=0) 
        bias = (factor_return_cum-ma_close)/ma_close 
        dif = bias-bias.shift(m1)
        dbcd = dif.apply(lambda x: tb.SMA(x, timeperiod = m2),axis=0)
        mm = dbcd.apply(lambda x: tb.SMA(x, timeperiod=m3),axis=0)
        dbcd = data_process.signal_df_process(dbcd)
        mm = data_process.signal_df_process(mm)
        # self.signal_df_dict['dbcd']=dbcd
        self.signal_df_dict['mm']=mm
        return (dbcd,mm)

    def calc_ulcer_index(self):
        d = self.duration_dict['ulcer_index']
        factor_return_cum = self.factor_return_backward_1d_cum.copy()
        signal_df = factor_return_cum.apply(lambda x: ta.volatility.ulcer_index(x, window=d),axis=0)
        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['ulcer_index']=signal_df
        return (signal_df)
    
    def calc_mass_index(self):
        d,m = self.duration_dict['mass_index']
        diff = self.high-self.low
        single_ema = diff.apply(lambda x: tb.EMA(x,timeperiod=d),axis=0)
        double_ema = single_ema.apply(lambda x: tb.EMA(x,timeperiod=d),axis=0)
        ratio = single_ema/double_ema
        mass_index = ratio.rolling(window=m).apply(lambda x: sum(x))
        mass_index = data_process.signal_df_process(mass_index)
        self.signal_df_dict['mass_index']=mass_index
        return (mass_index)
    
    def calc_hurst(self,d=20):
        d = self.duration_dict['hurst']
        def hurst(price, min_lag=2, max_lag=18):
            lags = np.arange(min_lag, max_lag + 1)
            tau = [np.std(np.subtract(price.values[lag:], price.values[:-lag])) 
                for lag in lags]
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            return m[0]
        factor_return_cum = self.factor_return_backward_1d_cum.copy()
        signal_df = factor_return_cum.rolling(window=d).apply(
            lambda x: hurst(x))
        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['hurst'] = signal_df
        return (signal_df)        
        
    def calc_coppock(self):
        '''
        https://medium.com/codex/the-coppock-curve-coding-and-backtesting-a-trading-strategy-in-python-8dac8bbe3c3f
        '''
        m1,m2,m3 = self.duration_dict['coppock']
        factor_return_cum = self.factor_return_backward_1d_cum.copy()
        M = factor_return_cum.diff(m1)
        N = factor_return_cum.shift(m1)
        long_ROC = M / N
        M = factor_return_cum.diff(m2)
        N =factor_return_cum.shift(m2)
        short_ROC = M / N
        sum_ROC = long_ROC + short_ROC
        Copp = sum_ROC.apply(lambda x: tb.WMA(x,timeperiod=m3),axis=0)
        Copp = data_process.signal_df_process(Copp)
        self.signal_df_dict['coppock'] = Copp
        return (Copp)

    def calc_ema(self):
        # Exponential Moving Average
        d = self.duration_dict['ema']
        signal_df = self.factor_return_backward_1d.apply(lambda x: tb.EMA(x,timeperiod=d),axis=0)
        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['ema'] = signal_df
        return (signal_df)

    def calc_aroon(self):
        d = self.duration_dict['aroon']
        df = np.cumprod(1 + self.factor_return.copy())
        min_periods_1d =10
        signal_df = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(min_periods_1d, len(df)):
            start_index = max(i - d, 0)
            window = df.iloc[start_index:i, :].reset_index()  

            row_max = window.apply(lambda x: list(x).index(x.max()))  
            row_min = window.apply(lambda x: list(x).index(x.min())) 
            up = (len(window) - (window.index[-1] - row_max)) / len(window)
            down = (len(window) - (window.index[-1] - row_min)) / len(window)
            signal_df.iloc[i, :] = (up - down)[1:] 

        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['aroon'] = signal_df
        return (signal_df)
    
    def calc_cmo(self):
            '''
            CHANDE MOMENTUM OSCILLATOR
            '''
            d = self.duration_dict['cmo']
            df = self.factor_return_backward_1d.copy()
            min_periods_1d = 10

            df1 = df[df > 0].fillna(0)
            df2 = abs(df[df < 0].fillna(0))

            Su = df1.rolling(window=d, min_periods=min_periods_1d).sum()
            Sd = df2.rolling(window=d, min_periods=min_periods_1d).sum()
            signal_df = (Su - Sd) / (Su + Sd)

            signal_df = data_process.signal_df_process(signal_df)
            self.signal_df_dict['cmo'] = signal_df
            return (signal_df)
    
        
    def calc_sharpe_ratio(self):
        '''
        Used two periods for mean and std 
        the higher the better 
        '''
        d = self.duration_dict['sharpe_ratio']
        mean = np.log(1 + self.factor_return_backward_1d).rolling(window=d).mean()
        std = np.log(1 + self.factor_return_backward_1d).rolling(window=d).std()
        signal_df = mean / std 

        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['sharpe_ratio'] = signal_df 
        return (signal_df)
    
    def calc_kurt_skew_combination(self):
        '''
        the lower the better 
        '''
        m1,m2= self.duration_dict['inverse_kurt_skew_combination']
        kurt_weight=0.5
        signal_df_kurt = self.factor_return_backward_1d.rolling(
            window=m1).apply(lambda x: stats.kurtosis(x))

        signal_df_skew = self.factor_return_backward_1d.rolling(
            window=m2).apply(lambda x: stats.skew(x))

        signal_df = signal_df_kurt * kurt_weight + signal_df_skew * (1 - kurt_weight)

        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['kurt_skew_combination'] = signal_df
        return (signal_df)
        
    def calc_inverse_kurt_skew_combination(self):
        try:
            kurt_skew_combination = self.signal_df_dict['kurt_skew_combination']
        except KeyError:
            kurt_skew_combination = self.calc_kurt_skew_combination()
            
        signal_df = -kurt_skew_combination
        self.signal_df_dict['inverse_kurt_skew_combination'] = signal_df
        return (signal_df)

    def calc_var(self,var_dim=1):
        '''
        Predict next day return using Vector Autoregression model 
        '''
        d = self.duration_dict['var']
        df = self.factor_return.copy()
        signal_df = pd.DataFrame(columns=df.columns, index=df.index)

        n = var_dim
        window_len = d
        for i in range(2 + n, len(df)):
            # retrieve training data 
            start_index = max(i - window_len, 0)
            train = df.iloc[start_index:i, :]
            model = VAR(train)
            model_fitted = model.fit(n)
            # predict return @t based on t-1 returns 
            test = df.iloc[i - n:i, :]
            forecast = model_fitted.forecast(y=test.values, steps=1)
            signal_df.iloc[i, :] = forecast

        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['var'] = signal_df
        return (signal_df)

   
    def calc_auto_correlation(self):
        '''
        AR
        ''' 
        d= self.duration_dict['auto_correlation']

        signal_df = self.factor_return_backward_1d.rolling(window=d).apply(lambda x: smt.stattools.acf(x)[1])

        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['auto_correlation'] = signal_df
        return (signal_df)
    
    def calc_plrc(self, d=20):
        # PLRC (close / mean(close)) = beta * t + alpha
        d= self.duration_dict['plrc']
        df = self.factor_return_backward_1d_cum.copy()

        def regress(window):
            y = window / window.mean()
            X = [x + 1 for x in range(len(window))]
            model = LinearRegression()
            model.fit(np.reshape(X, (-1, 1)), y)  # use LR to calculate alpha and beta
            beta = model.coef_
            return (beta)

        signal_df = df.rolling(window=d).apply(regress)
        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['plrc'] = signal_df
        return (signal_df)

    def calc_position(self, d = 30):
        df = self.factor_return_backward_1d_cum.copy()

        def quantile(window):
            sort_window = sorted(list(window), reverse=True)
            position = sort_window.index(window[-1]) / len(sort_window)
            return position

        signal_df = df.rolling(window=d).apply(quantile)
        signal_df = data_process.signal_df_process(signal_df)
        self.signal_df_dict['position'] = signal_df
        return (signal_df)

    def run_all(self):
        self.signal_used_to_agg_list = ['w_percent'
                                ,'plrc'
                                ,'auto_correlation'
                                ,'var'
                                ,'inverse_kurt_skew_combination'
                                ,'sharpe_ratio'
                                ,'cmo'
                                ,'aroon'
                                ,'ema'
                                ,'coppock'
                                ,'hurst'
                                ,'mass_index'
                                ,'ulcer_index'
                                ,'dbcd'
                                ,'cci'
                                ,'kdj'
                                ,'position']

        for signal in self.signal_used_to_agg_list:
            exec(f"self.calc_{signal}()")
        return self.signal_df_dict
