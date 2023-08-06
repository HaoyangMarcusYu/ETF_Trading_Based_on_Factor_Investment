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


from weight_allocator import weight_allocator

class data_process():
    today_str=(datetime.datetime.utcnow()+datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
    def __init__(self):
        pass 

    @classmethod
    def max_drawdown(cls, df):
        '''
        Calculate maximum drawdown
        '''
        index_end = np.argmax(np.maximum.accumulate(df) / df)
        max_drawdown = 1 - df[index_end] / np.max(df[:index_end])
        return (max_drawdown)

    @classmethod
    def signal_df_process(cls, signal_df):
        '''
        Process data 
        1. drop nan value 
        2. turn into z-score by  x-mean/std
        3. clip off any value >3 or <-3 
        '''
        signal_df = signal_df.dropna()
        signal_df = signal_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)  # 标准化
        signal_df = np.clip(signal_df, -3, 3)  # 缩尾
        return (signal_df)

    @classmethod
    def annualized_turnover(cls, weight):
        '''
        Calculate annualized turnover rate
        '''
        weight_change = (weight - weight.shift(1)).dropna()
        turnover_daily = weight_change.apply(lambda x: np.abs(x).sum(), axis=1)
        num_of_year = len(
            turnover_daily) /252
        annualized_turnover = turnover_daily.sum() / num_of_year / 2
        return (annualized_turnover)

    @classmethod
    def evaluate(cls, signal_df, factor_return, evaluate_start_date=None, evaluate_end_date=None):
        """
        IC : correlation between shift(1) signal with ETFs return
        IR : abs(mean(IC)/std(IC)*np.sqrt(252))
        cumulative_return : value_n/value_1 -1
        annualized_return 
        max_drawdown 
        sharpe_ratio : np.log(1+return).mean()/np.log(1+return).std()*np.sqrt(252))
        df_value : compounded return
        direction : if IR < 0 then -1, else, 1
        portfolio_time_series_return : weight*ETFs return
        annualized_turnover 
        long_position_mean
        short_position_mean 
        """

        signal_df = signal_df.loc[evaluate_start_date:evaluate_end_date, :]
        IC = signal_df.corrwith(factor_return, axis=1).sort_index().dropna()
        IR = float(IC.mean() / IC.std()) * \
             np.sqrt(252)  # Calculate IR
        weight = weight_allocator(signal_df).linear_weight()
        if IR < 0:  
            weight = -weight
            direction = -1
        else:
            direction = 1
        portfolio_time_series_return = (
                weight * factor_return).dropna().sum(axis=1).dropna()  # Daily return
        sharpe_ratio = np.log(1 + portfolio_time_series_return).mean() / \
                       np.log(1 + portfolio_time_series_return).std() * \
                       np.sqrt(252)  
        df_value = np.cumprod(1 + portfolio_time_series_return)  
        portfolio_cum_return = df_value[-1] - 1  
        num_of_year = len(df_value) / 252
        annualized_return = df_value[-1]**(1/num_of_year)-1
        max_drawdown = data_process.max_drawdown(df_value)
        annualized_turnover = data_process.annualized_turnover(
            weight)
        long_position_mean = weight[weight > 0].sum(axis=1).mean()
        short_position_mean = weight[weight < 0].sum(axis=1).mean()

        evaluation_dict = {'IC': IC, 'IC_mean': IC.mean() * direction, 'IR': IR * direction,
                           'cumulative_return': portfolio_cum_return,
                           'annualized_return': annualized_return,
                           'portfolio_time_series_return': portfolio_time_series_return,
                           'max_drawdown': max_drawdown, 'sharpe_ratio': sharpe_ratio, 'value': df_value,
                           'direction': direction, 'annualized_turnover': annualized_turnover,
                           'long_position_mean': long_position_mean, 'short_position_mean': short_position_mean}

        return (evaluation_dict)
    
