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
from weight_allocator import weight_allocator 



class fitting_model:
    '''
    Use signal to create composite signal using MAX_ICIR
    '''

    def __init__(self, signal_dict, close):
        self.signal_used_to_agg_list =  ['w_percent'
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
                                        ,'mm'
                                        ,'cci'
                                        ,'kdj_d'
                                        ,'position']
       
        self.close = close
        self.factor_return = self.close.pct_change().dropna()
        self.factor_return_backward_1d = self.factor_return.shift(1).dropna()  
        self.factor_return_backward_1d_cum = np.cumprod(1 + self.factor_return_backward_1d)  

        self.signal_df_dict = signal_dict

        self.signal_evaluation = None  # df, index=signal_name, column=signal evaluation result
        self.agg_signal_df_dict = {}  # dict, record composite signal
        self.agg_signal_weight = {}  # dict, record weight to create composite signal
        self.reg_panel_df = None  # panal data 

        
    def evaluate_all(self, evaluate_start_date=None, evaluate_end_date=None):
        '''
        Evaluate all the signal in signal_df (within the specified window) using the following metrics: 
        
        self.signal_evaluation.loc[:, ['IC_mean', 'IR', 'cumulative_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown',
                                               'annualized_turnover', 'direction', 'long_position_mean', 'short_position_mean']] : pd.DataFrame
        index=date
        columns=metric
        only the ones with numerical result is returned (the others are pd.Series)
        '''
        signal_evaluation_dict = {}
        for signal_name in self.signal_df_dict.keys():
            signal_evaluation_dict[signal_name] = data_process.evaluate(self.signal_df_dict[signal_name],
                                                                        self.factor_return,
                                                                        evaluate_start_date=evaluate_start_date,
                                                                        evaluate_end_date=evaluate_end_date)

        self.signal_evaluation = pd.DataFrame(
            signal_evaluation_dict).T.sort_values(by='sharpe_ratio')
        return (self.signal_evaluation.loc[:, ['IC_mean', 'IR', 'cumulative_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown',
                                               'annualized_turnover', 'direction', 'long_position_mean',
                                               'short_position_mean']])
    def generate_reg_panel_df(self):
        '''
        Generate panal dataframe for simple weight calculation in the next step 
        double row index: 1)date->2)factor name 
        col: signal_name 
        '''

        # Combine all signal_df in signal_dict and factor return
        list_df = []
        df = self.factor_return.copy()
        df.loc[:,"date"]=df.index
        df.loc[:, 'signal_name'] = 'y'
        list_df.append(df)
        for signal_name in self.signal_used_to_agg_list:
            df = self.signal_df_dict[signal_name].copy()
            df.loc[:, 'signal_name'] = signal_name  # mark signal name 
            list_df.append(df)
        df = pd.concat(list_df)

        # Group the data by date (this takes a long time) 

        df = df.reset_index().sort_values(
            'date').set_index(['date', 'signal_name'])
        sr1 = pd.Series(df.groupby(by='date'))
        sr2 = sr1.apply(lambda x: x[1])
        sr2.index = sr1.apply(lambda x: x[0]).values

        # delete the cross-sectional date 
        def fun_process(df):
            date = df.index[0][0]
            date_40 = [date for i in df.columns]
            df = df.reset_index().drop('date', axis=1).set_index(
                'signal_name').T
            df = df.set_index([date_40, df.index])
            return (df)

        sr2 = sr2.apply(fun_process)

        # create the panel data 
        reg_panel_df = pd.concat(sr2.values)
        list_columns = list(reg_panel_df.columns)
        list_columns.remove('y') 
        reg_panel_df = reg_panel_df.dropna(subset = list_columns) 
        reg_panel_df.index.names = ['date', 'factor_name'] 
        self.reg_panel_df = reg_panel_df
        return (reg_panel_df)
    
    def aggregate_signal_by_max_ICIR(self, duration=None):
        '''
        Return: composite signal calculated by MAX_ICIR
        Adjust the covariance matrix using LedoitWolf 
        '''
        duration = 120

        IC_dict = {}
        
        self.evaluate_all()
        for signal_name in self.signal_used_to_agg_list:
            ############## Need to shift IC because today's return is used ##############
            IC = self.signal_evaluation.loc[signal_name, 'IC'] # 
            IC.loc[
                pd.to_datetime(data_process.today_str)] = None
            IC = IC.shift(1).dropna()
            IC_dict[signal_name] = IC

        IC_df = pd.DataFrame(IC_dict).dropna()
        IC_df_rolling = pd.DataFrame(IC_df.rolling(
            window=duration,
            min_periods=min(duration, 10)))

        # Calcualte the optimal weight to create the composite signal 
        # Uses markovitz model 
        def fun_weight(x):
            df = x[0]
            if len(df) < min(duration, 10):
                return (None)
            else:
                mean = df.mean()
                cov = LedoitWolf().fit(df).covariance_
                cov_inv = np.linalg.inv(cov)
                weight = cov_inv.dot(df.mean().values)  # (k*k) x (k*1) 
                weight = pd.DataFrame(
                    weight, index=df.columns, columns=[df.index[-1]]).T
                return (weight)

        weight = IC_df_rolling.apply(fun_weight, axis=1)
        print(weight)
        weight = pd.concat(weight.dropna().tolist())

        num = 0  # to represent col 
        # sum up signal accordnig to weight to get composite signal
        for signal_name in IC_df.columns:
            signal_df = self.signal_df_dict[signal_name]
            if num == 0:
                agg_signal_df = signal_df.mul(weight[signal_name], axis=0)
                num = num + 1
            else:
                agg_signal_df = agg_signal_df + \
                                signal_df.mul(weight[signal_name], axis=0)
                num = num + 1

        agg_signal_df = data_process.signal_df_process(agg_signal_df)
        self.agg_signal_df_dict['by_max_ICIR'] = agg_signal_df
        self.agg_signal_weight['by_max_ICIR'] = weight

        return (agg_signal_df)

