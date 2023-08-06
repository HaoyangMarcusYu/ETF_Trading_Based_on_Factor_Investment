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


class weight_allocator_bear:
    '''Use signal to generate portfolio weight'''

    def __init__(self, signal_df):

        self.signal_df = signal_df
        self.market_symbol = ['SPY R735QTJ8XC9X',
                            'BND TRO5ZARLX6JP']
        self.style_symbol=['USMV V0WRDXSSH205',
                            'SIZE VFUDGZIY8ZMT',
                            'MTUM VFUDGZIY8ZMT',
                            'IWF RV0PWMLXVHPH',
                            'IWD RV0PWMLXVHPH',
                            'IVE RV0PWMLXVHPH',
                            'DVY STHD6FIMA3XH',
                            'RSP SO9ASKNDO09X',
                            'QUAL VIBZ5HTB7N8L',
                            'TIP SU8XP1RCF8F9']
        # ['IWF RV0PWMLXVHPH',
        #                     'PIV TE9OXHIWFHWL',
        #                     'IWD RV0PWMLXVHPH',
        #                     'IVE RV0PWMLXVHPH',
        #                     'DVY STHD6FIMA3XH',
        #                     'RSP SO9ASKNDO09X',
        #                     'DWM TJIQRFRB4CPX']
        self.industry_symbol=['DBA TP2MIF0KNIAT',
                            'KBE TDP0JIUCTNJ9',
                            'IBB S23QOUCNOW9X',
                            'ITB TIDDZIE7WNZ9',
                            'ICLN U3RDKMG7QNHH',
                            'XLP RGRPZX100F39',
                            'XLY RGRPZX100F39',
                            'ITA TIDDZIE7WNZ9',
                            'XLE RGRPZX100F39',
                            'PBS T9Q8PYSVQ99H',
                            'XLF RGRPZX100F39',
                            'IYK RVLEALAHHC2T',
                            'XLV RGRPZX100F39',
                            'PEJ T9Q8PYSVQ99H',
                            'KIE TDP0JIUCTNJ9',
                            'QQQ RIWIV7K5Z9LX',
                            'XLI RGRPZX100F39',
                            'XLB RGRPZX100F39',
                            'XME TJONFZYBKVOL',
                            'IHE TIDDZIE7WNZ9',
                            'XRT TJONFZYBKVOL',
                            'SMH V2LT3QH97TYD',
                            'SLX TMUVAUDWGEAT',
                            'XLK RGRPZX100F39',
                            'IYT SSPSNT9W4ZFP',
                            'XLU RGRPZX100F39']

    def linear_weight(self):
        '''
        Post process the portfolio weight determined by the composite signal 
        '''

        # allow more freedom on the short side when in bear market 
        weight = 2 * \
            self.signal_df.div(self.signal_df.apply(abs).sum(axis=1), axis=0)
        weight.loc[:, self.style_symbol] = np.clip(
            weight.loc[:, self.style_symbol], -0.1, 0.05) 
        
        weight.loc[:, self.industry_symbol] = np.clip(
            weight.loc[:, self.industry_symbol], -0.05, 0.025) 
        
        # weight.loc[weight[self.market_symbol]<0]=0
        weight.loc[:, self.market_symbol] = abs(weight.loc[:, self.market_symbol])

        weight.loc[:, self.market_symbol] = np.clip(
            weight.loc[:, self.market_symbol], -0.1, 0.05) 
        weight = 0.5*weight.div(weight.apply(abs).sum(axis=1), axis=0)
        return (weight)
        

'''
market neutral
'''
# def make_market_neutral(weights):
#         long_weights = abs(weights[weights > 0].sum())
#         short_weights = abs(weights[weights < 0].sum())
#         if long_weights>short_weights:
#                 adjustment =short_weights/long_weights
#                 weights[weights > 0] *= adjustment
#         else: 
#                 adjustment =long_weights/short_weights
#                 weights[weights < 0] *= adjustment
#         return weights

# weight = weight.apply(make_market_neutral, axis=1)


'''
with some short 
'''
         # weight = 2 * \
        #     self.signal_df.div(self.signal_df.apply(abs).sum(axis=1), axis=0)
        # weight.loc[:, self.style_symbol] = 2 * weight.loc[:, self.style_symbol]
        # weight.loc[:, self.style_symbol] = np.clip(
        #     weight.loc[:, self.style_symbol], -0.05, 0.1)  # bound style factor within [-0.05,+0.1]
        
        # weight.loc[:, self.industry_symbol] = np.clip(
        #     weight.loc[:, self.industry_symbol], -0.0075, 0.015)  # bound industry factor within [-0.0075,+0.015]
        
        # # weight.loc[weight[self.market_symbol]<0]=0
        # weight.loc[:, self.market_symbol] = abs(weight.loc[:, self.market_symbol])

        # weight.loc[:, self.market_symbol] = np.clip(
        #     weight.loc[:, self.market_symbol], -0.05, 0.1)  # bound market factor within [-0.05,+0.1]
'''
Best weight
'''
        # weight = 6 * \
        #     self.signal_df.div(self.signal_df.apply(abs).sum(axis=1), axis=0)
        # weight.loc[:, self.style_symbol] = 2 * weight.loc[:, self.style_symbol]
        # for s in self.style_symbol:
        #     weight.loc[weight.loc[:,s]<-0.1,s] = -0.1
        #     weight.loc[weight.loc[:,s]>=-0.1,s]=np.clip(weight.loc[weight.loc[:,s]>=-0.1,s], 0, 0.2)  # bound style factor within [0,+0.20]

        # for s in self.industry_symbol:
        #     weight.loc[weight.loc[:,s]<-0.015,s]=-0.015
        #     weight.loc[weight.loc[:,s]>=-0.015,s]=np.clip(weight.loc[weight.loc[:,s]>=-0.015,s], 0, 0.03)  # bound industry factor within [0,+0.03]

        # for s in self.market_symbol:
        #     weight.loc[weight.loc[:,s]< -0.1,s]= -0.1
        #     weight.loc[weight.loc[:,s]>= -0.1,s]=np.clip(weight.loc[weight.loc[:,s]>=-0.1,s], 0, 0.2)  # bound market factor within [0,+0.20]
        # return (weight)
'''
Long only
'''
        # weight = 6 * \
        #     self.signal_df.div(self.signal_df.apply(abs).sum(axis=1), axis=0)
        # weight.loc[:, self.style_symbol] = 2 * weight.loc[:, self.style_symbol]
        # weight.loc[:, self.style_symbol] = np.clip(
        #     weight.loc[:, self.style_symbol], 0, 0.2)  # bound style factor within [0,+0.20]
        
        # weight.loc[:, self.industry_symbol] = np.clip(
        #     weight.loc[:, self.industry_symbol], 0, 0.03)  # bound industry factor within [0,+0.03]
        
        # # weight.loc[weight[self.market_symbol]<0]=0
        # # weight.loc[:, self.market_symbol] = abs(weight.loc[:, self.market_symbol])

        # weight.loc[:, self.market_symbol] = np.clip(
        #     weight.loc[:, self.market_symbol], 0, 0.2)  # bound market factor within [0,+0.20]
        
'''
Old
'''

        # weight.loc[:, self.style_symbol] = np.clip(
        #     weight.loc[:, self.style_symbol], -0.2, 0.2)  # bound style factor within [-0.20,+0.20]
        
        # weight.loc[:, self.industry_symbol] = np.clip(
        #     weight.loc[:, self.industry_symbol], -0.03, 0.03)  # bound indsutry factor within [-0.03,0.03]
        
        # # weight.loc[weight[self.market_symbol]<0]=0
        # weight.loc[:, self.market_symbol] = abs(weight.loc[:, self.market_symbol])

        # weight.loc[:, self.market_symbol] = np.clip(
        #     weight.loc[:, self.market_symbol], -0.2, 0.2)  # bound market factor within [-0.20,+0.20]
        
       
    
