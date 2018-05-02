# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:01:21 2018

@author: Topher
"""

from pandas_datareader import data

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


def get_adjclose( tickers, start_date, end_date, data_source='quandl'):
    # Use pandas_reader.data.DataReader to load the desired data. As simple as that.

    adj_close = pd.DataFrame()
    query_list = [ ticker for ticker in tickers]
    for iter in range(len(tickers)):
        # Use pandas_reader.data.DataReader to load the desired data. As simple as that.
        panel_data = data.DataReader(query_list[iter], data_source, start_date, end_date)
    
        # Getting just the adjusted closing prices. This will return a Pandas DataFrame
        # The index in this DataFrame is the major index of the panel_data.
        adj_close[tickers[iter]] = panel_data.AdjClose

    #yahoo no longer works
    #panel_data = data.DataReader(tickers, data_source, start_date, end_date)
    
    # Getting just the adjusted closing prices. This will return a Pandas DataFrame
    # The index in this DataFrame is the major index of the panel_data.
    #adj_close = panel_data.loc['Adj Close']

    # Yahoo may be missing some weekdays due to holidays or whatever, so fill in 
    # the missing dates:
    
    # Getting all weekdays between 01/01/2000 and 12/31/2016
    all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # How do we align the existing prices in adj_close with our new set of dates?
    # All we need to do is reindex adj_close using all_weekdays as the new index
    adj_close = adj_close.reindex(all_weekdays)
    
    # Reindexing will insert missing values (NaN) for the dates that were not present
    # in the original set. To cope with this, we can fill the missing by replacing them
    # with the latest available price for each instrument.
    adj_close = adj_close.fillna(method='ffill')
    adj_close = adj_close.dropna()

    return adj_close




#from datetime import date
#ok so let's use the ema price at the end of each month for the past 12 months
#thisDate = date(2010,01,29) # the date for the price we're trying to predict
#dates_eom = pd.date_range(end=thisDate, periods=13, freq='BM') #last day of month
#dates_bom = pd.date_range(end=thisDate, periods=13, freq='BMS') #first day of month

#features = np.log10(ema_short.loc[dates_eom[:-1], :] / ema_short.loc[dates_bom[:-1], :].shift(1,freq='BM')) #features are the first 12 values

#yvals = features.iloc[-1, :] #last change is the thing we're trying to predict!

#ok let's build a whole set of training examples going back to 2006

def get_features( prices, prices_y, start_date, end_date, num_months=13):
    
    #1D array of all the dates we wanna predict. start num_months after earliest data point for complete feature set for first training example
    dates_eom_all = pd.date_range(start = pd.to_datetime(start_date) + relativedelta(months=num_months+1), end=end_date, freq='BM') #last day of month
    print(dates_eom_all)
    num_tickers = prices.shape[1]
    num_dates = len(dates_eom_all)
    
    num_extra_features = 1 #for the ticker ID
    
    num_features = num_months + num_extra_features
    
    num_examples = num_dates * num_tickers
    
    features = np.zeros([num_dates, num_tickers, num_features])
    
    yvals = np.zeros([num_dates, num_tickers])
    
    for iter in range(num_dates):
        
        thisDate = dates_eom_all[iter] # the date for the price we're trying to predict
        dates_eom = pd.date_range(end=thisDate, periods=num_months+1, freq='BM') #last day of month
        dates_bom = pd.date_range(end=thisDate, periods=num_months+1, freq='BMS') #first day of month
        these_features = np.log10(prices.loc[dates_eom, :] / prices.loc[dates_bom, :].shift(1,freq='BM')).T 
        
        #features are the first num_months values
        features[iter,:,0:num_months] = these_features.iloc[:,:-1] #features are the first num_month values

        #try to predict a different feature, y
        these_features = np.log10(prices_y.loc[dates_eom, :] / prices_y.loc[dates_bom, :].shift(1,freq='BM')).T #features are the first 12 values

        yvals[iter,:] = these_features.iloc[:,-1] #last change is the thing we're trying to predict!

    #add information about which ticker it is
    features[:,:,-1] =  np.tile( np.arange(num_tickers)/(num_tickers-1.), [num_dates,1])
    
    #combine all the tickers together
    features = features.reshape([num_examples, num_features])
    yvals = yvals.reshape(num_examples)
        
    #get rid of any examples with NaNs
    these = np.where(np.isfinite(yvals) & np.isfinite( np.sum(features,axis=1)))[0]
    yvals = yvals[these]
    features = features[these,:]
    
    #discretize yvals
    #yvals = np.round(yvals*100.)
    #yvals -= np.amin(yvals)
    
#    #yvals[]
#    if classifier:
#        limits = [-np.inf,0.01, np.inf]
#        yvals_new = yvals.copy()
#        for iter in range(len(limits)-1):
#            yvals_new[(yvals > limits[iter]) & (yvals < limits[iter+1]) ] = iter
#        yvals = yvals_new.copy()
#    #plt.hist(yvals_new)
#    
    return features, yvals

