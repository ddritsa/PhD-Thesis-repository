#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import read_csv
import pandas as pd
import numpy as np

import datetime
from datetime import datetime 
from datetime import timedelta
import time

import os
import glob


# To use this file:
# 
# - We only use the last definition. <b>'create_final_dict'<b> 
#     
# Use this line:
# 
# 'from definitions_for_opening_E4_files_and_making_a_df import create_final_dict'
# 
# Give a name to the df (i.e. all_data_together) and use it like this:
# 
# all_data_together = create_final_dict(this_directory)
# 
# Then you can access each session of the df like this:
# 
# all_data_together[all_data_together['E4_Session']==0]

# definition for making dictionaries for each data stream

# In[2]:


def convert_raw_data_to_dict_EDA(data_stream):
    this_df = pd.DataFrame(data_stream)
    #the first item gives the date
    this_date = this_df[0][0]
    #the second item gives the frequency
    this_frequency = this_df[0][1]
    #convert date to DateTime (in UTC)
    this_date_converted = pd.to_datetime(this_date, unit='s',utc=True)
    this_date_converted = this_date_converted.tz_convert('Australia/Sydney')
    #remove the first 2 items (the date and frequency) and keep all the others
    this_df = this_df[2:]
    
    #create datetime index
    index_list = []
    
    i=0
    for item in this_df.index:
        index_list.append(this_date_converted + timedelta(seconds=i/this_frequency))
        i+=1

    this_df.loc[:, 'Datetime'] = index_list
    this_df.set_index('Datetime', inplace=True)        
    this_df = this_df.sort_index()
    
    return(this_df)
    


# definition for combining the dictionaries of data steams into one

# In[3]:


def read_EDA_csvs(this_path_list):

    EDA_series = read_csv(this_path_list[0], header=None)
    HR_series = read_csv(this_path_list[1], header=None)
    ACC_series = read_csv(this_path_list[2], header=None)
    #BVP_series = read_csv('D:\\UTS\\Data\\EDA data\\sessions for data analysis\\1559010421_A01AB1\\BVP.csv', header=None)
    #IBI_series = read_csv('D:\\UTS\\Data\\EDA data\\1547860514_A01F3B\\IBI.csv', header=None)
    TEMP_series = read_csv(this_path_list[3], header=None)

    raw_EDA = EDA_series.values
    raw_HR = HR_series.values
    raw_ACC = ACC_series.values
    #raw_BVP = BVP_series.values
    #raw_IBI = IBI_series.values
    raw_TEMP = TEMP_series.values
    
    EDA_df = convert_raw_data_to_dict_EDA(raw_EDA)
    HR_df = convert_raw_data_to_dict_EDA(raw_HR)
    ACC_df = convert_raw_data_to_dict_EDA(raw_ACC)
    TEMP_df = convert_raw_data_to_dict_EDA(raw_TEMP)
    

    this_EDA_data_dict = pd.concat([EDA_df,HR_df,ACC_df,TEMP_df], axis=1)
    this_EDA_data_dict.columns = ['EDA','HR','ACC_1','ACC_2','ACC_3','TEMP']
    this_EDA_data_dict = this_EDA_data_dict.fillna(method='backfill')
    this_EDA_data_dict = this_EDA_data_dict.fillna(method='pad')
    return(this_EDA_data_dict)


# parse the E4 files in the defined directory

# In[10]:


def parse_directory(rootDir):

    filelists = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        for item in subdirList:
            #print('---')
            path = os.path.join(rootDir, item )
            csv_files = glob.glob(path + "\*.csv")
            ACC_file,EDA_file,HR_file,TEMP_file=0,0,0,0
            #print(subdirList)
            for file_ in csv_files:
                split_file_ = file_.split('\\')[-2],file_.split('\\')[-1]
                split_file_tag = split_file_[-1].split('.')[-2]
                if 'HR' in split_file_tag:
                    #print('HR:', file_)
                    HR_file = file_
                if 'ACC' in split_file_tag:
                    #print('ACC:', file_)
                    ACC_file = file_
                if ('EDA' in split_file_tag) & ('scrlist' not in split_file_tag):
                    #print('EDA:', file_)
                    EDA_file = file_
                if 'TEMP' in split_file_tag:
                    #print('TEMP:', file_)
                    TEMP_file = file_
                #if 'IBI' in split_file_tag:
                    #print('IBI:', file_)
                    #IBI_file = file_
            file_list = [EDA_file,HR_file,ACC_file,TEMP_file]
            if file_list!=[0,0,0,0]:
                filelists.append(file_list)
    return(filelists)


# In[8]:


def create_final_dict(rootDir):
    filelists = parse_directory(rootDir)
    all_data_df = pd.DataFrame()
    index_list_for_indoor_experiments = []

    E4_session_counter = 0
    start = time.time()
    for item in filelists:
        if 'scrlist' not in item:
            this_EDA_df = read_EDA_csvs(item)
            if 'E4_session' not in this_EDA_df.columns:
                this_EDA_df = this_EDA_df.assign(EDA_session=0)
            #print(this_EDA_df.head(1))
            print(this_EDA_df.index[0])
            index_list_for_indoor_experiments.append(this_EDA_df.index[0])
            #index_list_for_indoor_experiments.append(this_EDA_df.index[-1])
            this_EDA_df['E4_Session'] = E4_session_counter
            all_data_df = all_data_df.append(this_EDA_df)
            E4_session_counter +=1
            #print(all_data_df.index[0])

    end = time.time()
    print((end-start)/60)
    return(all_data_df)


# In[ ]:




