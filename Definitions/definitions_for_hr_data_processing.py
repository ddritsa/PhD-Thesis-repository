#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import scipy as sp
import scipy.fftpack
from scipy.signal import savgol_filter


# In[25]:


from definitions_for_scraping_fitbit_data import scrape_movement_data
from definitions_for_speed_data_processing import basic_speed_functions
import matplotlib.pyplot as plt


# In[53]:



# In[27]:


def add_smoother_HR(this_df):
    if 'HR_smoother' not in this_df.columns:
        this_df = this_df.assign(HR_smoother=0)
    hr_smoother = this_df['Heart rate'].copy()
    if len(this_df)>=60:
        hr_smoother = savgol_filter(this_df['Heart rate'].copy(), 35,2)

    this_df.loc[:,'HR_smoother'] = hr_smoother 
    return(this_df)


# In[29]:


def create_firstgradediff_forHR(this_df):
    if 'HR_dif' not in this_df.columns:
        this_df = this_df.assign(HR_dif=0)
    this_df.loc[this_df.index[0], 'HR_dif']=0
    this_df.loc[this_df.index[1]:, 'HR_dif'] = this_df['Heart rate'][1:].diff().fillna(0)/this_df['Time_Passed'][1:]

    return(this_df)


# In[34]:


def find_HR_mean_and_std_thirtyseconds(this_df):
    
    resampled_hr = this_df['Heart rate'].resample('30s').std()
    #resampled_hr_mean = this_df['Heart rate'].resample('30s').mean()

    #len(resampled_hr), len(session_df.loc[resampled_hr.index])
    if 'HR_std' in this_df.columns:
        this_df = this_df.drop('HR_std', axis=1)

    if 'HR_mean_30sec' in this_df.columns:
        this_df = this_df.drop('HR_mean_30sec', axis=1)

    this_df = this_df.assign(HR_std=0)
    this_df = this_df.assign(HR_mean_30sec=0)
    
    
    for i in range(0, len(resampled_hr.index)-1):
        this_index = resampled_hr.index[i]
        next_index = resampled_hr.index[i+1]
        this_df.loc[this_index:next_index,'HR_std'] = this_df.loc[this_index:next_index,'Heart rate'].std()
        this_df.loc[this_index:next_index,'HR_mean_30sec'] = this_df.loc[this_index:next_index,'Heart rate'].mean() 
    
    return(this_df)    
    


# In[72]:


def detect_faulty_hr_firststep(resampled_df,this_df, this_buffer, signal_name, new_col_name, freq):
    this_index_array = np.array(resampled_df[resampled_df[signal_name]==this_buffer].index)
    resampled_df['Integer_index']=np.arange(len(resampled_df))
    integer_index_array = np.array(resampled_df[resampled_df[signal_name]==this_buffer]['Integer_index'])
    time_difference_array = pd.to_timedelta(np.roll(this_index_array, -1) - this_index_array)/np.timedelta64(1, 's')
    break_indices = np.where(time_difference_array>freq*1)
    partitioned_index = np.split(integer_index_array, break_indices[0]+1)
    for item in partitioned_index:
        if len(item)>1:
            sub_df_start = resampled_df[resampled_df['Integer_index']==item[0]]

            sub_df_end = resampled_df[resampled_df['Integer_index']==item[-1]]
            continuous_segment_duration = (sub_df_end.index[0]-sub_df_start.index[0]).total_seconds()
            if continuous_segment_duration>2*freq:
                this_df.loc[sub_df_start.index[0]:sub_df_end.index[0], new_col_name]=1
    return(this_df)


# In[57]:


def detect_faulty_hr_secondstep(main_df,col_name, new_col):

    if new_col not in main_df.columns:
        main_df =main_df.assign(faulty_HR=0)
    main_df[new_col]=0


    resampled = main_df.copy().resample('20s').std()
    corrected_hr = detect_faulty_hr_firststep(resampled, main_df, 0, col_name, new_col, 20)
    if len(corrected_hr[corrected_hr[new_col]==1])>0:
        main_df.loc[corrected_hr.index, new_col]= corrected_hr[new_col]
    if main_df[col_name].mean()==0:
        #print('is zero')
        main_df.loc[main_df.index, new_col]=1
    return(main_df)


# In[79]:


def calculate_intensity_of_HR(this_df,this_age):
    #age = 30 
    maximum_HR = 220 - this_age 

    moderate_bottom = 0.5*maximum_HR
    moderate_upper = 0.7*maximum_HR
    vigorous_bottom = 0.7*maximum_HR
    vigorous_upper = 0.85*maximum_HR
    anaerobic_bottom = 0.85*maximum_HR
    #print(moderate_bottom, moderate_upper,vigorous_bottom, vigorous_upper)

    #the activity listed below is in minutes
    light_activity_time = 0
    moderate_activity_time = 0
    vigorous_activity_time = 0
    anaerobic_activity_time = 0

    light_activity_index = this_df[(this_df['Heart rate']<=moderate_bottom) &(this_df['Time_Passed']<=500)]
    light_activity_time = (this_df[(this_df['Heart rate']<=moderate_bottom) &(this_df['Time_Passed']<=500)]['Time_Passed'].sum())/60
    this_df.loc[light_activity_index.index, 'Light activity']=1

    moderate_activity_time = (this_df[(this_df['Heart rate']>=moderate_bottom) & (this_df['Heart rate']<moderate_upper) &(this_df['Time_Passed']<=500)]['Time_Passed'].sum())/60
    moderate_activity_index = this_df[(this_df['Heart rate']>=moderate_bottom) & (this_df['Heart rate']<moderate_upper) &(this_df['Time_Passed']<=500)]
    this_df.loc[moderate_activity_index.index, 'Moderate activity']=1


    vigorous_activity_time = (this_df[(this_df['Heart rate']>=vigorous_bottom) & (this_df['Heart rate']<vigorous_upper) &(this_df['Time_Passed']<=500)]['Time_Passed'].sum())/60
    vigorous_activity_index = this_df[(this_df['Heart rate']>=vigorous_bottom) & (this_df['Heart rate']<vigorous_upper) &(this_df['Time_Passed']<=500)]
    this_df.loc[vigorous_activity_index.index, 'Vigorous activity']=1

    anaerobic_activity_time = (this_df[(this_df['Heart rate']>=anaerobic_bottom) &(this_df['Time_Passed']<=500)]['Time_Passed'].sum())/60
    anaerobic_activity_index = this_df[(this_df['Heart rate']>=anaerobic_bottom) &(this_df['Time_Passed']<=500)]
    this_df.loc[anaerobic_activity_index.index, 'Anaerobic activity']=1
    this_df = this_df.fillna(0)
    return(this_df)


# In[88]:


'''
How to run the definitions:
session_df = basic_df.copy()
#Find faulty HR
session_df = detect_faulty_hr_secondstep(session_df, 'Heart rate','faulty_HR')
#Compute basic features
session_df = add_smoother_HR(session_df)
session_df = create_firstgradediff_forHR(session_df)
session_df['HR_dif'] = round(session_df['HR_dif'], 2)
session_df = find_HR_mean_and_std_thirtyseconds(session_df)
#Calculate intensity of activity based on age
activity_df = calculate_intensity_of_HR(session_df,30)
# If we want to add peak detection:
from definitions_for_HR_peak_detection import HR_peak_detection

with_peaks = HR_peak_detection(session_df.copy(),10, 'Heart rate')
with_peaks
'''


# In[ ]:




