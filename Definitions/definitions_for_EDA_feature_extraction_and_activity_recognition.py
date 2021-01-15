#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random

import datetime
from datetime import datetime 
from datetime import timedelta
import time

from scipy.signal import savgol_filter

from definitions_for_artifact_recognition import label_artifact_data

from definitions_for_activity_recognition import label_accelerometer_data, detect_change_of_intensity


# In[ ]:


def resample_EDA_df(df):
    # 'df_for_EDA_analysis_withACC' replaces 'resampled'
    EDA_df_resample = df.resample('250ms').mean().copy()
    EDA_df_resample = EDA_df_resample.fillna(method='backfill')
    # 'df_for_EDA_analysis' replaces 'test_df'
    #df_for_EDA_analysis = resampled.drop(columns=['ACC_1','ACC_2','ACC_3']).copy()
    return(EDA_df_resample)


# In[ ]:


# Create a list with EDA data points going up and down (based on the difference)
def parse_EDA_data_points(df):

    integer_arr = np.arange(len(df))
    df['Integer_index']=integer_arr
    df['peak_rise']=0
    df['peak_fall']=0

    lower_cutoff = 0.00001
    upper_cutoff = 0.6
    df = df.assign(peak_rise=0)
    df.loc[df[(df['EDA'].diff()>lower_cutoff) & (df['EDA'].diff()<upper_cutoff)].index,'peak_rise'] = 1
    df = df.assign(peak_fall=0)
    df.loc[df[(df['EDA'].diff()< (-1*lower_cutoff)) & (df['EDA'].diff()> (-1*upper_cutoff))].index,'peak_fall'] = 1
    df


    EDR_amplitudes_up = []
    EDR_amplitudes_down = []
    start=time.time()
    EDA_df_index = np.array(df.index)
    continuous_peaks_array = np.array(df[df['peak_rise']==1].index)
    continuous_falls_array = np.array(df[df['peak_fall']==1].index)
    integer_index_array_peaks = np.array(df[df['peak_rise']==1]['Integer_index'])
    integer_index_array_falls = np.array(df[df['peak_fall']==1]['Integer_index'])
    corrected_peak_array = np.array(df['peak_rise'])
    corrected_fall_array = np.array(df['peak_fall'])
    EDR_array=np.array(df['EDA'])
    continuous_peaks_indices = pd.to_timedelta(np.roll(continuous_peaks_array, -1) - continuous_peaks_array)/np.timedelta64(1, 's')
    continuous_falls_indices = pd.to_timedelta(np.roll(continuous_falls_array, -1) - continuous_falls_array)/np.timedelta64(1, 's')
    break_indices_peaks = np.where(continuous_peaks_indices>2*0.25)
    break_indices_falls = np.where(continuous_falls_indices>2*0.25)
    partitioned_index_riseups = np.split(integer_index_array_peaks, break_indices_peaks[0]+1)
    partitioned_index_falldowns = np.split(integer_index_array_falls, break_indices_falls[0]+1)


    rise_ups_list=[]
    for item in partitioned_index_riseups:
        if len(item)==1:
            corrected_peak_array[item[0]]=0
        else:
            EDR_amplitude = EDR_array[item[-1]]-EDR_array[item[0]]
            if ((item[-1]+7)<len(corrected_fall_array))& (1 not in corrected_fall_array[item[-1]:item[-1]+7]):

                corrected_peak_array[item[0]:item[-1]]=0
            else:
                    #only then it is a valid rise up list
                rise_ups_list.append(item.tolist())

    fall_downs_list=[]
    for item in partitioned_index_falldowns:
        if len(item)==1:
            corrected_fall_array[item[0]]=0
        else:
            EDR_amplitude = EDR_array[item[0]]-EDR_array[item[-1]]
            if ((item[0]-4)>0) & (1 not in corrected_peak_array[item[0]-4:item[0]]):
                corrected_fall_array[item[0]:item[-1]]=0
            else:
                    #only then it is a valid rise up list
                if len(item)>9:
                    fall_downs_list.append(item[:9].tolist())
                else:
                    fall_downs_list.append(item.tolist())
    #We have two lists, one with points going up and one with points going down
    #Now match the series of points going up with the series of points going down
    mega_list = np.array(sorted(fall_downs_list+rise_ups_list))
    peaksandfalls = np.zeros((len(mega_list),2))
    for i in range(0, len(mega_list[:-2])):
        if ((mega_list[i+1][0]-mega_list[i][-1])<8*0.25):
            if (corrected_peak_array[mega_list[i][0]]==1) & (corrected_fall_array[mega_list[i+1][0]]==1) :
                peaksandfalls[i]= [mega_list[i][0],mega_list[i+1][-1]]
    corrected_peaksandfalls = peaksandfalls[np.nonzero(peaksandfalls[:,1])]
    
    return(corrected_peaksandfalls, EDR_array)


# In[2]:


#find and exclude artifacts 
# This creates a copy of the df, with a new column 'EDA artifact'
# If 'EDA artifact'=0, there is an artifact, otherwise it is 1
# The model is trained at 250ms sampling (1/4 second, the typical E4 sampling)
def find_EDA_artifacts(df):
    df_with_artfs = label_artifact_data(df.copy())
    return(df_with_artfs)
    #test_df['EDA artifact']=df_with_artifacts['EDA artifact'].copy()
    


# In[ ]:


#Now check if each of the peaks satifies the criteria (amplitude>0.05)
#and if it is an artifact based on the assessment

#create columns in the EDA df 

def create_EDA_features(df):
    
    identify_peaks = parse_EDA_data_points(df)
    list_with_identified_peaks = identify_peaks[0]
    EDR_array = identify_peaks[1]
    df_with_artifacts = find_EDA_artifacts(df)
    
    df['EDA artifact']=df_with_artifacts['EDA artifact'].copy()
    df.loc[:,'EDA_peak_fall']=0
    df.loc[:,'EDA_peak_rise']=0
    df.loc[:,'Stress']=0
    df['TEMP_smoothed']=savgol_filter(df['TEMP'].copy(),51,3)
    df = df.assign(EDA_session=0)
    df.loc[:,'EDA_session']=0
    df['EDA_tonic']=df['EDA'].copy()

    #create helper arrays
    EDA_tonic_array=np.array(df['EDA_tonic'])
    corrected_peak_rise_array=np.zeros(len(df))
    corrected_peak_fall_array=np.zeros(len(df))
    EDR_amplitude_array=np.zeros(len(df))
    EDA_duration_array=np.zeros(len(df))
    EDA_session_array=np.zeros(len(df))
    TEMP_array=np.array(df['TEMP_smoothed'])
    stress_array=np.zeros(len(df))
    artifact_array = np.array(df['EDA artifact'])
    this_E4_session = df.loc[df.index[0],'E4_Session']

    for item in list_with_identified_peaks:
        #print('---')
        up = int(item[0])
        down = int(item[1])
        #print(up,down)
        max_index = np.argmax(EDR_array[up:down])+up
        #print('up:',up,'max',max_index,'down',down)
        EDA_df_index = np.array(df.index)
        EDR_amplitude = np.max(EDR_array[up:down]) - EDR_array[up]
        #print('amplitude:',EDR_amplitude)
        EDA_rise_duration = (EDA_df_index[down]-EDA_df_index[up])/np.timedelta64(1, 's')
        #print('rise duration', EDA_rise_duration)
        #assign the values of tonic EDA
        EDA_tonic_array[up]=EDR_array[up]
        EDA_tonic_array[down]=EDR_array[down]
        EDA_tonic_array[up:down]=np.linspace(EDR_array[up],EDR_array[down], down-up) 


        #TO EXCLUDE ACTIVITY DO THE FOLLOWING:
        #if (EDR_amplitude>=0.05) & (EDA_rise_duration>=0.05) & (3 not in activity_data_array[up:max_index]):
        if (EDR_amplitude>=0.05) & (EDA_rise_duration>0.02) & (0 not in artifact_array[up:down]):
            #print('valid')
            corrected_peak_rise_array[up:max_index]=1
            corrected_peak_fall_array[up:max_index]=0
            corrected_peak_rise_array[max_index:down+1]=0
            corrected_peak_fall_array[max_index:down+1]=1

            EDR_amplitude_array[up:down+1]=EDR_amplitude
            EDA_duration_array[up:down+1]=EDA_rise_duration
            this_EDA_session = random.randint(1,100000)
            if this_EDA_session in np.unique(EDA_session_array):
                print('is in')
                while this_EDA_session in np.unique(EDA_session_array):
                    this_EDA_session = random.randint(1,len(list_with_identified_peaks) + 1000000000)
            EDA_session_array[up:down+1]=this_EDA_session

            #find where the temperature goes up and where down
            this_temp = TEMP_array[up:down]
            #print('TEMP:', this_temp, np.diff(this_temp), np.where(np.diff(this_temp)<0),'NUM:',len(np.where(np.diff(this_temp)<0)[0]))
            if len(np.where(np.diff(this_temp)<0)[0])>0:
                #print('stress')
                stress_array[up:down+1]=1
            #if len(eda_df_subset[eda_df_subset['TEMP_smoothed'].diff()<0])>0:
                #test_df.loc[up[0]:down[-1]+timedelta(seconds= 1* 0.25), 'Stress'] = 1
            '-----'
    df['EDA_tonic']=EDA_tonic_array
    df['EDR_amplitude']=EDR_amplitude_array
    df['EDA_duration']=EDA_duration_array
    df['EDA_peak_rise']=corrected_peak_rise_array
    df['EDA_peak_fall']=corrected_peak_fall_array
    df['Stress']=stress_array
    df['EDA_session'] = EDA_session_array

    if 'EDR_amplitude' not in df.columns:
        df['EDR_amplitude']=0
    if 'EDA_duration' not in df.columns:
        df['EDA_duration']=0
    df['EDR_amplitude'] = df['EDR_amplitude'].fillna(0)
    df['EDA_duration'] = df['EDA_duration'].fillna(0)

    # smooth EDA tonic 
    if 'EDA_tonic' not in df.columns:
        df['EDA_tonic']= df['EDA']


    #Boucsein recommends sampling between 10 and 30s
    resampled_tonic_EDA = df['EDA_tonic'].resample('20s').mean().interpolate(method='spline', order=3)
    #resampled_tonic_EDA = test_df['EDA_tonic'].resample('10s').mean()
    df['EDA_tonic'] = resampled_tonic_EDA.fillna(method='pad')
    #test_df['EDA_tonic'] = test_df['EDA_tonic'].fillna(method='ffill')
    if ((df.index[-1]-df.index[0])/np.timedelta64(60,'s'))>=5:
        df['EDA_tonic'] = df['EDA_tonic'].interpolate(method='cubic')
    #test_df['EDA_tonic'] = savgol_filter(test_df['EDA_tonic'].copy(),51,3)
    
    
    #make sure that there is no EDR in artifact
    for item in df[(df['EDR_amplitude']>0)&(df['EDA artifact']==0)]['EDA_session'].unique():
        print('art')
        subdf = df[df['EDA_session']==item]
        df.loc[subdf.index,'EDR_amplitude']=0
        df.loc[subdf.index,'EDA_duration']=0
        df.loc[subdf.index,'EDA_peak_rise']=0
        df.loc[subdf.index,'EDA_peak_rise']=0
        df.loc[subdf.index,'EDA_peak_fall']=0
        df.loc[subdf.index,'Stress']=0

    #Calculate percentage of artifacts in session
    print('percentage of artifacts:', len(df[df['EDA artifact']==0]),len(df), len(df[df['EDA artifact']==0])/len(df))

    if 'percentage_of_EDA_artifacts' not in df.columns:
        df = df.assign(percentage_of_EDA_artifacts=0)
    df['percentage_of_EDA_artifacts']=len(df[df['EDA artifact']==0])/len(df)

    
    return(df)


# In[ ]:


def detect_activity(df):
    
    accelerometer_df = label_accelerometer_data(df.copy(), 40)
    accelerometer_df_2 = label_accelerometer_data(df.copy(), 8)

    accelerometer_df['Detected Activity_2s']=accelerometer_df_2['Detected Activity']

    if 'main_activity' not in accelerometer_df.columns:
        accelerometer_df = accelerometer_df.assign(main_activity=0)

    accelerometer_df.loc[accelerometer_df[accelerometer_df['Detected Activity']==2].index, 'main_activity']='walking'
    accelerometer_df.loc[accelerometer_df[accelerometer_df['Detected Activity']==0].index, 'main_activity']='movement'
    accelerometer_df.loc[accelerometer_df[accelerometer_df['Detected Activity']==1].index, 'main_activity']='still'

    if 'main_activity_2s' not in accelerometer_df.columns:
        accelerometer_df = accelerometer_df.assign(main_activity_2s=0)


    accelerometer_df.loc[accelerometer_df[accelerometer_df['Detected Activity_2s']==2].index, 'main_activity_2s']='walking'
    accelerometer_df.loc[accelerometer_df[accelerometer_df['Detected Activity_2s']==0].index, 'main_activity_2s']='movement'
    accelerometer_df.loc[accelerometer_df[accelerometer_df['Detected Activity_2s']==1].index, 'main_activity_2s']='still'

    cleanup_nums = {"still": 1, "walking": 2, "movement":3}
    categorical_data = accelerometer_df['main_activity'].copy()
    categorical_data.replace(cleanup_nums, inplace=True)                
    activity_data= categorical_data.reindex(df.index, method='ffill').fillna(method='bfill')

    df['activity']=activity_data

    categorical_data_2s = accelerometer_df['main_activity_2s'].copy()
    categorical_data_2s.replace(cleanup_nums, inplace=True)
    activity_data_2s= categorical_data_2s.reindex(df.index, method='ffill').fillna(method='bfill')
    df['activity_2s']=activity_data_2s
    #categorical_data.head()
    
    return(df)

