#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import datetime
from datetime import datetime 
from datetime import timedelta
import time

import scipy as sp
import scipy.fftpack

import random


# In[1]:


#signal_name = 'HR_from e4'

def HR_peak_detection(df,frq, signal_name):
    def HR_frequency_cutoff(this_df):  
        subset_for_fft = this_df.copy()
        #plt.figure(figsize=(20,3))
        #plt.plot(subset_for_fft[signal_name])

        # compute the fast fourier transform of the signal 
        hr_fft = sp.fftpack.fft(subset_for_fft[signal_name].values)
        hr_fft
        # compute the power spectral density 
        hr_psd = np.abs(hr_fft) ** 2
        hr_psd
        # get the frequencies corresponding to the values of the PSD
        fftfreq = sp.fftpack.fftfreq(len(hr_psd), 1. / 5)
        #print(fftfreq)
        '''
        plt.figure(figsize=(20,3))
        plt.plot(hr_fft[(hr_fft<2000)&(hr_fft>-2000)])
        plt.figure(figsize=(20,3))
        plt.plot(hr_psd[(hr_psd<5000000)&(hr_psd>-2000)])
        plt.figure(figsize=(20,3))
        plt.plot(fftfreq)
        plt.figure(figsize=(20,3))
        plt.plot(fftfreq[fftfreq>0], 10 * np.log10(hr_psd[fftfreq>0]))
        '''
        hr_fft_bis = hr_fft.copy()
        hr_fft_bis[np.abs(fftfreq) > 0.2] = 0
        hr_afterfreqprocess = np.real(sp.fftpack.ifft(hr_fft_bis))
        #plt.figure(figsize=(20,3))
        #plt.plot(subset_for_fft[signal_name][:2000])
        #plt.figure(figsize=(20,3))
        #plt.plot(subset_for_fft[signal_name][:2000].index, hr_afterfreqprocess[:2000])
        return(hr_afterfreqprocess)



    # frequencies: 5, 20
    # PUT THE FOLLOWING
    # FREQ 5 
    #this_df = test_df
    def HRPeak_createlists(this_df, this_freq):
        #create a (resampled) copy of the original df to work on
        freq = this_freq
        psd_copy = this_df.copy()
        psd_copy[signal_name]=HR_frequency_cutoff(this_df)
        copy_df = psd_copy.copy().resample(str(freq)+'s').mean()
        copy_df = copy_df.fillna(method='backfill')


        integer_arr = np.arange(len(copy_df))

        if 'Integer_index' not in copy_df.columns:
            copy_df = copy_df.assign(Integer_index=integer_arr)
        else:
            copy_df['Integer_index']=integer_arr

        if 'peak_rise' not in copy_df.columns:
            copy_df = copy_df.assign(peak_rise=0)
        else:
            copy_df['peak_rise']=0

        if 'peak_fall' not in copy_df.columns:
            copy_df = copy_df.assign(peak_fall=0)
        else:
            copy_df['peak_fall']=0

        #SET DIFFERENCE CUTOFFS
        lower_cutoff = 0.0001
        upper_cutoff = 50
        copy_df = copy_df.assign(peak_rise=0)
        copy_df.loc[copy_df[(copy_df[signal_name].diff()>lower_cutoff) & (copy_df[signal_name].diff()<upper_cutoff)].index,'peak_rise'] = 1
        copy_df = copy_df.assign(peak_fall=0)
        copy_df.loc[copy_df[(copy_df[signal_name].diff()< (-1*lower_cutoff)) & (copy_df[signal_name].diff()> (-1*upper_cutoff))].index,'peak_fall'] = 1

        signal_amplitudes_up = []
        signal_amplitudes_down = []




        continuous_peaks_array = np.array(copy_df[copy_df['peak_rise']==1].index)
        continuous_falls_array = np.array(copy_df[copy_df['peak_fall']==1].index)
        integer_index_array_peaks = np.array(copy_df[copy_df['peak_rise']==1]['Integer_index'])
        integer_index_array_falls = np.array(copy_df[copy_df['peak_fall']==1]['Integer_index'])
        corrected_peak_array = np.array(copy_df['peak_rise'])
        corrected_fall_array = np.array(copy_df['peak_fall'])
        this_signal_array=np.array(copy_df[signal_name])
        continuous_peaks_indices = pd.to_timedelta(np.roll(continuous_peaks_array, -1) -continuous_peaks_array)/np.timedelta64(1, 's')
        continuous_falls_indices = pd.to_timedelta(np.roll(continuous_falls_array, -1) - continuous_falls_array)/np.timedelta64(1, 's')
        break_indices_peaks = np.where(continuous_peaks_indices>2*freq)
        break_indices_falls = np.where(continuous_falls_indices>2*freq)
        partitioned_index_riseups = np.split(integer_index_array_peaks, break_indices_peaks[0]+1)
        partitioned_index_falldowns = np.split(integer_index_array_falls, break_indices_falls[0]+1)


        rise_ups_list=[]
        for item in partitioned_index_riseups:
            if (len(item)<=2)& (len(item)>0):
                #print(len(corrected_peak_array), item)
                corrected_peak_array[item[0]]=0
            elif len(item)>0:
                this_signal_amplitude = this_signal_array[item[-1]]-this_signal_array[item[0]]
                #print(item, EDR_amplitude)
                #if EDR_amplitude <0.02:
                    #corrected_peak_array[item[0]:item[-1]]=0
                #else:
                if ((item[-1]+15*freq)<len(corrected_fall_array))& (1 not in corrected_fall_array[item[-1]:item[-1]+15*freq]):
                    #print(item, corrected_fall_array[item[-1]:item[-1]+7])
                    corrected_peak_array[item[0]:item[-1]]=0
                else:
                        #print('valid')
                        #only then it is a valid rise up list
                    rise_ups_list.append(item.tolist())

        fall_downs_list=[]
        #print('fall')
        for item in partitioned_index_falldowns:
            if (len(item)<=2)& (len(item)>0):

                corrected_fall_array[item[0]]=0
            elif len(item)>0:

                this_signal_amplitude = this_signal_array[item[0]]-this_signal_array[item[-1]]
                #if EDR_amplitude <0.02:
                    #corrected_fall_array[item[0]:item[-1]]=0
                #else:
                if ((item[0]-15*freq)>0) & (1 not in corrected_peak_array[item[0]-15*freq:item[0]]):
                    corrected_fall_array[item[0]:item[-1]]=0
                else:
                        #only then it is a valid rise up list
                    #if len(item)>9:
                        #print('very big')
                        #fall_downs_list.append(item[:9].tolist())
                    #else:
                    fall_downs_list.append(item.tolist())

        #copy_df is the resampled df 
        #rise_ups_list is the list with the consecutive points with positive 1st grade diff
        #rise_ups_list is the list with points with consecutive negative 1st grade diff
        return(copy_df, rise_ups_list, fall_downs_list,corrected_peak_array,corrected_fall_array,this_signal_array)


    def HRPeak_connect_ups_and_downs(rise_ups_list,fall_downs_list,corrected_peak_array,corrected_fall_array):
        mega_list = np.array(sorted(fall_downs_list+rise_ups_list))
        #rise_ups_corrected=[]
        #fall_downs_corrected=[]

        peaksandfalls = np.zeros((len(mega_list),2))
        #fall_downs_corrected_array = np.zeros((len(mega_list),2))


        for i in range(0, len(mega_list[:-2])):
            #maybe we need to add the following again to control the width:
            #if ((mega_list[i+1][0]-mega_list[i][-1])<20*freq):
            if (corrected_peak_array[mega_list[i][0]]==1) & (corrected_fall_array[mega_list[i+1][0]]==1) :
                #print('efyges:', (mega_list[i+1][0]-mega_list[i][-1])/0.25)
            #else:
                    #print([mega_list[i][0],mega_list[i+1][-1]])
                peaksandfalls[i]= [mega_list[i][0],mega_list[i+1][-1]]
                #fall_downs_corrected_array[i]= [mega_list[i+1][0],mega_list[i+1][-1]]
                #rise_ups_corrected.append(mega_list[i])
                #fall_downs_corrected.append(mega_list[i+1])


        corrected_peaksandfalls = peaksandfalls[np.nonzero(peaksandfalls[:,1])]
        return(corrected_peaksandfalls)

    def HRPeak_find_duration_and_amplitude(copy_df, corrected_peaksandfalls,this_signal_array):

        #create columns in the EDA df 
        copy_df.loc[:,'HR_peak_fall']=0
        copy_df.loc[:,'HR_peak_rise']=0
        #test_df.loc[:,'Stress']=0
        #test_df['TEMP_smoothed']=savgol_filter(test_df['TEMP'].copy(),51,3)
        copy_df = copy_df.assign(HR_peak_session=0)
        this_df_index = np.array(copy_df.index)
        #test_df.loc[:,'EDA_session']=0
        #test_df['EDA_tonic']=test_df['EDA'].copy()

        #create helper arrays
        #this_signal_tonic_array=np.array(test_df['EDA_tonic'])
        corrected_peak_rise_array=np.zeros(len(copy_df))
        corrected_peak_fall_array=np.zeros(len(copy_df))
        this_signal_amplitude_array=np.zeros(len(copy_df))
        this_signal_duration_array=np.zeros(len(copy_df))
        this_signal_session_array=np.zeros(len(copy_df))
        #this_signal_array=np.array(test_df['TEMP_smoothed'])
        #stress_array=np.zeros(len(test_df))

        for item in corrected_peaksandfalls:
            #print('---')
            up = int(item[0])
            down = int(item[1])
            #print(up,down)
            max_index = np.argmax(this_signal_array[up:down])+up
            #print('up:',up,'max',max_index,'down',down)
            #DA_df_index
            this_signal_amplitude = np.max(this_signal_array[up:down]) - this_signal_array[up]
            #print('amplitude:',EDA_amplitude)
            this_signal_rise_duration = (this_df_index[down]-this_df_index[up])/np.timedelta64(1, 's')
            #print(this_signal_rise_duration)
            #print('rise duration', EDA_rise_duration)
            #assign the values of tonic EDA
            #EDA_tonic_array[up]=this_signal_array[up]
            #EDA_tonic_array[down]=this_signal_array[down]
            #EDA_tonic_array[up:down]=np.linspace(EDR_array[up],EDR_array[down], down-up) 


            #print(activity_data_array[up:down])
            #if (this_signal_amplitude>=0.05) & (EDA_rise_duration>=1) & (3 not in activity_data_array[up:max_index]):
                #print('valid')
            corrected_peak_rise_array[up:max_index]=1
            corrected_peak_fall_array[up:max_index]=0
            corrected_peak_rise_array[max_index:down+1]=0
            corrected_peak_fall_array[max_index:down+1]=1

            this_signal_amplitude_array[up:down+1]=this_signal_amplitude
            this_signal_duration_array[up:down+1]=this_signal_rise_duration
            this_signal_session = random.randint(1,100000)
            if this_signal_session in np.unique(this_signal_session_array):
                #print('is in')
                while this_signal_session in np.unique(this_signal_session_array):
                    this_signal_session = random.randint(1,len(corrected_peaksandfalls) + 1000000000)
            this_signal_session_array[up:down+1]=int(this_signal_session)

                #find where the temperature goes up and where down
                #this_temp = TEMP_array[up:down]
                #print('TEMP:', this_temp, np.diff(this_temp), np.where(np.diff(this_temp)<0),'NUM:',len(np.where(np.diff(this_temp)<0)[0]))
                #if len(np.where(np.diff(this_temp)<0)[0])>0:
                    #print('stress')
                    #stress_array[up:down+1]=1
                #if len(eda_df_subset[eda_df_subset['TEMP_smoothed'].diff()<0])>0:
                    #test_df.loc[up[0]:down[-1]+timedelta(seconds= 1* 0.25), 'Stress'] = 1
                #'-----'
        #test_df['EDA_tonic']=EDA_tonic_array
        copy_df['HR_amplitude']=this_signal_amplitude_array
        copy_df['HR_peak_duration']=this_signal_duration_array
        copy_df['HR_peak_rise']=corrected_peak_rise_array
        copy_df['HR_peak_fall']=corrected_peak_fall_array
        copy_df['HR_peak_session']=this_signal_session_array
        #test_df['Stress']=stress_array

        return(copy_df)

    def HRPeak_store_peaks_in_df(copy_df,target_df):

        if 'HR_peak_session' not in target_df.columns:
            target_df = target_df.assign(HR_peak_session=0)
        else:
            target_df['HR_peak_session']=0
        if 'HR_amplitude' not in target_df.columns:
            target_df = target_df.assign(HR_amplitude=0)
        else:
            target_df['HR_amplitude']=0
        if 'HR_peak_duration' not in target_df.columns:
            target_df = target_df.assign(HR_peak_duration=0)
        else:
            target_df['HR_peak_duration']=0


        for item in copy_df['HR_peak_session'].unique():
            this_df = copy_df[copy_df['HR_peak_session']==item]
            #print(this_df.index[0],this_df.index[-1])
            a = target_df[(target_df.index>this_df.index[0])&(target_df.index<this_df.index[-1])]
            #print(len(a))
            target_df.loc[(target_df.index>this_df.index[0])&(target_df.index<this_df.index[-1]),'HR_amplitude']=this_df.loc[this_df.index[0],'HR_amplitude']
            target_df.loc[(target_df.index>this_df.index[0])&(target_df.index<this_df.index[-1]),'HR_peak_duration']=this_df.loc[this_df.index[0],'HR_peak_duration']
            #test_df['HR_peak_rise']=corrected_peak_rise_array
            #test_df['HR_peak_fall']=corrected_peak_fall_array
            target_df.loc[(target_df.index>this_df.index[0])&(target_df.index<this_df.index[-1]),'HR_peak_session']=this_df.loc[this_df.index[0],'HR_peak_session']
        return(target_df)

    #the definition that collects the other defs about HR peaks and applies them
    #returning a copy of the original df
    def find_HR_peaks(this_df, this_frequency):
        if 'HR peak duration and amplitude' not in this_df.columns:
            this_df = this_df.assign(HR_peak_duration_and_amplitude=0)
        else:
            this_df['HR_peak_duration_and_amplitude']=0       
        find_HR_consec_points = HRPeak_createlists(this_df,this_frequency)
        corrected_list = HRPeak_connect_ups_and_downs(find_HR_consec_points[1],find_HR_consec_points[2],find_HR_consec_points[3],find_HR_consec_points[4])
        df_with_peaks_resampled = HRPeak_find_duration_and_amplitude(find_HR_consec_points[0], corrected_list,find_HR_consec_points[5])

        df_with_peaks = HRPeak_store_peaks_in_df(df_with_peaks_resampled, this_df)

        this_df = df_with_peaks
        this_df['HR_peak_duration_and_amplitude'] = this_df['HR_amplitude']*this_df['HR_peak_duration']
        #this_df[this_df['faulty HR']]
        return(this_df)
    
    df_dummy = find_HR_peaks(df,frq)
    return(df_dummy)
    #"df_with_peaks_resampled" is resampled at a lower frequency and contains the 'peak rise' and 'peak fall' columns


# In[ ]:




