#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd 
import numpy as np
import scipy as sp
import scipy.fftpack
import geopy.distance as geo_di
from scipy.signal import savgol_filter


# In[3]:




# add distance column

# In[11]:


def basic_distance(dict_1, dict_2, i, j):
    if isinstance(dict_1.loc[i, 'Latitude'], float)==False and isinstance(dict_2.loc[j, 'Latitude'], float)==True:        
        #print("condition 1")
        coords_1 = (dict_1['Latitude'][i][0], dict_1['Longitude'][i][0])
        coords_2 = (dict_2['Latitude'][j], dict_2['Longitude'][j])
    elif isinstance(dict_1.loc[i, 'Latitude'], float)==True and isinstance(dict_2.loc[j, 'Latitude'], float)==False:        
        #print("condition 2")
        coords_1 = (dict_1['Latitude'][i], dict_1['Longitude'][i])
        coords_2 = (dict_2['Latitude'][j][0], dict_2['Longitude'][j][0])
    elif isinstance(dict_1.loc[i, 'Latitude'], float)==False and isinstance(dict_2.loc[j, 'Latitude'], float)==False:        
        #print("condition 3")
        coords_1 = (dict_1['Latitude'][i][0], dict_1['Longitude'][i][0])
        coords_2 = (dict_2['Latitude'][j][0], dict_2['Longitude'][j][0])
    else:
        coords_1 = (dict_1['Latitude'][i], dict_1['Longitude'][i])
        coords_2 = (dict_2['Latitude'][j], dict_2['Longitude'][j])
    dist = geo_di.distance(coords_1, coords_2).m
    return(dist)


# In[6]:


def correct_distance(item_index, this_df, distance_buffer):
    corrected_dist = this_df.loc[item_index, 'Distance']
    #print(corrected_dist)
    overwrite = False
    if corrected_dist>distance_buffer:
        print('old:', corrected_dist)
        this_index = this_df[this_df.index==item_index].index
        print(this_index, type(this_index))
        bottom_limit = this_index - pd.Timedelta(seconds=10)
        upper_limit = this_index + pd.Timedelta(seconds=10)
        
        following_distances = this_df[(this_df.index>this_index[0]) & (this_df.index<upper_limit[0])]['Distance'].mean()
        previous_distances = this_df[(this_df.index<this_index[0]) & (this_df.index>bottom_limit[0])]['Distance'].mean()
        #should we take into account the previous distances as well, or not?
        #corrected_dist = this_df_df.loc[this_index, 'Distance'] = (previous_distances + following_distances)/2
        corrected_dist = this_df.loc[this_index, 'Distance'] = following_distances
        print('new:', corrected_dist)
        overwrite = True
    return(overwrite, corrected_dist)


# In[7]:


#we are not using this one currently 
def correct_speed(item_index, this_df):
    corrected_speed = this_df.loc[item_index, 'Speed']
    #print(corrected_dist)
    #overwrite = False
    #if corrected_speed>speed_buffer
    print('old:', corrected_speed)
    this_index = this_df[this_df.index==item_index].index
    print(this_index, type(this_index))
    bottom_limit = this_index - pd.Timedelta(seconds=10)
    upper_limit = this_index + pd.Timedelta(seconds=10)
        
    following_speeds = this_df[(this_df.index>this_index[0]) & (this_df.index<upper_limit[0])]['Speed'].mean()
    previous_speeds = this_df[(this_df.index<this_index[0]) & (this_df.index>bottom_limit[0])]['Speed'].mean()
        #should we take into account the previous distances as well, or not?
        #corrected_dist = this_df_df.loc[this_index, 'Distance'] = (previous_distances + following_distances)/2
    corrected_speed = following_speeds
    this_df.loc[this_index, 'Speed'] = corrected_speed
    print('new:', corrected_speed)
        #overwrite = True
    return(corrected_speed)


# In[8]:


def add_distance_column(this_df):
    
    if 'Distance' in this_df.columns:
        this_df = this_df.drop('Distance', axis=1)
        this_df = this_df.assign(Distance=0)
    else:
        this_df = this_df.assign(Distance=0)


    #array to store distances (it is faster than accessing df rows)
    distance_array=np.zeros(len(this_df))
    for i in range(0, len(this_df.index)):
        dist=0
        if i>0:
            dist = basic_distance(this_df, this_df, this_df.index[i], this_df.index[i-1])
        distance_array[i]=dist
        #session_df.loc[session_df.index[i], 'Distance']=dist
    this_df['Distance']=distance_array
    return(this_df)



# calculate time passed

# In[14]:


def calculate_time_passed(this_df):
    if 'Time_Passed' in this_df.columns:
        this_df = this_df.drop('Time_Passed', axis=1)
        this_df = this_df.assign(Time_Passed=0)
    else:
        this_df = this_df.assign(Time_Passed=0)

    #create an array from the datetime index
    time_index_array = np.array([this_df.index.copy().tz_localize(None)])
    #find the time difference by:
    #-creating an array that is a copy of the first, offseted by 1 (so that the 1st element is the 2nd, etc.)
    #-findin the difference between the original and the offseted array
    #-converting to seconds
    timediff = (np.roll(time_index_array, -1)-time_index_array)/np.timedelta64(1, 's')
    this_df['Time_Passed']= np.delete(np.insert(timediff[0],0,0),-1)
    return(this_df)


# Calculate speed

# In[16]:


def calculate_speed(this_df):

    this_df = this_df.assign(Speed=0)

    distance_array = np.array(this_df['Distance'])
    time_passed_array = np.array(this_df['Time_Passed'])
    distance_array.shape
    speed_array = np.zeros(len(this_df.index))

    for i in range(0, len(this_df.index)):
        item = this_df.index[i]
        if isinstance(distance_array[i], float)==False:
            if time_passed_array[i][0]==0:
                #session_df.loc[item, 'Speed']=0
                speed_array[i]=0
            else:
                #if more than 2h have passed from the last measurement, this means that a new session is beginning 
                #and therefore we should not take into account this measurement
                if time_passed_array[i]/60 >120:
                    speed_array[i]=0
                    distance_array[i]=0
                else:
                    speed_array[i]=distance_array[i][0]/time_passed_array[i][0]
        else:
            if time_passed_array[i]==0:
                speed_array[i]=0
            else:
                if (time_passed_array[i]/60) >10:
                    speed_array[i]=0
                    distance_array[i]=0
                else: 
                    c_d = correct_distance(item, this_df, 100)
                    if c_d[0]==True:
                        distance_array[i]=c_d[1]                
                    speed_array[i]=distance_array[i]/time_passed_array[i]

    this_df['Distance'] = distance_array
    this_df['Time_Passed'] = time_passed_array
    this_df['Speed'] = speed_array
    this_df.loc[:, 'Speed'] = this_df.loc[:, 'Speed'].fillna(0)
    return(this_df)


# In[27]:


#add smoother speed
def add_smoother_speed(this_df):
    if 'Speed_smoother' not in this_df.columns:
        this_df = this_df.assign(Speed_smoother=0)
    speed_smoother = this_df['Speed']
    if len(this_df)>=60:
        speed_smoother = savgol_filter(this_df['Speed'].copy(), 35,2)
       
    this_df.loc[:,'Speed_smoother'] = speed_smoother  
    return(this_df)


# In[29]:


def transform_to_kmperhour(this_df):
    #Transform to km/h
    this_df.loc[:, 'Speed'] = this_df.loc[:, 'Speed']*(0.001/(1/3600))
    this_df.loc[:, 'Speed_smoother'] = this_df.loc[:, 'Speed_smoother']*(0.001/(1/3600))
    return(this_df)


# In[31]:


def classify_mode_of_transport_basic(this_df):
    #Classify activity (if its bike/train)
    bike_speed_limit = 10
    train_speed_limit = 40

    walk_speeds_median=[]
    walk_speeds_mean=[]
    this_df = this_df.assign(Activity_type='Walk')


    for item in this_df[this_df['Speed']>bike_speed_limit]['Session'].unique():
        sub_df = this_df[this_df['Session']==item]
        #session_df[(session_df['Session']==item) & (session_df['Speed_smoother']>bike_speed_limit)]
        bike_df = this_df[(this_df['Session']==item) & (this_df['Speed']>bike_speed_limit)]    
        if (len(sub_df)>120) & (len(bike_df)>60):
            #print(item, 'possible train')
            total_secs = bike_df['Time_Passed'].sum()/60
            all_secs = (sub_df.index[-1]-sub_df.index[0]).total_seconds()/60
            total_secs_ratio = total_secs/all_secs
            if total_secs_ratio>0.1:
                print(item, 'possible bike', sub_df['Speed'].median())
                this_df.loc[sub_df.index, 'Activity_type'] = 'Bike'
    return(this_df)


# In[33]:


def check_moving(this_df):
    #Check if the subject is moving or no
    if 'Moving' in this_df.columns:
        this_df = this_df.drop('Moving', axis=1)
    else:
        this_df = this_df.assign(Moving=0)

    this_df.loc[this_df[this_df['Time_Passed']<4].index, 'Moving'] = 1
    return(this_df)


# In[35]:


def fast_fourier_speed_smooth(this_df):
    
    #Smooth with fast fourier transform
    # compute the fast fourier transform of the signal 
    speed_fft = sp.fftpack.fft(this_df['Speed'].values)
    # compute the power spectral density 
    speed_psd = np.abs(speed_fft) ** 2
    # get the frequencies corresponding to the values of the PSD
    fftfreq = sp.fftpack.fftfreq(len(speed_psd), 1. / 5)
    speed_fft_bis = speed_fft.copy()
    speed_fft_bis[np.abs(fftfreq) > 0.5] = 0
    speed_slow = np.real(sp.fftpack.ifft(speed_fft_bis))
    this_df['Speed']=speed_slow
    return(this_df)


# In[40]:


def create_firstgradediff_forspeed(this_df):
    if 'Speed_dif' not in this_df.columns:
        this_df = this_df.assign(Speed_dif=0)
    #this_df = this_df.assign(Speed_dif=0)
    this_df.loc[this_df.index[0], 'Speed_dif']=0
    this_df.loc[this_df.index[1]:, 'Speed_dif'] = this_df['Speed'][1:].diff().fillna(0)/this_df['Time_Passed'][1:]
    return(this_df)


# In[42]:


def round_speed_features(this_df):
    this_df['Speed'] = round(this_df['Speed'], 1)
    if 'Speed_smoother' in this_df.columns:
        this_df['Speed_smoother'] = round(this_df['Speed_smoother'], 1)
    this_df['Speed_dif'] = round(this_df['Speed_dif'], 2)
    return(this_df)


# In[45]:


def find_speed_mean_and_std_thirtyseconds(this_df):
    resampled_speed = this_df['Speed'].resample('30s').std()
    resampled_speed_mean = this_df['Speed'].resample('30s').mean()

    if 'Speed_std' in this_df.columns:
        this_df = this_df.drop('Speed_std', axis=1)
    if 'Speed_mean_30sec' in this_df.columns:
        this_df = this_df.drop('Speed_mean_30sec', axis=1)
    this_df = this_df.assign(Speed_std=0)
    this_df = this_df.assign(Speed_mean_30sec=0)
    
    
    for i in range(0, len(resampled_speed.index)-1):
        this_index = resampled_speed.index[i]
        next_index = resampled_speed.index[i+1]
        this_df.loc[this_index:next_index,'Speed_std'] = this_df.loc[this_index:next_index,'Speed'].std()
        this_df.loc[this_index:next_index,'Speed_mean_30sec'] = this_df.loc[this_index:next_index,'Speed'].mean() 
    
    return(this_df)


# In[46]:


#This is the one that collects everything:
def basic_speed_functions(df):

    this_session_df = df.copy()
    #Calculate speed
    this_session_df = add_distance_column(this_session_df)
    this_session_df = calculate_time_passed(this_session_df)
    this_session_df['Distance'] = round(this_session_df['Distance'], 2) 
    this_session_df = calculate_speed(this_session_df)
    this_session_df = add_smoother_speed(this_session_df)
    this_session_df = transform_to_kmperhour(this_session_df)
    this_session_df = classify_mode_of_transport_basic(this_session_df)
    this_session_df = check_moving(this_session_df)
    this_session_df = fast_fourier_speed_smooth(this_session_df)

    #Compute features
    this_session_df = create_firstgradediff_forspeed(this_session_df)
    this_session_df = round_speed_features(this_session_df)
    this_session_df = find_speed_mean_and_std_thirtyseconds(this_session_df)
    return(this_session_df)


# In[ ]:




