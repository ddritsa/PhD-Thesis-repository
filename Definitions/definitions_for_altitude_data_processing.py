
# coding: utf-8

# In[12]:


import pandas as pd 
import numpy as np
import datetime
from datetime import datetime 
from datetime import timedelta
import geopy.distance as geo_di


# In[6]:


from definitions_for_scraping_fitbit_data import scrape_movement_data
from definitions_for_speed_data_processing import basic_speed_functions
import matplotlib.pyplot as plt


# In[8]:


def create_firstgradediff_forAlt(this_df):
    this_df = this_df.assign(Altitude_dif=0)
    this_df.loc[this_df.index[0], 'Altitude_dif']=0
    this_df.loc[this_df.index[1]:, 'Altitude_dif'] = this_df['Altitude'][1:].diff().fillna(0)/this_df['Time_Passed'][1:]
    return(this_df)


# In[14]:


def find_altitude_features(this_df):

    this_df = this_df.assign(Integer_index=0)
    this_df['Integer_index']=np.arange(len(this_df))

    this_df['Altitude change_duration']=0
    this_df['Altitude change_slope']=0
    alt_dif_array=np.zeros(len(this_df))
    alt_dif_buffer = 0.01

    steady_altdif_array = np.array(this_df[this_df['Altitude_dif']>=alt_dif_buffer].index)
    integer_index_array = np.array(this_df[this_df['Altitude_dif']>=alt_dif_buffer]['Integer_index'])
    steady_altdif_indices = pd.to_timedelta(np.roll(steady_altdif_array, -1) - steady_altdif_array)/np.timedelta64(1, 's')
    break_indices = np.where(steady_altdif_indices>5)
    partitioned_index = np.split(integer_index_array, break_indices[0]+1)
    for item in partitioned_index:
        if len(item)>1:
            sub_df_start = this_df[this_df['Integer_index']==item[0]]
            sub_df_end = this_df[this_df['Integer_index']==item[-1]]
            this_df.loc[sub_df_start.index[0]:sub_df_end.index[0], 'Altitude change_duration']=(sub_df_end.index[0]-sub_df_start.index[0]).total_seconds()
            #print('secs:',(sub_df_end.index[0]-sub_df_start.index[0]).total_seconds())
            start_coords = (this_df.loc[sub_df_start.index[0], 'Latitude'],this_df.loc[sub_df_start.index[0], 'Longitude'])
            end_coords = (this_df.loc[sub_df_end.index[0], 'Latitude'],this_df.loc[sub_df_end.index[0], 'Longitude'])
            this_dist = geo_di.distance(start_coords,end_coords).m
            #print('alt change:', (test_df.loc[sub_df_end.index[0], 'Altitude']-test_df.loc[sub_df_start.index[0], 'Altitude'])/this_dist)
            this_df.loc[sub_df_start.index[0]:sub_df_end.index[0], 'Altitude change_slope'] = (this_df.loc[sub_df_end.index[0], 'Altitude']-this_df.loc[sub_df_start.index[0], 'Altitude'])/this_dist
    return(this_df)


# In[17]:


# How to use:
'''
session_df = df.copy()
#First we have to have the columns 'Time_Passed', 'Longitude' and 'Latitude', and 'Altitude'
#So we have to create the df (use the following definition set:)
from definitions_for_scraping_fitbit_data import scrape_movement_data
#We also have to create the 'Time_Passed' column and this definition is included in the speed processing defs
#We can either do all the speed functions or this one. The following line does all of them 
#Otherwise we should import only the definition for Time_Passed 
from definitions_for_speed_data_processing import basic_speed_functions
session_df = basic_speed_functions(session_df)
#Then we can use the following defs:
session_df = create_firstgradediff_forAlt(session_df)
session_df['Altitude_dif'] = round(session_df['Altitude_dif'], 2)
session_df = find_altitude_features(session_df)
session_df
'''

