#!/usr/bin/env python
# coding: utf-8

# ### To use these definitions: 
# (in another file)
# Uncomment the following:

# In[ ]:


#from definitions_for_activity_recognition import label_accelerometer_data, detect_change_of_intensity


# In[7]:


import pandas as pd
import numpy as np
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from datetime import timedelta

# In[8]:


import h5py
from keras.models import load_model

#this_path = r'C:\Users\Karl Mirx\important files until Sept 2018\Definitions'
this_path = r'C:\Users\demdr\UTS\Important files for PhD thesis\Definitions'

model_m = load_model(this_path + r'\activity_recognition_model_40s_7layers.h5')
model_m_8s = load_model(this_path + r'\activity_recognition_model_8s_7layers.h5')

def label_accelerometer_data(this_df, time_periods):

    LABEL = 'ActivityEncoded'
    #TIME_PERIODS = 40 corresponds to 10 seconds, 8 corresponds to 2 seconds
    # time periods should be either 40 or 8, to work with the 2 models that we have loaded. Otherwise we have to 
    # load another model that works for other seconds
    TIME_PERIODS = time_periods
    STEP_DISTANCE = time_periods
    
    #definition for feature extraction
    def create_segments_and_labels(df, time_steps, step):

        # x, y, z acceleration as features
        N_FEATURES = 9
        segments = []
        labels = []
        for i in range(0, len(df) - time_steps, step):
            xs = df['ACC_1'].values[i: i + time_steps]
            ys = df['ACC_2'].values[i: i + time_steps]
            zs = df['ACC_3'].values[i: i+ time_steps]
            xs_std = np.std(df['ACC_1'].values[i: i + time_steps])
            ys_std = np.std(df['ACC_2'].values[i: i + time_steps])
            zs_std = np.std(df['ACC_3'].values[i: i + time_steps])
            xs_diff = np.diff(df['ACC_1'].values[i: i + time_steps])
            ys_diff = np.diff(df['ACC_2'].values[i: i + time_steps])
            zs_diff = np.diff(df['ACC_3'].values[i: i + time_steps])
            xs_diff = np.insert(xs_diff, 0, xs_diff[0])
            ys_diff = np.insert(ys_diff, 0, ys_diff[0])
            zs_diff = np.insert(zs_diff, 0, zs_diff[0])
            xs_std_arr = np.full(xs.shape, xs_std)
            ys_std_arr = np.full(ys.shape, ys_std)
            zs_std_arr = np.full(zs.shape, zs_std)
            segments.append([xs, ys, zs, xs_std_arr, ys_std_arr, zs_std_arr, xs_diff, ys_diff, zs_diff])
     

        # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
 

        return reshaped_segments
   
    
    #definition for putting the resulting activity label to the df
    def put_detected_labels(df, time_steps, step):
        cnt = 0
        for i in range(0, len(df) - time_steps, step):
            #do this if we want to check the algorithm with data that are labelled
            #if label_name in df.columns:
                #print('already labelled')
                #label = stats.mode(df[label_name][i: i + time_steps])[0][0]
                #df.loc[df.index[i]:df.index[i + time_steps], 'this_label']=label
            df.loc[df.index[i]:df.index[i + time_steps], 'Detected Activity']=max_y_pred_test[cnt]
            cnt+=1
        return(df)

    # Normalize features for training data set
    df_for_labelling = this_df.copy()
    df_for_labelling['Detected Activity']=0
    
    df_for_labelling['ACC_1'] = df_for_labelling['ACC_1'] / df_for_labelling['ACC_1'].max()
    df_for_labelling['ACC_2'] = df_for_labelling['ACC_2'] / df_for_labelling['ACC_2'].max()
    df_for_labelling['ACC_3'] = df_for_labelling['ACC_3'] / df_for_labelling['ACC_3'].max()

    df_for_labelling = df_for_labelling.round({'ACC_1': 4, 'ACC_2': 4, 'ACC_3': 4})

    x_test = create_segments_and_labels(df_for_labelling,
                                                TIME_PERIODS,
                                                STEP_DISTANCE)
    input_shape = (x_test.shape[1]*x_test.shape[2])
    num_classes = 3
    # Set input_shape / reshape for Keras
    x_test = x_test.reshape(x_test.shape[0], input_shape)

    x_test = x_test.astype('float32')


    model_for_prediction = 0
    if time_periods == 40:
        model_for_prediction = model_m
    elif time_periods == 8:
        model_for_prediction = model_m_8s
    else:
        print('select time period = 40 or 8')
    y_pred_test = model_for_prediction.predict(x_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)



    #df_predicted is the resulting df which has the labels
    df_predicted = put_detected_labels(df_for_labelling,TIME_PERIODS,STEP_DISTANCE)
    df_predicted = df_predicted .fillna(0)
    return(df_predicted)


# In[136]:


def detect_change_of_intensity(df):
    this_df = df.copy()
    this_df['ACC_together'] = np.sqrt((this_df['ACC_1']**2) + (this_df['ACC_2']**2)+(this_df['ACC_3']**2))
    this_df['Change of movement intensity_After prediction']=0
    this_df['Change of activity state_After prediction']=0
    this_df['Spontaneous movement_After prediction']=0
    this_df['Steady state']=0
    
    this_index = np.where(this_df['Detected Activity'].diff()!=0)
    print(this_index[0])
    
    for ind in this_index[0][1:]:
        phase_duration = (this_index[0][np.where(this_index[0]==ind)[0]]-this_index[0][np.where(this_index[0]==ind)[0]-1])/4
        current = this_df.index[this_index[0][np.where(this_index[0]==ind)[0]]][0]
        former = this_df.index[this_index[0][np.where(this_index[0]==ind)[0]-1]][0]
        #The standard buffer is 20 seconds
        buffer = 72
        # if the previous state lasted more than 2 minutes:
        if phase_duration>120:
            #buffer = 1 minute after the change 
            buffer = 232 
        elif phase_duration<=12:
            #buffer = 2.5 seconds 
            buffer = 52
        else:
            #buffer = 15 seconds
            buffer = 72
        prev = 0
        nex = 0
        if ind-20<0:
            prev = 0
        else:
            prev = ind-20
        if ind+buffer>len(this_df.index):
    
           
            nex = len(this_df.index)

        else:
            nex = ind+buffer
        
        if buffer==52:
            this_df.loc[this_df.index[prev:nex], 'Spontaneous movement_After prediction']=1
        elif buffer==232:
            this_df.loc[this_df.index[prev:nex], 'Change of activity state_After prediction']=1
            this_df.loc[former:current, 'Steady state']=1
        elif (len(this_df.index[prev:nex])>0):
            this_df.loc[this_df.index[prev:nex], 'Change of movement intensity_After prediction']=1
            
    this_df = this_df.fillna(0)
    return(this_df)






