#!/usr/bin/env python
# coding: utf-8

# ### To use these definitions:
# (in another file) Uncomment the following:

# In[ ]:


#from definitions_for_activity_recognition import label_artifact_data


# In[ ]:


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


# In[ ]:


import h5py
from keras.models import load_model


#Path in the old laptop:
#this_path = r'C:\Users\Karl Mirx\important files until Sept 2018\Definitions'
#Path for new laptop:
this_path = r'C:\Users\demdr\UTS\Important files for PhD thesis\Definitions'


model_FOR_ARTIFACT = load_model(this_path + r'\artifact_recognition_model_12secs-7params.h5')

def label_artifact_data(this_df):

    #LABEL = 'Specific artifact'
    #good: TIME_PERIODS = 40 (10 seconds)
    TIME_PERIODS = 12
    STEP_DISTANCE= 12
    
    def create_segments_and_labels_for_artifact_detection(df, time_steps, step):

        # x, y, z acceleration as features
        N_FEATURES = 7
        # Number of steps to advance in each iteration (for me, it should always
        # be equal to the time_steps in order to have no overlap between segments)
        # step = time_steps
        segments = []
        labels = []
        for i in range(0, len(df) - time_steps, step):
            mean_ed = df['EDA'].values[i: i + time_steps]
            prev_ed = 0
            if i>0:
                prev_ed = df['EDA'].values[i-1: i + time_steps-1]
            else:
                prev_ed = np.insert(df['EDA'].values[i : i + time_steps-1],-1,0)
            xs = df['EDA_diff'].values[i: i + time_steps]
            xs2 = df['EDA_diff_2'].values[i: i + time_steps]
            xs_mean = np.mean(df['EDA_diff'].values[i: i + time_steps])
            xs_mean_arr = np.full(xs.shape, xs_mean)
            xs_min = np.full(xs.shape, np.min(df['EDA_diff'].values[i: i + time_steps]))
            xs_max = np.full(xs.shape, np.max(df['EDA_diff'].values[i: i + time_steps]))
            xs2_min = np.full(xs.shape, np.min(df['EDA_diff_2'].values[i: i + time_steps]))
            xs2_max = np.full(xs.shape, np.max(df['EDA_diff_2'].values[i: i + time_steps]))
            xs2_mean = np.mean(df['EDA_diff_2'].values[i: i + time_steps])
            xs2_mean_arr = np.full(xs.shape, xs2_mean)
            xs_std = np.std(df['EDA'].values[i: i + time_steps])
            xs_std_arr = np.full(xs.shape, xs_std)
            xs_diff_std= np.std(df['EDA_diff'].values[i: i + time_steps])
            xs_diff_std_arr = np.full(xs.shape, xs_diff_std)
            
            #label = stats.mode(df[label_name][i: i + time_steps])[0][0]
            segments.append([mean_ed,  xs, xs2, xs_mean_arr, xs2_mean_arr, xs_std_arr,xs_diff_std_arr])
            #labels.append(label)

        # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
        #labels = np.asarray(labels)

        return reshaped_segments
    
    #definition for putting the resulting activity label to the df
    def put_detected_labels(df, time_steps, step, label_name):
        cnt = 0
        for i in range(0, len(df) - time_steps, step):
            #do this if we want to check the algorithm with data that are labelled
            #if label_name in df.columns:
                #label = stats.mode(df[label_name][i: i + time_steps])[0][0]
                #df.loc[df.index[i]:df.index[i + time_steps], 'this_label']=label
            df.loc[df.index[i]:df.index[i + time_steps], 'EDA artifact']=max_y_pred_test[cnt]
            cnt+=1
        return(df)

    # Normalize features for training data set
    df_for_labelling = this_df.copy()
    df_for_labelling['EDA artifact']=0
    
    df_for_labelling['EDA_diff']=df_for_labelling['EDA'].diff()
    df_for_labelling['EDA_diff_2']=df_for_labelling['EDA_diff'].diff()
    
    df_for_labelling['EDA_diff']= abs(df_for_labelling['EDA_diff'])
    df_for_labelling['EDA_diff_2']= abs(df_for_labelling['EDA_diff_2'])
    df_for_labelling = df_for_labelling.fillna(0)

    df_for_labelling = df_for_labelling.round({'EDA_diff': 4, 'EDA_diff_2': 4})

    x_test = create_segments_and_labels_for_artifact_detection(df_for_labelling,
                                                TIME_PERIODS,
                                                STEP_DISTANCE)
    input_shape = (x_test.shape[1]*x_test.shape[2])
    #num_classes = le.classes_.size
    num_classes = 2
    # Set input_shape / reshape for Keras
    x_test = x_test.reshape(x_test.shape[0], input_shape)

    x_test = x_test.astype('float32')
    #y_test = y_test.astype('float32')


    y_pred_test = model_FOR_ARTIFACT.predict(x_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)



    #df_predicted is the resulting df which has the labels
    LABEL = 'Specific artifact'
    df_predicted = put_detected_labels(df_for_labelling,TIME_PERIODS,STEP_DISTANCE,LABEL)
    df_predicted = df_predicted .fillna(0)
    #The definition returns a copy of the original artifact, with a new column, 'EDA artifact'
    #If 'EDA artifact =0, this is an artifact'
    return(df_predicted)

