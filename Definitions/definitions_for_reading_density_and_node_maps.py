#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from scipy import spatial
import math


# In[20]:


def import_pois(POI_df_directory):
    POI_df = pd.read_csv(POI_df_directory)
    POI_df = POI_df.drop(columns='Unnamed: 0.1')
    POI_df = POI_df.drop(columns='Unnamed: 0')
    return(POI_df)


# In[62]:


def import_nodes_and_links(NODES_dir):
    this_node_dir = NODES_dir + 'nodes.csv'
    this_links_dir = NODES_dir + 'links.csv'
    NODE_csv = pd.read_csv(this_node_dir)
    NODE_csv = NODE_csv.set_index('Unnamed: 0.1')
    if 'Unnamed: 0' in NODE_csv.columns:
        NODE_csv = NODE_csv.drop(columns='Unnamed: 0')
    LINKS_csv = pd.read_csv(this_links_dir)
    return(NODE_csv,LINKS_csv)


# In[8]:


# find closest point 

import math

def cartesian(latitude, longitude, elevation = 0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371 # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)
    return (X, Y, Z)


def deg2rad(degree):
    '''
    function to convert degree to radian
    '''
    rad = degree * 2*np.pi / 360
    return(rad)

def rad2deg(rad):
    '''
    function to convert radian to degree
    '''
    degree = rad/2/np.pi * 360
    return(degree)

 
#The KDTree is computing the euclidean distance between the two points (cities).
#The two cities and the center of the earth form an isosceles triangle.
def distToKM(x):
    '''
    function to convert cartesian distance to real distance in km
    '''
    R = 6371 # earth radius
    gamma = 2*np.arcsin(deg2rad(x/(2*R))) # compute the angle of the isosceles triangle
    dist = 2*R*math.sin(gamma/2) # compute the side of the triangle
    return(dist)

def kmToDIST(x):
    '''
    function to convert real distance in km to cartesian distance 
    '''
    R = 6371 # earth radius
    gamma = 2*np.arcsin(x/2./R) 
    
    dist = 2*R*rad2deg(math.sin(gamma / 2.))
    return(dist)


# In[28]:


#this_df: The df that has the points that we want to process for the KDTree
#this_latitude,this_longitude: the name of the columns that contain the corresponding points 
def construct_KDTree(this_df,this_latitude,this_longitude):
    this_df_places = []
    for index, row in this_df.iterrows():
        coordinates = [row[this_latitude], row[this_longitude]]
        cartesian_coord = cartesian(*coordinates)
        this_df_places.append(cartesian_coord)

    this_df_tree = spatial.KDTree(this_df_places)
    return(this_df_tree)


# In[32]:


# find traffic for nodes based on the links 

def find_traffic_for_nodes_based_on_links(NODE_csv,LINKS_csv):
    

    node_csv_index_array = np.array(NODE_csv.index)
    node_csv_traffic_values = np.zeros(len(NODE_csv.index))
    links_csv_v_values = np.array(LINKS_csv['v'])
    links_csv_u_values = np.array(LINKS_csv['u'])
    links_csv_traffic_values = np.array(LINKS_csv['traffic'])
    len(np.in1d(links_csv_v_values,node_csv_index_array))

    for i in range(0,len(node_csv_index_array)):
        item = node_csv_index_array[i]
        if (len(links_csv_traffic_values[np.where(links_csv_v_values == item)])==0) & (len(links_csv_traffic_values[np.where(links_csv_u_values == item)])==0):
            print('empty')
            node_csv_traffic_values[i]=0
        elif len(links_csv_traffic_values[np.where(links_csv_v_values == item)])!=0:
            node_csv_traffic_values[i]=np.max(links_csv_traffic_values[np.where(links_csv_v_values == item)])
        else: 
            node_csv_traffic_values[i]=np.max(links_csv_traffic_values[np.where(links_csv_u_values == item)])

    NODE_csv['traffic']=node_csv_traffic_values
    return(NODE_csv)


# In[36]:


def find_closest_node(this_df_tree, this_latitude,this_longitude):
    this_closest_node = this_df_tree.query([cartesian(this_latitude,this_longitude)], p = 2)
    return(this_closest_node)

def find_closest_nodes(this_df_tree, this_latitude,this_longitude, n):
    this_closest_nodes = this_df_tree.query([cartesian(this_latitude,this_longitude)], p = 2, k=n)
    return(this_closest_nodes)



