#!/usr/bin/env python
# coding: utf-8

# To use this:
# 
# Put in the parenthesis: User email, password, user ID)
# 
# (Email+password that are used in Strava
# 

# In[3]:


import pandas as pd 
from xml.dom import minidom
import csv

from bs4 import BeautifulSoup
import requests
#import regular expressions
import re
import numpy as np
import time
import datetime
from datetime import datetime 
from datetime import timedelta


# In[5]:


def import_gpx(file_dir):
    #read each datastream from the gpx file
    #file_dir = file_dir.decode('UTF-8')
    mydoc2 = minidom.parseString(file_dir)
    trkpt = mydoc2.getElementsByTagName('trkpt')
    time = mydoc2.getElementsByTagName('time')
    ele = mydoc2.getElementsByTagName('ele')
    hr = mydoc2.getElementsByTagName('gpxtpx:hr')
    cad = mydoc2.getElementsByTagName('ns3:cad')
    #create empty lists to store the data streams
    times = []
    lats = []
    longs = []
    eles = []
    hrs = []
    distances = []
    #extract data from each element of the created lists and append it to the corresponding list
    for elem in trkpt:
        lats.append(float(elem.attributes['lat'].value))
        longs.append(float(elem.attributes['lon'].value))
    for elem in time:
        times.append(elem.firstChild.data)
    for elem in hr:
        hrs.append(int(elem.firstChild.data)) 
    for elem in ele:     
        eles.append(float(elem.firstChild.data))
        
    if len(hrs)>0:
        data = {'Datetime': times[1:],
                'Latitude':lats,
                'Longitude':longs,
                'Altitude':eles,
                'Distance':0,
                'Heart rate':hrs}
    else:
        data = {'Datetime': times[1:],
                'Latitude':lats,
                'Longitude':longs,
                'Altitude' : eles,
                'Distance':0,
                'Heart rate':0
                }
    #make data frame from dictionary
    data_df = pd.DataFrame(data=data)
    data_df['Datetime'] = pd.to_datetime(data_df.loc[:,'Datetime'],utc=True)
    data_df.set_index('Datetime', inplace=True)
    data_df.index = data_df.index.tz_convert('Australia/Sydney')
    data_df = data_df.sort_index()
    return(data_df)


# In[4]:


VERSION = '0.1.0'


class StravaScraper(object):
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent
    USER_AGENT = "strava_scraper/%s" % VERSION

    HEADERS = {'User-Agent': USER_AGENT}

    BASE_URL = "https://www.strava.com"

    URL_LOGIN = "%s/login" % BASE_URL
    URL_SESSION = "%s/session" % BASE_URL
    URL_DASHBOARD = "%s/dashboard" % BASE_URL
    

    is_authed = False

    def __init__(self, email, password, user_id):
        self.email = email
        self.password = password
        self.session = requests.Session()
        self.user_id = user_id
        self.gpx_files = []
        self.df = pd.DataFrame()

    def get_page(self, url):
        response = self.session.get(url, headers=StravaScraper.HEADERS)
        response.raise_for_status()
        return response

    def login(self):
        response = self.get_page(StravaScraper.URL_LOGIN)
        soup = BeautifulSoup(response.content, 'html.parser')
        utf8 = soup.find_all('input',
                             {'name': 'utf8'})[0].get('value').encode('utf-8')
        token = soup.find_all('input',
                              {'name': 'authenticity_token'})[0].get('value')
        data = {
            'utf8': utf8,
            'authenticity_token': token,
            'plan': "",
            'email': self.email,
            'password': self.password,
        }
        response = self.session.post(StravaScraper.URL_SESSION,
                                     data=data,
                                     headers=StravaScraper.HEADERS)
        response.raise_for_status()
        
        # Simulate that redirect here:
        response = self.get_page(StravaScraper.URL_DASHBOARD)
        response_soup = BeautifulSoup(response.content,'html.parser')
                
        #find athlete url
        athlete_url = 0
        all_links = response_soup.find_all('a')
        for link in all_links:
            if 'My Profile' in link:
                athlete_url = self.BASE_URL + link.get('href') + '/training/log'               
        print('athlete_url:', athlete_url)
        
        
        all_urls = [self.BASE_URL +'/athlete/calendar/2018', self.BASE_URL +'/athlete/calendar/2019']
        activities = []
        #find activities from training log 
        #here we can use "athlete_url" instead of "year_url"
        #this gives us limited results though (as not all the activities are shown in the fetched page)
        for year_url in all_urls:
            training_log = self.session.get(year_url)
            training_log_soup = BeautifulSoup(training_log.content,'html.parser')
            #print('training log:', type(training_log.content.decode('UTF-8')))
        #print(training_log_soup.prettify())
            decoded_log = training_log.content.decode('UTF-8')
            find_ids = [m.start() for m in re.finditer('"id":', decoded_log)]
            for index_start in find_ids:
                teststr = decoded_log[index_start:index_start+22]
                if "name" in teststr:
                    result_id = teststr[5:-7].strip(' ')
                    #print('activity id:', result_id, type(result_id))
                    activities.append(result_id)

        #make sure that we have the correct num of activities
        unique_activities = set(activities)
        print('no of activities:', len(unique_activities))
        
        for activity_id in unique_activities:
            new_url = self.BASE_URL + '/activities/' + activity_id + '/export_gpx.json'
            #print(new_url)
            test_act = self.session.get(new_url) 
            test_act_soup = BeautifulSoup(test_act.content,'html.parser')
            self.gpx_files.append(test_act.content)
                        
        self.is_authed = True
    
    def export_gpx(self):
        for item in range(0, len(self.gpx_files)):
            this_df = import_gpx(self.gpx_files[item].decode('utf-8'))
            if "gpxtpx" in self.gpx_files[item].decode('utf-8'):
                print(this_df.index[0])
            else:
                print(this_df.index[0])
                print('no heart rate')
            #assign session
            this_df = this_df.assign(Session=0)
            collection_date = (this_df.index[0].day, this_df.index[0].month, this_df.index[0].year, this_df.index[0].hour)
            session_id = str(self.user_id) + '_' + this_df.index[0].strftime("%d-%m-%Y %H:%M")  
            if 'Session' in self.df.columns:
                if session_id in self.df['Session'].unique():
                    print('is already: ', session_id)
            this_df.loc[:, 'Session'] = session_id
            #assign session-end
            self.df = self.df.append(this_df)
        self.df = self.df.sort_index()
        self.df = self.df.assign(UserID = 0)
        self.df.loc[:,'UserID'] = self.user_id 
        return(self.df)
    
    
    def merge_sessions(self):
        if not self.df.empty:
            self.df = self.df[~self.df.index.duplicated(keep='first')]
            if 'Session' in self.df.columns:
                sessions = self.df['Session'].unique().copy()
                for i in range(0, len(sessions)):
                    item = sessions[i]
                    df_a = self.df[self.df['Session']==item]
                    if i>0:
                        pos = self.df.index.get_loc(df_a.index[0])
                        session_a = self.df.loc[self.df.index[pos],'Session']
                        session_b = self.df.loc[self.df.index[pos-1],'Session']
                        #print(session_a, session_b)
                        df_b = self.df[self.df['Session']==session_b]
                        #check how many minutes have passed
                        #print(df_a.index[0]-df_b.index[-1], (df_a.index[0]-df_b.index[-1]).total_seconds()/60)
                        minutes_passed = (df_a.index[0]-df_b.index[-1]).total_seconds()/60
                        if minutes_passed <15:
                            self.df.loc[df_a.index, 'Session'] = session_b
            return(self.df)

def scrape_movement_data(email,password, userid):
    #import sys

    #email,password = "demdritsa@gmail.com", "YYZeredjum31"
    
    scraper = StravaScraper(email, password, userid)
    scraper.login()
    df = scraper.export_gpx()
    df = scraper.merge_sessions()
    return(df)


# In[ ]:




