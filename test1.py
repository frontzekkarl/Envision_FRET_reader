#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:30:55 2018

@author: karlfrontzek
"""

import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt


# In[14]:
filename=input('enter filename with extension: ')
#read csv
data=pd.read_csv(filename,sep="\t")
#exclude columns right of plate
data=data.drop(data.columns.to_series()["Humidity":"date"],axis=1) 
data.columns=["row","1","2","3","4","5","6","7","8","9","10","11","12"]

#%%
# create dictionary with information for array
array_dict={}
# retrieve number of plates
array_dict['plates']=pd.to_numeric(data['row'][data['row'].astype(str).str.isdigit()].max())
# retrieve number of repeats per plate
# get range of plates
plate_range=list(range(1,array_dict['plates']+1,1))
repeats=[]
for x in plate_range:
        repeat_nr=data.loc[data['row']==str(x)]
        max_series=pd.to_numeric(repeat_nr['1'].loc[repeat_nr['1'].astype(str).str.isdigit()]).max()
        repeats.append(max_series)
array_dict['repeats']=repeats

print('The current run has {} plates and {} repeats per plate(s) {}, respectively'.format(array_dict['plates'],array_dict['repeats'],plate_range))