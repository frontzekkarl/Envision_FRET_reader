
# coding: utf-8

# In[1]:

#%% import packages
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import itertools
import seaborn as sns
%matplotlib inline
import os
os.chdir('/home/kajo/Dokumente/automation_camp')

#%% enter and read filenames
#filename=input('enter filename with extension: ')
#read csv
data=pd.read_excel('camp_3rd.xls',sep="\t")
#exclude columns right of plate
# for initial csv save: 
# data=data.drop(data.columns.to_series()["Humidity":"date"],axis=1) 
# for xls save
data=data.drop(data.columns.to_series()['Unnamed: 13':'Unnamed: 16'],axis=1)
data.columns=["row","1","2","3","4","5","6","7","8","9","10","11","12"]
colnames=list(range(1,13,1))
rownames=['A','B','C','D','E','F','G','H']

#%%
# create dictionary with information for array
# for initial csv save 
array_dict={}
# retrieve number of plates
array_dict['plates']=pd.to_numeric(data['row'][data['row'].astype(str).str.isdigit()].max())
# retrieve number of repeats per plate
# get repeat numbers of plates
plate_range=list(range(1,array_dict['plates']+1,1))
repeats=[]
for x in plate_range:
        repeat_nr=data.loc[data['row']==str(x)]
        max_series=pd.to_numeric(repeat_nr['1'].loc[repeat_nr['1'].astype(str).str.isdigit()]).max()
        repeats.append(max_series)
array_dict['repeats']=repeats

#set number of channels
array_dict['channels']=2

#calculate number of total reads
tot_reads=sum(array_dict['repeats'])*array_dict['channels']
print('This assay contains {} plates with a total of {} repeats and {} channels'.format(array_dict['plates'],sum(array_dict['repeats']),array_dict['channels']))
#%%
# create dictionary with information for array
#for initial xls save from 3rd fret experiment KFEX220
array_dict={}
# retrieve number of plates
array_dict['plates']=pd.to_numeric(data['row'][data['row'].astype(str).str.isdigit()].max())
# retrieve number of repeats per plate
# get repeat numbers of plates
plate_range=list(range(1,array_dict['plates']+1,1))
repeats=220
#for x in plate_range:
 #       repeat_nr=data.loc[data['1']==str(x)]
  #      max_series=pd.to_numeric(repeat_nr['1'].loc[repeat_nr['1'].astype(str).str.isdigit()]).max()
   #     repeats.append(max_series)
array_dict['repeats']=repeats

#set number of channels
array_dict['channels']=2

#calculate number of total reads
tot_reads=repeats*array_dict['channels']
print('This assay contains {} plates with a total of {} repeats and {} channels'.format(array_dict['plates'],repeats,array_dict['channels']))

#%% read tables in xarray Dataset - originally saved as csv
def add_to_dataset():
    global ds
    #gather first row per plate
    row_id_start=[(9+(i*20)) for i in range(0,tot_reads,1)]
    #create empty np array with size of reads x rows x columns
    foo=np.empty([tot_reads,8,12])
    #transfer reads to xarray
    for i in range(0,tot_reads,1):
        foo[i]=data.iloc[row_id_start[i]:row_id_start[i]+8,1:13]
    runs=list(range(0,tot_reads,1))
    ds=xr.Dataset({'plates':(['run','row','column'],foo)}, coords={'run':runs,'row':rownames,'column':colnames})
    return 
add_to_dataset()
#%% originally saved as xlsx
def add_to_dataset():
    global ds
    #gather first row per plate
    row_id_start=[(10+(i*20)) for i in range(0,tot_reads,1)]
    #create empty np array with size of reads x rows x columns
    foo=np.empty([tot_reads,8,12])
    #transfer reads to xarray
    for i in range(0,tot_reads,1):
        foo[i]=data.iloc[row_id_start[i]:row_id_start[i]+8,1:13]
    runs=list(range(0,tot_reads,1))
    ds=xr.Dataset({'plates':(['run','row','column'],foo)}, coords={'run':runs,'row':rownames,'column':colnames})
    return 
add_to_dataset()
#%% calculate NET FRET - optimized for columns 2-4 (->check columns in for loop in net_fret)
#well=np.array([[r_ow,0],[r_ow,1],[r_ow,2]])
time=[x*(19/60) for x in range(1,int((tot_reads/2)+1))]
t_array=np.array(time)
rows=np.array([0,1,2,3,4,5,6,7])
replicates=[0,1,2]
listiter=list(itertools.product(rows,replicates))
def net_fret():
        global netfret_normalized
        netfret_normalized=np.random.rand(len(rows),len(replicates),len(range(0,tot_reads-1,2)))
        normalized=np.random.rand(len(rows))
        nf_normal=pd.DataFrame(index=time)
        for (x,p) in listiter:
            netfret_normalized[x,p]=[np.float(ds.isel(run=i+1,row=[x],column=[p+1]).to_array().values)/np.float(ds.isel(run=i,row=[x],column=[p+1]).to_array().values) for i in range(0,tot_reads-1,2)]
        normalize=[netfret_normalized[x,0:3,:40].mean() for x in rows]
        netfret_normalized=[1/(netfret_normalized[x]/normalize[x]) for x in rows]
        return 
    
net_fret()
nff=np.array([netfret_normalized[i].mean(axis=0) for i in rows])
#normlizd=1/(nf/normalized)
#normlizd.to_csv('H1-3.csv')
#%%
g, axes=plt.subplots(2,4)
sns.set_style('darkgrid')
for i in rows:
    sns.regplot(x=t_array,y=nff[i],fit_reg=False,ax=axes[i])
#sns.regplot(x=t_array,y=nff,fit_reg=False)
#netfret_normalized=np.array(netfret_normalized)

#%%
#plt.figure(figsize=(5,6))
names=['mPrP$_{23-50}$ 10$^{-9}$M','hPrP$_{23-231}$ 10$^{-8}$M','hPrP$_{23-231}$ 10$^{-9}$M','hPrP$_{23-231}$ 10$^{-10}$M','hPrP$_{23-231}$ 10$^{-11}$M','hPrP$_{23-231}$ 10$^{-12}$M','hPrP$_{23-231}$ 10$^{-13}$M','assay buffer']
fig, axes =plt.subplots(nrows=2,ncols=4,figsize=(16,10))
axes=axes.flatten()
for i in range(0,8):
    axes[i].plot([t_array[40],t_array[40]],[0.8,1.25],'r-')
    #axes[i].plot([0,t_array[-1]],[1,1],'-')
    axes[i].scatter(x=t_array, y=nff[i])
    axes[i].set(title=names[i])
    axes[i].set_xlabel('time [min]')
    axes[i].set_ylabel('normalized NET FRET^-1')
    axes[i].set_ylim([0.8,1.25])
#plt.show()
#plt.show()
 #   plt.plot([0,t_array[-1]],[1,1],linewidth=2,ax=axes[i%8])
#%%
import itertools
rows=np.array([0,1,2,3,4,5,6,7]
replicates=[0,1,2]
listiter=list(itertools.product(rows,replicates))
for (x,p) in listiter:
    print("Printing row {} with well {}".format(x,p))
g=sns.regplot(t_array,normlizd_mean,fit_reg=False)
g.set_ylim(0,1.5)
#%% append wells to csv
def append_to_csv():
    tt=np.empty([205,1])
    ts=pd.DataFrame(data=tt)
    ts[1]=tt
    return ts.head()
append_to_csv()