
# coding: utf-8

# In[1]:

#%% import packages
import pandas as pd
import numpy as np

#%% enter and read filenames
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

#create array with tot_reads times 8x12 tables 
fret_data_raw=np.ones((tot_reads,8,12))


#%% read raw FRET values
row_id=9
for i in range(0,tot_reads,1):
    row_id = []
    row_id = 9+(i*20)
    print(row_id)
data["Plate"].value_counts()


# In[141]:
#%% read tables in sublists

# make lists of data rows containing plate rows

df={}
df=pd.DataFrame(data.iloc[9:17,1:13])
df.index=['a','b','c','d','e','f','g','h']
df2={}
df2=pd.DataFrame(data.iloc[29:37,1:13])
df2.index=['a','b','c','d','e','f','g','h']
xr.DataArray(data.iloc[9:17,1:13])
ds=xr.Dataset


#row indices of "background" in "Plate"
background_ind=np.where(data["Plate"]=="Background")
#add 2 for plate number
background_ind=np.array([x+2 for x in background_ind])
background_ind=background_ind.tolist()
background_ind=background_ind[0]


# In[153]:


data["Plate"].iloc[background_ind].value_counts()


# In[158]:


data["end"]


# In[81]:


data.head(50)

