#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pysal as ps
import geopandas as geo
 
shape ='counties-temporal.shp'
shape

file = geo.read_file(shape)
file.head()


# In[2]:


RookWeight = ps.rook_from_shapefile(shape)
RookWeightMatrix , ids = RookWeight.full()
RookWeightMatrix


# In[3]:


#Spatial Lag
#Is a variable that averages the neighboring values of a locaation
#Accounts for the acutocorrelation in the model with the weight matrix
#
data = ps.pdio.read_files(shape)
Rook = ps.rook_from_shapefile(shape)
Rook.transform = 'r'
percent16Lag = ps.lag_spatial(Rook, data.percent16)


# In[4]:


#This is a spatial lag graph of the percentages of suicide for the year 2016. 
#Spatial lag is a form of regression that accounts for the weight matrix of the shape file
#and the dependent veriable that you have chosen.

import matplotlib.pyplot as plt
us = file
percent16LagQ16 = ps.Quantiles(percent16Lag, k =10)
f, ax = plt.subplots(1, figsize=(150, 150))
us.assign(cl=percent16LagQ16.yb).plot(column='cl', categorical=True,         k=10, cmap='OrRd', linewidth=0.1, ax=ax,         edgecolor='white', legend=True)
ax.set_axis_off()
plt.title("Percentage of Suicides 2016 Spatial Lag Deciles")

plt.show()


# In[28]:


percent16 = file.percent16
#Calculation of Moran's I
#Moran’s I global and local measures of spatial autocorrelation
#Moran’s I is a correlation coefficient that measures the overall spatial autocorrelation of your data set. 
#In other words, it measures how one object is similar to others surrounding it.
#-1 is perfect clustering of dissimilar values (you can also think of this as perfect dispersion).
#0 is no autocorrelation (perfect randomness.)
#+1 indicates perfect clustering of similar values (it’s the opposite of dispersion).
#b is the slope of the Moran's I line 
b,a = np.polyfit(percent16, percent16Lag, 1)
f, ax = plt.subplots(1, figsize=(9, 9))

plt.plot(percent16,percent16Lag, '.', color='firebrick')

 # dashed vert at mean of the last year's PCI
plt.vlines(percent16.mean(), percent16Lag.min(), percent16Lag.max(), linestyle='--')
 # dashed horizontal at mean of lagged PCI
plt.hlines(percent16Lag.mean(), percent16.min(), percent16.max(), linestyle='--')

# red line of best fit using global I as slope
plt.plot(percent16, a + b*percent16, 'r')
plt.title('Morans Rook Scatterplot')
plt.ylabel('Spatial Lag of Percentagew in 2016 per County ')
plt.xlabel('Percent of Deaths')
plt.show()


# In[29]:


#Calculating Moran's I for the dataset that is being used
#This Caluculates the Slope of the Red line, AKA the Moran's I value. Along with the seudo P-Value
I_percent16 = ps.Moran(data.percent16.values, Rook)
I_percent16.I, I_percent16.p_sim


# In[ ]:





# In[ ]:





# In[30]:


#Calculating the Local Autocorrelation Statistic
#Autocorrelation is a characteristic of data in which the correlation between the values of the same variables is based on related objects.
#The output depicts teh Moran's I of each county specified and their related P-Value
LMo_percent16 = ps.Moran_Local(data.percent16.values, Rook)
LMo_percent16.Is[0:50],LMo_percent16.p_sim[0:50]


# In[8]:


LMo_percent16 = ps.Moran_Local(data.percent16.values, Rook, permutations=9999)
LMo_percent16.Is[0:50],LMo_percent16.p_sim[0:50]


# In[9]:


LMo_percent16.Is[0:50]


# In[10]:


np.savetxt('Morans I',LMo_percent16.Is[0:50])


# In[11]:


LMo_percent16.p_sim[0:50]


# In[12]:


#TO dertermine which counties have significant data reading
significant = LMo_percent16.p_sim < 0.05
hotspots = LMo_percent16.q==1 * significant
hotspots.sum()
print ('Siginicant Counities withing the United States',hotspots.sum()) 


# In[13]:


data


# In[14]:


data.percent16[hotspots]


# In[15]:


data.percent16[hotspots]


# In[16]:


data[hotspots]


# In[17]:


coldspots = LMo_percent16.q==3 * significant
coldspots.sum()
print ('NON-Siginicant Counities withing the United States',coldspots.sum()) 
data[coldspots]


# In[18]:


data.percent16[coldspots]


# In[24]:


from matplotlib import colors

hmap = colors.ListedColormap(['black', 'red'])
f, ax = plt.subplots(1, figsize=(100, 100))
us.assign(cl=hotspots*1).plot(column='cl', categorical=True,         k=2, cmap=hmap, linewidth=0.2, ax=ax,         edgecolor='white', legend=True)
ax.set_axis_off()
plt.show()


# In[25]:


plt.savefig('RookHotspot.png')


# In[20]:


from matplotlib import colors

hmap = colors.ListedColormap(['black', 'blue'])
f, ax = plt.subplots(1, figsize=(50, 50))
us.assign(cl=coldspots*1).plot(column='cl', categorical=True,         k=2, cmap=hmap, linewidth=0.1, ax=ax,         edgecolor='purple', legend=True)
ax.set_axis_off()
plt.show()


# In[21]:


from matplotlib import colors
hcmap = colors.ListedColormap(['black', 'red','blue'])
hotcold = hotspots*1 + coldspots*2
f, ax = plt.subplots(1, figsize=(50, 50))
us.assign(cl=hotcold).plot(column='cl', categorical=True,         k=2, cmap=hcmap,linewidth=0.1, ax=ax,         edgecolor='black', legend=True)
ax.set_axis_off()
plt.show()


# 

# In[ ]:




