#!/usr/bin/env python
# coding: utf-8

# In[83]:


# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.autograd import Variable
import random


# In[84]:


class countyData(Dataset):
    def __init__(self,shuffle = True):
      #  if shuffle = True
        self.dataframe = pd.read_csv('100H-100L-Trainer.csv',index_col = 0,na_values = ['N','-','2,500-','(X)','250,000+']).fillna(value = 0)#.values[:58,:]
        self.dataframe1 = pd.read_csv('100H-100L-Tester.csv',index_col = 0,na_values = ['N','-','2,500-','(X)','250,000+']).fillna(value = 0)#.values[:58,:]
        indexes = list(range(len(self.dataframe)))
        index = list(range(len(self.dataframe1)))
        if shuffle == True:
            random.shuffle(indexes)
        training_size = int(len(indexes))
        print('Training Size: ',training_size)
        training_indexes = indexes[:training_size]
        testing_size = int(len(index))
        print('Testing Size: ',testing_size)
        test_indexes = index[:testing_size]
        
        target = self.dataframe['LISA_CL']
        self.count = np.array([0 for x in np.unique(target).astype('float32')])
        for label in target:
            self.count[label] += 1
        
        print(self.count)
        print(self.count/len(target))
        
        self.train_sampler = SubsetRandomSampler(training_indexes)
        self.test_sampler = SubsetRandomSampler(test_indexes)
        
        
       # print(self.dataframe.head())
    def __getitem__(self,index):
        row = self.dataframe.iloc[index].values
#         print(row[3])
        target = row[1].astype('long')
        inpt = row[2:].astype('float32')
        inpt = np.expand_dims(inpt, axis=0)
        #CNN pass the kernel over the 1D  row
        #The filters are created by kernel and then the data is trained based off those features
#         print(type(row))
#         print(type(target))

        
        return inpt, target
    def __len__(self):
        return len(self.dataframe)
    
data = countyData()
print(data.dataframe.shape, data.__getitem__(0)[0].shape)
print(len(data))
train_loader = DataLoader(data, sampler=data.train_sampler,batch_size = 1)
test_loader = DataLoader(data,sampler=data.test_sampler)
print(len(train_loader))
print(len(test_loader))


# In[85]:


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self,).__init__()
        self.conv1 = nn.Conv1d(1,16, kernel_size=3)
        self.conv2 = nn.Conv1d(16,32, kernel_size=3)
        self.conv3 = nn.Conv1d(32,64,kernel_size=3)
        self.fc1 = nn.Linear(1664,output_size)
    #    self.fc2 - nn.Linear(78976,output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(1, -1) #Flattening the batach size and everything else into one layer
        return self.fc1(x)
    
input_size = data.__getitem__(0)[0].shape
output_size = 2
model = Model(input_size[0], output_size)
print(model)


# In[86]:


learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
loss = torch.nn.CrossEntropyLoss(torch.as_tensor(data.count.astype('float32') / len(data)))


# In[87]:


import matplotlib.pyplot as plt
rightig  = [0,0] #target
zahlen = [0,0] #counts
werden = [0,0] #predictions
true_pos = []
true_neg = []
false_pos = []
false_neg = []
for e in range(3):
    correct = 0.0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        
        y_pred = model(data)
        pred = y_pred.data.max(1)[1]  # get the index of the max log-probability

        out_loss = loss(y_pred, target)
        out_loss.backward()
        optimizer.step()
    
        print('Loss :', out_loss.item())
        
        
        if not pred.eq(target.data):
          #  print(y_pred)
            pass
        
        
        correct += pred.eq(target.data).sum()
        print("Batch Accuracy:", 100. * correct.item() / len(train_loader.dataset))
        
        
        for tg, p in zip(target,pred):
            rightig[tg] += int(tg==p)
            zahlen[tg]+=1
            werden[p]+=1
        
        if(p == tg and p == 1):
            tp +=1
        if(p == tg and p == 0 ):
            tn+=1
        if(p != tg and p == 1):
            fp+=1
        else :
            fn+=1
        #print(tp,tn,fp,fn)
    print("Epoch Accuracy:", 100. * correct.item() / len(train_loader.dataset))


# In[92]:


print('Testing Data_______________________')
rightig  = [0,0] #target
zahlen = [0,0] #counts
werden = [0,0] #predictions
testing_true_pos = []
testing_true_neg = []
testing_false_pos = []
testing_false_neg = []
for e in range(3):
    correct1 = 0.0
    ttp = 0
    ttn = 0
    tfp = 0
    tfn = 0
    for data, target in test_loader:
        optimizer.zero_grad()
        
        y_pred1 = model(data)
        pred1 = y_pred1.data.max(1)[1]  # get the index of the max log-probability

        out_loss = loss(y_pred1, target)
        print('Loss :', out_loss.item())
        
        
        if not pred1.eq(target.data):
          #  print(y_pred)
            pass
        
        
        correct1 += pred1.eq(target.data).sum()
       # print("Epoc Accuracy:", 100 * correct1.item() / len(test_loader.dataset))
        for tg, x in zip(target,pred1):
            rightig[tg] += int(tg==x)
            zahlen[tg]+=1
            werden[x]+=1
        
        if(x == tg and x == 1):
            ttp +=1
        if(x == tg and x == 0 ):
            ttn+=1
        if(x != tg and x == 1):
            tfp+=1
        else :
            tfn+=1
            
    print("Epoch Accuracy:", 100 * correct1.item() / len(test_loader.dataset))

        
     

        
    testing_true_pos.append(ttp)
    testing_true_neg.append(ttn)
    testing_false_pos.append(tfp)
    testing_false_neg.append(tfn)


# In[93]:


#print("Epoch Accuracy:", 100. * correct.item() / (len(train_loader)*3))


# In[94]:


print('True Pos: ', tp, '\nTrue Negs:', tn, '\nFalse Pos:', fp, '\nFalse Negs:', fn)
cm = tp , tn, fp ,fn
print(cm)


# In[95]:


print('True Pos: ', ttp, '\nTrue Negs:', ttn, '\nFalse Pos:', tfp, '\nFalse Negs:', tfn)
cm = ttp , ttn, fp ,fn
print(cm)


# In[56]:


print(testing_true_pos)


# In[ ]:




