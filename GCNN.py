#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import pandas as pd
import numpy as np
import pysal as ps
import geopandas as geo
 
shape ='counties.shp'



file = geo.read_file(shape)
print(file)

RookWeight = ps.rook_from_shapefile(shape)
RookWeightMatrix , ids = RookWeight.full()
RookWeightMatrix


# In[16]:


class countydata(Dataset):
    def __init__(self,shuffle = True):
        adjacency = RookWeightMatrix
        dataframe = pd.read_csv('NORMALIZED FEATURES-2.csv',na_values = ['N','-','2,500-','(X)']).fillna(value=0).values[:,:]
        #print(adjacency.size)
        ind = dataframe[:,2].astype('int')
       # print(adjacency[ind])
        
        indexes = list(range(len(dataframe)))
        #if shuffle == True:
           # random.shuffle(indexes)
        training_size = int(len(dataframe) * .7)
        training_indexes = indexes[:training_size]
        print(len(training_indexes))
        test_indexes = indexes[training_size:]
        print(len(test_indexes))
        #
        self.a = adjacency
        #print(self.a.size)
        self.y = np.array(dataframe[:,1]).astype('float32')
        #Risk
        self.v = np.array(dataframe[:,2:]).astype('float32')
        #print(self.v.size)
        #Every feature afterwards
      
        self.count = np.array([0 for x in np.unique(self.y).astype('float32')])
        for label in self.y:
                self.count[int(label)]+=1
        print(self.count/len(self.y))
        self.train_sampler = SubsetRandomSampler(training_indexes)
        self.test_sampler = SubsetRandomSampler(test_indexes)
        
    def __len__(self):
        return 1 #temp 1 example
    def __getitem__(self, index):
        #index is index of the sample
        #select what data example we want to pass
        #v is features, a is adj. matrix, y is the labels for counties
     #   row = self.dataframe.iloc[index].values
     #   inpt = row[2:].astype('float32')
     #   inpt = np.expand_dims(inpt, axis = 0)
     #   target = row[1].astype('long')
        
        return [self.v, self.a, self.y]

data = countydata()
print(len(data))

train_loader = DataLoader(data, sampler=data.train_sampler,batch_size = 1)
test_loader = DataLoader(data,sampler=data.test_sampler)   
#print('Traing Loader Size:' , test_loader.size())


# In[4]:


class graphgcnn(nn.Module):
    #nodes is counties and features are features lmao
    def __init__(self, nb_features_pernode, nb_features):
        super(graphgcnn,self).__init__()
        self.linear = nn.Linear(nb_features_pernode, nb_features)
        #self.act = nn.Relu()
    def forward(self, v, a):
        #message passing between nodes
        v_prime = torch.matmul(a,v)
        #feature mapping on a per node basis
        v_prime = self.linear(v_prime)
        #update v_prime and pass through act function
        v_prime = F.relu(v_prime)
        return v_prime

class modelgcnn(nn.Module):
    def __init__(self):
        super(modelgcnn,self).__init__()
        self.conv = graphgcnn(34,32)
        self.conv2 = graphgcnn(32,64)
        self.classer = nn.Linear(64,2)
    def forward(self, v, a):
       # print(v.size())
        #print(a.size())
        y = self.conv(v,a)
        y = self.conv2(y,a)
        y = self.classer(y)
        return y
print(modelgcnn)
print(graphgcnn)


# In[ ]:





# In[5]:


def main():
    correct = 0
    total = 0
    data = countydata()
    
  
    
    #batch_size is default value
#     dataloader = DataLoader(training_size, batch_size = 1, shuffle = True, num_workers = 1)
    graphgcnn = modelgcnn()
    #categorical crossentrophy
    lossfunct = nn.CrossEntropyLoss(torch.as_tensor(data.count.astype('float32')/len(data)))
    #lr = learning rate default is .0001
    optimizer = Adam(graphgcnn.parameters(),lr = .0001)
    min_loss = 0.0
    for e in range(10):
        preds = []
        trues = []
        epoch_loss = 0.0
        
    
        
    
        # Training Loop
        for i,data in enumerate(train_loader):
            v,a,y = data
            #print(v.size(),a.size(),y.size())
            #casting v and a to a float, forward pass
            out = graphgcnn(v.float(),a.float())
            #print('Out', out.size())
            out2 = out.view((3141,2))
            y2 = y.view((3141))
            optimizer.zero_grad()
            loss = lossfunct(out2,y2.long())
            loss.backward()
            optimizer.step()
            if e > 3:
                print('Loss: ',loss.detach().numpy())
                
            out2 = F.softmax(out2,dim = 1)
            #detach removes from model (not in graph anymore)
            preds.append(out2.detach().numpy())
            trues.append(y2.detach().numpy())
            
            epoch_loss += loss.item()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_temp = []
        for t in preds:
            preds_temp.append(np.argmax(t))
        preds = np.array(preds_temp)
        correct = np.sum(preds == trues)
        totals = len(preds)
        print(' Training Accuracy: ', 100 *correct / totals)

    
    
        for k in range(1):
            pred2 = []
            true2 = []
            totals1 = 0
            correct1 = 0
            epoch_loss1 = 0.0
        
            for x,data in enumerate(test_loader):
                v,a,y = data
            
               # print(v.size(),a.size(),y.size())
                #casting v and a to a float, forward pass
                out = graphgcnn(v.float(),a.float())
               # print('Out',out)
                out2 = out.view((943,2))
                y2 = y.view((943))
                loss = lossfunct(out2,y2.long())

                if e > 3:
                    print('Loss: ',loss.detach().numpy() )

                out2 = F.softmax(out2,dim = 1)
                #detach removes from model (not in graph anymore)
                #print(type(preds))
                pred2.append(out2.detach().numpy())

                #print(pred2)
                true2.append(y2.detach().numpy())

                epoch_loss1 += loss.item()

            pred2 = np.concatenate(pred2, axis=0)
            true2 = np.concatenate(true2, axis=0)
            pred2_temp = []

            for t in pred2:
                pred2_temp.append(np.argmax(t))
            pred2 = np.array(pred2_temp)
            correct1 = np.sum(pred2 == true2)
            totals1 = len(preds)
            print(' Epoch Accuracy: ', 100 *correct1 / totals1)

   #    print('Predict: ', pred2[:20])
    #    print('Trues: ', true2[:20])


# In[ ]:





# In[6]:


main()


# def classifier():
#     dataset = countydata()
#     #batch_size is default value
#     dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 1)
#     graphgcnn = modelgcnn()
#     #categorical crossentrophy
#     lossfunct = nn.CrossEntropyLoss()
#     #lr = learning rate default is .0001
#     optimizer = Adam(graphgcnn.parameters(),lr = .0001)
#     
#     for e in range(0,100):
#         preds = []
#         trues = []
#     
#         for i,data in enumerate(dataloader):
#             v,a,y = data
#             #requires_grad true important bc keeps gradients
#             v_att = torch.autograd.Variable(v,requires_grad = True)
#             v_att.retain_grad
#             
#             #print(v.size(),a.size(),y.size())
#             #casting v and a to a float, forward pass
#             out = graphgcnn(v_att.float(),a.float())
#             out2 = out.view((58,2))
#             y2 = y.view((58))
#             #optimizer.zero_grad()
#             loss = lossfunct(out2,y2.long())
#             loss.backward()
#             print(v_att*v_att.grad)
#             #optimizer.step() only for training
#             print('Loss: ',loss.detach().numpy())
#             #out2 = F.softmax(out2,dim = 1) TRAINING
#             #detach removes from model (not in graph anymore)
#             preds.append(out2.detach().numpy())
#             trues.append(y2.detach().numpy())
#         preds = np.concatenate(preds, axis=0)
#         trues = np.concatenate(trues, axis=0)
#                 
#         trues_temp = []
#         
#         for t in preds:
#             trues_temp.append(np.argmax(t))
# 
#         preds = np.array(trues_temp)
#         
#     

# In[ ]:


classifier()


# In[ ]:





# In[ ]:





# In[ ]:




