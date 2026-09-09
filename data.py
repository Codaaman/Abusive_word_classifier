import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
df=pd.read_csv(r"D:\Torch_frame_work\clean_data.csv")


X=df.iloc[:5000,1]
Y=df.iloc[:5000,2]

x0_train,x0_test,y0_train,y0_test=train_test_split(X,Y,train_size=0.8,random_state=42)

d={}
d["text"]=x0_train.tolist()
d["label"]=y0_train.tolist()
# print(d)
# print(x0_train.shape)
# print(x0_test.shape)

# print(y0_train.shape)
# print(y0_test.shape)

# data=DataLoader([(x,y) for x,y in zip(x0_test.tolist(),y0_test.tolist())],batch_size=16,shuffle=True)

# for text,label in list(data):
#     print(label,label.size())



def trainning():
    data=DataLoader([(x,y) for x,y in zip(x0_train.tolist(),y0_train.tolist())],batch_size=50,shuffle=True)
    return list(data)


def testing():
    data=DataLoader([(x,y) for x,y in zip(x0_test.tolist(),y0_test.tolist())],batch_size=50,shuffle=True)
    return list(data)


if __name__=="__main__":
    
    trainning()
    testing()


       