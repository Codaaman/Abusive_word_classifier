import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import numpy as np
from data import trainning,testing
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR



device=torch.device("cuda")


class Block_1(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed=nn.Embedding(151665,256)
        self.rms1=nn.RMSNorm(256)
        self.rms2=nn.RMSNorm(256)
        self.atten=nn.MultiheadAttention(256,8,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(256,1024),
            nn.GELU(),
            nn.Linear(1024,256)
        )
        self.head=nn.Linear(256,256,bias=False)
        self.dropout=nn.Dropout(0.3)


    def forward(self,x):
    
        atten,_=self.atten(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        x=x+self.mlp(self.rms2(x))
        x=self.head(x)
        return x




class Block_2(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed=nn.Embedding(151665,256)
        self.rms1=nn.RMSNorm(256)
        self.rms2=nn.RMSNorm(256)
        self.atten=nn.MultiheadAttention(256,8,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(256,1024),
            nn.GELU(),
            nn.Linear(1024,256)
        )
        self.head=nn.Linear(256,256,bias=False)
        self.dropout=nn.Dropout(0.3)



    def forward(self,x):
        atten,_=self.atten(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        x=x+self.mlp(self.rms2(x))
        x=self.head(x)
        return x


class Block_3(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed=nn.Embedding(151665,256)
        self.rms1=nn.RMSNorm(256)
        self.rms2=nn.RMSNorm(256)
        self.atten=nn.MultiheadAttention(256,8,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(256,1024),
            nn.GELU(),
            nn.Linear(1024,256)
        )
        self.head=nn.Linear(256,256,bias=False)
        self.dropout=nn.Dropout(0.3)



    def forward(self,x):
    
        atten,_=self.atten(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        x=x+self.mlp(self.rms2(x))
        x=self.head(x)
        #x=self.dropout(x)
        return x


class Block_4(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed=nn.Embedding(151665,256)
        self.rms1=nn.RMSNorm(256)
        self.rms2=nn.RMSNorm(256)
        self.atten=nn.MultiheadAttention(256,8,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(256,1024),
            nn.GELU(),
            nn.Linear(1024,256)
        )
        self.head=nn.Linear(256,256,bias=False)
        self.dropout=nn.Dropout(0.3)



    def forward(self,x):
    
        atten,_=self.atten(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        x=x+self.mlp(self.rms2(x))
        x=self.head(x)
        #x=self.dropout(x)
        return x





class Block_5(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed=nn.Embedding(151665,256)
        self.rms1=nn.RMSNorm(256)
        self.rms2=nn.RMSNorm(256)
        self.atten=nn.MultiheadAttention(256,8,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(256,1024),
            nn.GELU(),
            nn.Linear(1024,256)
        )
        self.head=nn.Linear(256,256,bias=False)


    def forward(self,x):
    
        atten,_=self.atten(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        x=self.mlp(x)
        x=x+self.rms2(x)
        x=self.head(x)
        return x


class Block_6(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed=nn.Embedding(151665,256)
        self.rms1=nn.RMSNorm(256)
        self.rms2=nn.RMSNorm(256)
        self.atten=nn.MultiheadAttention(256,8,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(256,1024),
            nn.GELU(),
            nn.Linear(1024,256)
        )
        self.head=nn.Linear(256,256,bias=False)


    def forward(self,x):
    
        atten,_=self.atten(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        x=self.mlp(x)
        x=x+self.rms2(x)
        x=self.head(x)
        return x



class Boss(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed=nn.Embedding(151665,256)
        self.pos_embed=nn.Embedding(405,256)
        self.rms1=nn.RMSNorm(256)
        self.rms2=nn.RMSNorm(256)
        self.atten=nn.MultiheadAttention(256,8,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(256,1024),
            nn.GELU(),
            nn.Linear(1024,256)
        )
        self.head=nn.Linear(256,256,bias=False)
        self.activation=nn.GELU()
        self.activation2=nn.GELU()
        self.dropout=nn.Dropout(0.3)
        self.clasification=nn.Linear(256,2)
        self.b_1=Block_1()
        self.b_2=Block_2()
        self.b_3=Block_3()
        self.b_4=Block_4()
        self.b_5=Block_5()
        self.b_6=Block_6()

    def forward(self,x):
        token=x
        seq_len=x.shape[1]
        postion=torch.arange(0,seq_len,device=device)
        pos_embed=self.pos_embed(postion)
        x=self.embed(token)
        x=x+pos_embed
        atten,_=self.atten(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        x=x+self.mlp(self.rms2(x))
        x=x+self.b_1(x)
        x=x+self.b_2(x)
        x=x+self.b_3(x)
        x=x+self.b_4(x)
        x=x+self.b_5(x)
        x=x+self.b_6(x)
        

        x=self.head(x)
        x=self.dropout(x)
        x=torch.max(x,dim=1)[0]
        x=self.activation(x)
        x=self.clasification(x)

        return x








if __name__=="__main__":

    print("tokinizer initialised")
    model_name=r"D:\HuggingFace_cache\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    print("model initialized")
    model=Boss().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-5,weight_decay=5e-4)
    #seduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3)
    seduler_2=CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-6)
    lossfn=nn.CrossEntropyLoss(label_smoothing=0.1)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    print("ready for traininng")
    model.train()
    train=trainning()
    test=testing()
    valid_total=0
    total_corect=0
    total_loss=0
    val_loss=0
    epoch=100

    train_loss=[]
    validation_loss=[]
    acuracy=[]
    scaler=torch.amp.GradScaler("cuda")
    for k in range(epoch+1):
        c=1
        for text,label in train:
                print(f"\rnumber of text batch process {c} ",end=" ",flush=True)
                tk=tokenizer(text,return_tensors="pt",padding=True,truncation=True)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    result=model(tk["input_ids"].to(device))
                    target=label.to(device)
                    loss=lossfn(result,target)

        
                scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)
                # for p in model.parameters():
                #     if p.requires_grad and p.grad is not None:
                #         p.grad.add(torch.sign(p.data),alpha=0)

                scaler.step(optimizer)
                scaler.update()
                total_loss+=loss.item()
                c+=1
        total_loss/=80
    

        model.eval()
        with torch.no_grad(): 
            for text,label in test:
                tk=tokenizer(text,return_tensors="pt",padding=True,truncation=True)

                with torch.amp.autocast("cuda"):
                    inputs=tk["input_ids"].to(device)
                    target=label.to(device)
                    pridect=model(inputs)
                    loss2=lossfn(pridect,target)

                val_loss+=loss2.item()
                valid_total+=target.size(0)
                _,predict=torch.max(pridect,1)
                total_corect+=(predict==target.to(device)).sum().item()

            
        val_loss/=80
        validation_acuracy=100*total_corect/valid_total
        acuracy.append(validation_acuracy)
        train_loss.append(total_loss)
        validation_loss.append(val_loss)
        #seduler.step(val_loss)
        seduler_2.step()


        print(f"epoch:{k}    loss: {total_loss:4f}    val_loss: {val_loss:4f}  validation_acuracy: {validation_acuracy:4f}")
        torch.cuda.empty_cache
        gc.collect()






    plt.plot(validation_loss,color="orange")
    plt.plot(train_loss,color="green")
    plt.xlabel("Trining_loss")
    plt.ylabel("validation_loss")
    plt.show()

    torch.save(model.state_dict(),"D:/Torch_frame_work/Abusive_word_clasifier/abusive_model.pth")
    torch.cuda.empty_cache()
    gc.collect()

            
                    
