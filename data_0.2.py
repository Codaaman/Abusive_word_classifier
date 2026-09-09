import pandas as pd
import numpy as np





df=pd.read_csv(r"D:\Torch_frame_work\clean_data.csv")



d1=[]
d2=[]




for k in df.iloc[:5000,1]:
    d1.append(k)
    #d1["text"]=k


# for i in df.iloc[:250,1]:
#     d1.append(i)
    #d1["text"]=k

for k in df.iloc[:5000,2]:

    d2.append(k)

# for i in df.iloc[:250,2]:
#     d2.append(i)


for k in df.iloc[57011:61374,1]:
    d1.append(k)


for k in df.iloc[57011:61374,2]:
    d2.append(k)





for k in df.iloc[44680:44999,1]:
    d1.append(k)


for k in df.iloc[44680:44999,2]:
    d2.append(k)



for k in df.iloc[40000:40319,1]:
    d1.append(k)


for k in df.iloc[40000:40319,2]:
    d2.append(k)











# d2["label"]=df.iloc[:250,2]
# d2["label"]=df.iloc[2500:2750,2]


#d3=d1.update(d2)
df=pd.DataFrame({"text":d1,"label":d2})


df.to_csv("./hindienglish_2.csv")


print(df)



