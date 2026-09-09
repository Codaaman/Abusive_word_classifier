import torch
from main import Boss
import gc
device=torch.device("cuda")
from transformers import AutoTokenizer

model_name=r"D:\HuggingFace_cache\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
tokenizer=AutoTokenizer.from_pretrained(model_name)

model=Boss().to(device)
model_state=torch.load(r"D:\Torch_frame_work\abusive_model_2.pth",map_location=device)
model.load_state_dict(model_state)
model.eval()

print("going to predict....")



while True:
    text=input("please comment : ")
    tk=tokenizer(text,return_tensors='pt',padding=True,truncation=True)

    with torch.no_grad():
        inputs=tk["input_ids"].to(device)
        pridict=model(inputs)
        _,p=torch.max(pridict,1)
        print(p[0].item())
        torch.cuda.empty_cache()
        gc.collect()