import torch
import time
from datetime import timedelta
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from config import Config
import json
import joblib

class My_Dataset(Dataset):
    def __init__(self,path,config):#### 读取数据集
        #启用训练模式，加载数据
        self.df = pd.read_csv(path)
        self.config=config
        self.scaler = joblib.load('scaler.joblib')

    def __getitem__(self, idx):
        num_input=self.df.iloc[idx,1:]
        # 将数据转换为 DataFrame 以保持特征名称
        num_input = pd.DataFrame([num_input], columns=self.df.columns[1:])
        num_input = self.scaler.transform(num_input)  # 使用 scaler 进行归一化
        num_input = torch.tensor(num_input, dtype=torch.float32).squeeze(0)
        #print(num_input.shape)

        cat_input=self.df.iloc[idx,0]
        cat_input=torch.tensor(cat_input, dtype=torch.long)
        #print(cat_input)
        return num_input.to(self.config.device), cat_input.to(self.config.device)


    def __len__(self):
        return len(self.df)#总数据长度

def get_time_dif(start_time):

    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__=='__main__':
    config=Config()
    train_data=My_Dataset('data/train.csv',config)
    train_iter = DataLoader(train_data, batch_size=1)
    n=0
    for x,y in train_iter:

        print(x)
        print(y)
        # #print(y)
        print('************')
        # break