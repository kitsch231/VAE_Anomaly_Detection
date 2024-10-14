import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler

df=pd.read_excel('./data/data.xlsx')
df=df.fillna(0)

# 创建行业类别的映射
category_dict = {'A': 0, 'B': 1, 'C': 2, 'E': 3, 'F': 4, 'L': 5, 'S': 6}
df['611-专业类别1'] = df['611-专业类别1'].map(category_dict)

# print(df)
train,_=train_test_split(df,test_size=0.2,shuffle=True,stratify=df['611-专业类别1'])
test,val=train_test_split(_,test_size=0.5,shuffle=True,stratify=_['611-专业类别1'])

# 训练 StandardScaler 模型
scaler = MinMaxScaler()
num_columns = df.columns[1:]  # 数值列
scaler.fit(df[num_columns])

# 保存归一化模型
import joblib
joblib.dump(scaler, 'scaler.joblib')


def scale_numeric_columns(df, scale_min=3, scale_max=10, min_scaling_indices=1, max_scaling_indices=3):
    """
    随机对指定 DataFrame 的数值型数据列进行缩放处理。

    参数:
    df : pandas.DataFrame
        输入的 DataFrame，包含数值型数据列和分类列。
    scale_min : int
        随机缩放因子的最小值，默认为 3。
    scale_max : int
        随机缩放因子的最大值，默认为 10。
    min_scaling_indices : int
        每行随机选择缩放的指标的最小数量，默认为 1。
    max_scaling_indices : int
        每行随机选择缩放的指标的最大数量，默认为 3。

    返回:
    pandas.DataFrame
        处理后的 DataFrame，数值型数据列被随机缩放。
    """
    # 假设数值型数据在第一列后
    num_columns = df.select_dtypes(include=[np.number]).columns[1:]  # 动态选择数值型列
    print(num_columns)

    # 对每一行进行处理
    for idx, row in df.iterrows():
        # 随机选择要缩放的指标数量
        num_scaling_indices = np.random.randint(min_scaling_indices, max_scaling_indices + 1)
        scaling_indices = np.random.choice(num_columns, num_scaling_indices, replace=False)  # 随机选择指标

        # 对选中的指标进行缩放
        for col in scaling_indices:
            scaling_factor = np.random.uniform(scale_min, scale_max)  # 随机生成缩放因子
            df.at[idx, col] *= scaling_factor  # 使用 at 修改 DataFrame 中的值

    return df  # 返回处理后的 DataFrame


t1,t2=train_test_split(test,test_size=0.5,shuffle=True)
newt1=scale_numeric_columns(t1,0.05,20,3,9)
newt1['label']=[1]*len(newt1)
t2['label']=[0]*len(t2)
newtest=pd.concat([newt1,t2],axis=0)
print(newtest)
train.to_csv('data/train.csv',index=None)
val.to_csv('data/val.csv',index=None)
test.to_csv('data/test.csv',index=None)
newtest.to_csv('data/newtest.csv',index=None)

