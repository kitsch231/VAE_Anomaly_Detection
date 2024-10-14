import pandas as pd
import torch
from models import *
from config import Config
# 定义加载模型的函数
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import confusion_matrix, classification_report,recall_score,accuracy_score,precision_score,f1_score
from torchviz import make_dot
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import time
# 定义预测推理函数
def predict_anomalies(model_path, scaler_path, input_data, device,config, threshold=0.1):
    """
    加载模型并进行异常检测。

    参数:
    model_path : str
        模型文件的路径。
    scaler_path : str
        scaler文件的路径。
    input_data : pd.DataFrame
        待检测的数据，包含类别和数值型数据。
    device : torch.device
        指定的设备（CPU或GPU）。

    返回:
    pd.DataFrame
        包含原始输入和重构误差的 DataFrame。
    """
    # 加载模型
    # 如果模型保存在状态字典中

    model= VAE(config.input_dim, config.hidden_dim, config.latent_dim,
               config.num_categories,config.embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式

    # 加载Scaler
    scaler = joblib.load(scaler_path)

    # 存储结果
    results = []
    t1=time()

    for _, row in input_data.iterrows():
        # 提取数值数据
        num_input = row[1:-1].values  # 排除类别数据
        cat_input = row[0]  # 获取类别数据

        # 将数值数据转换为 DataFrame 以进行归一化
        num_input_df = pd.DataFrame([num_input], columns=input_data.columns[1:-1])
        num_input_normalized = scaler.transform(num_input_df)  # 归一化
        #print(num_input_normalized)

        # 转换为 Tensor
        num_input_tensor = torch.tensor(num_input_normalized, dtype=torch.float32).to(device)
        cat_input_tensor = torch.tensor(cat_input, dtype=torch.long).to(device).unsqueeze(0)  # 增加维度

        # 进行推理
        with torch.no_grad():
            recon_x, mu, logvar = model(num_input_tensor, cat_input_tensor)

        # 计算重构误差和KL散度
        mse = torch.nn.functional.mse_loss(recon_x, num_input_tensor, reduction='sum')
        logvar_clamped = torch.clamp(logvar, min=-10, max=10)  # 防止 logvar 过大或过小
        kld = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
        total_loss = mse + kld

        # 根据阈值打标签
        label = 1 if total_loss.item() > threshold else 0

        results.append(label)
    t2=time()
    print(t2-t1,'********************')
    return pd.DataFrame(results)


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, cmap="Blues"):
    """
    绘制混淆矩阵图。

    参数:
    - y_true: 实际标签。
    - y_pred: 预测标签。
    - labels: 类别标签，用于设置混淆矩阵的坐标轴。
    - normalize: 是否对混淆矩阵进行归一化。
    - cmap: 颜色映射，默认使用 'Blues'。
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 设置绘图尺寸
    plt.figure(figsize=(8, 6))

    # 使用 Seaborn 绘制热力图
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=labels, yticklabels=labels, cbar=True)

    # 设置标题和坐标轴标签
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('cm.png')
    plt.show()


# 使用示例
input_data = pd.read_csv('data/newtest.csv')  # 加载测试数据
# input_data['label']=[0]*len(input_data)
model_path =  'model/vae.pt'  # 训练好的模型文件路径
scaler_path = 'scaler.joblib'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
# 进行异常检测
anomaly_results = predict_anomalies(model_path, scaler_path, input_data, device,config,threshold=0.4)
res=input_data
res['res']=anomaly_results
# 打印结果
print(res)


plot_confusion_matrix(res['label'], res['res'], [0,1], normalize=True)
class_report = classification_report(res['label'], res['res'])
print("\nClassification Report:")
print(class_report)
res.to_csv('result.csv',index=None)


medf=[]
#按专业类别评价性能
for x in range(0,7):
    y1=res[res['611-专业类别1']==x]['label']
    y2=res[res['611-专业类别1']==x]['res']
    p=precision_score(y1,y2,average='weighted')
    r=recall_score(y1,y2,average='weighted')
    f1=f1_score(y1,y2,average='weighted')
    acc=accuracy_score(y1,y2)
    print(x,len(y1),p,r,f1,acc)
    print('***********************')
    medf.append([x,len(y1),p,r,f1,acc])
medf=pd.DataFrame(medf)
medf.columns=['专业类别','样本量','precision','recall','f1-score','accuracy']
medf.to_csv('分类别评价指标.csv',index=None)




vae= VAE(config.input_dim, config.hidden_dim, config.latent_dim,
               config.num_categories,config.embedding_dim).to(device)
# 定义一个输入数据和类别（用于测试）
x = torch.randn(2, 18).to(config.device)  # 假设输入 1 个样本，维度为 input_dim
category = torch.randint(0, 7, (2,)).to(config.device)  # 随机生成一个类别
# 前向传播，获取输出
recon_x, mu, logvar = vae(x, category)
# 使用 make_dot 生成模型的计算图
model_vis = make_dot((recon_x, mu, logvar), params=dict(vae.named_parameters()))

# 保存图像为 PDF 或 PNG 文件
model_vis.render("vae_model_structure", format="png")  # 或 format="png"