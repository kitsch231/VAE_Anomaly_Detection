import os.path
import torch
import time
'''换网络运行只需要更换self.mynet=这个参数即可，其他根据情况微调'''

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5     # 随机失活
        self.require_improvement = 20000  # 若超过2000batch效果还没提升，则提前结束训练
        self.num_epochs =100  # epoch数
        self.learning_rate =1e-3#其他层的学习率

        self.input_dim = 18
        self.hidden_dim = 128
        self.latent_dim = 16
        self.embedding_dim = 4# 嵌入层的维度
        self.num_categories = 7

        self.batch_size = 16# mini-batch大小，看显存决定
        if not os.path.exists('model'):
            os.makedirs('model')

        self.save_path = 'model/'+'vae.pt'##保存模型的路径
        self.log_dir= './log/'+'/'+str(time.time())#tensorboard日志的路径


