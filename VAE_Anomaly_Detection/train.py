# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import sys
import torch
import numpy as np
from tensorboardX import SummaryWriter
from untils import My_Dataset,get_time_dif
from models import *
from config import Config
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def loss_function(recon_x, x, mu, logvar,beta=1):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # 对 KL 散度中的 logvar 做数值稳定性处理
    logvar = torch.clamp(logvar, min=-10, max=10)  # 防止 logvar 过大或过小
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def train(config, model, train_iter, dev_iter,writer):
    #model.load_state_dict(torch.load(config.save_path))
    start_time = time.time()
    model.train()

    optimizer = torch.optim.Adam(model.parameters() , lr=config.learning_rate)  ## 定义优化器

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,5, gamma=0.9, last_epoch=-1)#每2个epoch学习率衰减为原来的一半

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(config.num_epochs):
        loss_list=[]#承接每个batch的loss
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (batch_num, batch_cat) in enumerate(train_iter):
            recon_batch, mu, logvar=model(batch_num,batch_cat)
            optimizer.zero_grad()
            loss = loss_function(recon_batch, batch_num, mu, logvar)
            loss.backward()
            optimizer.step()


            writer.add_scalar('train/loss_iter', loss.item(),total_batch)
            #writer.add_scalar('train/acc1_iter', train_acc1, total_batch)
            msg1 = 'Iter: {0:>6},  Train Loss: {1:>5.2}'
            if total_batch%20==0:
                print(msg1.format(total_batch, loss.item()))
            loss_list.append(loss.item())


            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过2000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        dev_loss = evaluate(config, model, dev_iter)#model.eval()
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
            last_improve = total_batch
        else:
            improve = ''
        time_dif = get_time_dif(start_time)
        epoch_loss=np.mean(loss_list)

        msg2 = 'EPOCH: {0:>6},  Train Loss: {1:>5.2},  Val Loss: {2:>5.2},  Time: {3} {4}'
        print(msg2.format(epoch+1,epoch_loss, dev_loss, time_dif, improve))
        writer.add_scalar('train/loss_epoch',epoch_loss, epoch)
        writer.add_scalar('val/loss_epoch', dev_loss, epoch)


        model.train()
        scheduler.step()
        print('epoch: ', epoch+1, 'lr: ', scheduler.get_last_lr())

def evaluate(config, model, data_iter):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for val_num, val_cat in data_iter:
            recon_val, mu, logvar = model(val_num, val_cat)
            loss = loss_function(recon_val, val_num, mu, logvar)
            loss_total += loss

    return loss_total / len(data_iter)



if __name__ == '__main__':

    config = Config()
    writer = SummaryWriter(log_dir=config.log_dir)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("Loading data...")


    train_data=My_Dataset('./data/train.csv',config)
    dev_data = My_Dataset('./data/val.csv',config)


    train_iter=DataLoader(train_data, batch_size=config.batch_size,shuffle=True)   ##训练迭代器
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size,shuffle=True)      ###验证迭代器

    # 训练
    mynet= VAE(config.input_dim, config.hidden_dim, config.latent_dim,
               config.num_categories,config.embedding_dim)
    ## 模型放入到GPU中去
    mynet= mynet.to(config.device)
    print(mynet.parameters)


    train(config, mynet, train_iter, dev_iter,writer)

#tensorboard --logdir=log/1728542085.7592764 --port=6006