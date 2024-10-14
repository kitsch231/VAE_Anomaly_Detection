import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_categories, embedding_dim):
        super(VAE, self).__init__()

        # 类别特征嵌入层
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.num_categories = num_categories

        # 编码器部分
        self.fc1 = nn.Linear(input_dim + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值向量
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 方差向量

        # 解码器部分
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)  # 只输出数值型特征

        # Attention layer
        self.attention = AttentionLayer(hidden_dim)

        # 正则化层
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def encode(self, x, category):
        # 将类别转换为嵌入向量
        category_embedding = self.embedding(category)

        # 拼接数值特征和嵌入特征
        h1 = torch.cat((x, category_embedding), dim=1)
        h1 = F.leaky_relu(self.fc1(h1))
        h1 = self.bn1(F.leaky_relu(self.fc2(h1)))
        h1 = self.dropout(h1)

        # 通过自注意力层
        h1_attn = self.attention(h1.unsqueeze(0))  # 增加一个维度以适应 MultiheadAttention
        h1 = h1 + h1_attn.squeeze(0)  # 残差连接

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)  # 防止 logvar 过大或过小
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        h3 = self.bn2(F.leaky_relu(self.fc4(h3)))
        h3 = self.dropout(h3)

        # 通过自注意力层
        h3_attn = self.attention(h3.unsqueeze(0))
        h3 = h3 + h3_attn.squeeze(0)  # 残差连接

        # 输出数值特征
        recon_x = self.fc5(h3)

        return recon_x

    def forward(self, x, category):
        mu, logvar = self.encode(x, category)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        return recon_x, mu, logvar
