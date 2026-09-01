"""
CNN + Transformer Hybrid Architecture for ECG Analysis
基于Vision Transformer和医学信号处理的混合模型
Author: Modified from ECGFounder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math


class PositionalEncoding(nn.Module):
    """正弦位置编码 - 标准Transformer位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性投影
        Q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到值
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # 最终线性投影
        output = self.linear_out(context)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差 + LayerNorm
        attn_output, _ = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈 + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x


class Swish(nn.Module):
    """Swish激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class MyConv1dPadSame(nn.Module):
    """自定义1D卷积 - SAME padding"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups
        )
    
    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net


class CNNFeatureExtractor(nn.Module):
    """
    CNN特征提取器 - 用于捕获局部时频特征
    输入: (batch, 12, 5000)
    输出: (batch, seq_len, embedding_dim)
    """
    
    def __init__(self, in_channels=12, base_filters=64, embedding_dim=256):
        super(CNNFeatureExtractor, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 第一层卷积块 - 快速下采样
        self.conv1 = MyConv1dPadSame(in_channels, base_filters, kernel_size=16, stride=2)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.activation1 = Swish()
        self.dropout1 = nn.Dropout(p=0.3)
        
        # 第二层卷积块
        self.conv2 = MyConv1dPadSame(base_filters, base_filters * 2, kernel_size=16, stride=2)
        self.bn2 = nn.BatchNorm1d(base_filters * 2)
        self.activation2 = Swish()
        self.dropout2 = nn.Dropout(p=0.3)
        
        # 第三层卷积块
        self.conv3 = MyConv1dPadSame(base_filters * 2, base_filters * 4, kernel_size=16, stride=2)
        self.bn3 = nn.BatchNorm1d(base_filters * 4)
        self.activation3 = Swish()
        self.dropout3 = nn.Dropout(p=0.3)
        
        # 第四层卷积块
        self.conv4 = MyConv1dPadSame(base_filters * 4, base_filters * 8, kernel_size=16, stride=2)
        self.bn4 = nn.BatchNorm1d(base_filters * 8)
        self.activation4 = Swish()
        self.dropout4 = nn.Dropout(p=0.3)
        
        # 投影到embedding维度
        self.projection = nn.Linear(base_filters * 8, embedding_dim)
        
        # Squeeze-and-Excitation 通道注意力
        r = 2
        self.se_fc1 = nn.Linear(base_filters * 8, base_filters * 8 // r)
        self.se_fc2 = nn.Linear(base_filters * 8 // r, base_filters * 8)
        self.se_activation = Swish()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 12, 5000)
        Returns:
            (batch_size, seq_len, embedding_dim)
        """
        # 第一层卷积
        out = self.conv1(x)  # (batch, 64, 2500)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.dropout1(out)
        
        # 第二层卷积
        out = self.conv2(out)  # (batch, 128, 1250)
        out = self.bn2(out)
        out = self.activation2(out)
        out = self.dropout2(out)
        
        # 第三层卷积
        out = self.conv3(out)  # (batch, 256, 625)
        out = self.bn3(out)
        out = self.activation3(out)
        out = self.dropout3(out)
        
        # 第四层卷积
        out = self.conv4(out)  # (batch, 512, 312)
        out = self.bn4(out)
        out = self.activation4(out)
        out = self.dropout4(out)
        
        # Squeeze-and-Excitation注意力
        se = out.mean(-1)  # (batch, 512)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = torch.sigmoid(se)  # (batch, 512)
        out = torch.einsum('abc,ab->abc', out, se)
        
        # 转置为 (batch, seq_len, channels)
        out = out.transpose(1, 2)  # (batch, 312, 512)
        
        # 投影到embedding维度
        out = self.projection(out)  # (batch, 312, 256)
        
        return out


class CNNTransformerECG(nn.Module):
    """
    CNN + Transformer混合模型用于ECG分析
    
    架构:
    1. CNN特征提取器 - 捕获局部时频特征，快速下采样
    2. Positional Encoding - 添加位置信息
    3. Transformer编码器 - 建模全局长程依赖关系
    4. 多任务头 - 用于ECG参数预测
    """
    
    def __init__(self,
                 in_channels=12,
                 cnn_base_filters=64,
                 d_model=256,
                 num_heads=8,
                 num_layers=4,
                 d_ff=1024,
                 n_classes=150,
                 dropout=0.1,
                 return_features=False):
        super(CNNTransformerECG, self).__init__()
        
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.return_features = return_features
        
        # CNN特征提取器
        self.cnn_extractor = CNNFeatureExtractor(
            in_channels=in_channels,
            base_filters=cnn_base_filters,
            embedding_dim=d_model
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len=5000)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层标准化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 分类头
        self.dense = nn.Linear(d_model, n_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 12, 5000) - 12导联ECG信号
        Returns:
            output: (batch_size, n_classes)
            features: (batch_size, d_model) [可选]
        """
        # CNN特征提取
        cnn_features = self.cnn_extractor(x)  # (batch, seq_len, d_model)
        
        # 添加位置编码
        x_embed = self.positional_encoding(cnn_features)
        x_embed = self.dropout(x_embed)
        
        # Transformer编码器
        for layer in self.transformer_layers:
            x_embed = layer(x_embed)
        
        # 最终层标准化
        x_embed = self.final_norm(x_embed)
        
        # 全局平均池化
        deep_features = x_embed.mean(dim=1)  # (batch, d_model)
        
        # 分类输出
        output = self.dense(deep_features)  # (batch, n_classes)
        
        if self.return_features:
            return output, deep_features
        else:
            return output


class MultiTaskCNNTransformerECG(nn.Module):
    """
    多任务CNN+Transformer模型 - 用于ECG参数回归
    """
    
    def __init__(self,
                 in_channels=12,
                 cnn_base_filters=64,
                 d_model=256,
                 num_heads=8,
                 num_layers=4,
                 d_ff=1024,
                 dropout=0.1,
                 n_parameters=9):
        super(MultiTaskCNNTransformerECG, self).__init__()
        
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_parameters = n_parameters
        
        # CNN特征提取器
        self.cnn_extractor = CNNFeatureExtractor(
            in_channels=in_channels,
            base_filters=cnn_base_filters,
            embedding_dim=d_model
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len=5000)
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层标准化
        self.final_norm = nn.LayerNorm(d_model)
        
        # ECG参数列表
        self.parameter_names = [
            'HR_bpm', 'RR_median_ms', 'PR_ms', 'QRS_ms',
            'QT_ms', 'QTc_ms', 'AXIS_degree', 'SV1', 'RV5'
        ]
        
        # 多任务预测头
        self.parameter_heads = nn.ModuleDict({
            param: nn.Sequential(
                nn.Linear(d_model, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            )
            for param in self.parameter_names[:n_parameters]
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 12, 5000)
        Returns:
            predictions: dict of {parameter: tensor}
        """
        # CNN特征提取
        cnn_features = self.cnn_extractor(x)  # (batch, seq_len, d_model)
        
        # 添加位置编码
        x_embed = self.positional_encoding(cnn_features)
        x_embed = self.dropout(x_embed)
        
        # Transformer编码器
        for layer in self.transformer_layers:
            x_embed = layer(x_embed)
        
        # 最终层标准化
        x_embed = self.final_norm(x_embed)
        
        # 全局平均池化提取特征
        global_features = x_embed.mean(dim=1)  # (batch, d_model)
        
        # 多任务预测
        predictions = {}
        for param in self.parameter_names[:self.n_parameters]:
            pred = self.parameter_heads[param](global_features)  # (batch, 1)
            predictions[param] = pred.squeeze(-1)
        
        return predictions, global_features


# ============ 注意力可视化工具 ============

class AttentionVisualizer:
    """可视化Transformer注意力权重"""
    
    @staticmethod
    def get_attention_weights(model, x):
        """提取注意力权重用于可视化"""
        cnn_features = model.cnn_extractor(x)
        x_embed = model.positional_encoding(cnn_features)
        
        attention_weights = []
        for layer in model.transformer_layers:
            attn_output, attn_weight = layer.attention(x_embed, x_embed, x_embed)
            attention_weights.append(attn_weight.detach())
        
        return attention_weights


# ============ 测试代码 ============

if __name__ == "__main__":
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = CNNTransformerECG(
        in_channels=12,
        cnn_base_filters=64,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        n_classes=150,
        dropout=0.1,
        return_features=False
    ).to(device)
    
    # 测试输入
    x = torch.randn(4, 12, 5000).to(device)  # batch_size=4
    
    # 前向传播
    output = model(x)
    print(f"✅ 分类模型输出形状: {output.shape}")  # (4, 150)
    
    # 多任务模型
    multitask_model = MultiTaskCNNTransformerECG(
        in_channels=12,
        cnn_base_filters=64,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        dropout=0.1,
        n_parameters=9
    ).to(device)
    
    predictions, features = multitask_model(x)
    print(f"\n✅ 多任务模型输出:")
    for param, pred in predictions.items():
        print(f"  {param}: {pred.shape}")  # (4,)
    print(f"  特征: {features.shape}")  # (4, 256)
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
