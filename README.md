# 高性能AI渲染大赛 - 光照贴图神经网络重建与UE渲染管线集成方案

## 项目概述
本项目为高性能AI渲染大赛参赛方案，针对**光照贴图神经网络时序重建**与**UE渲染管线工程化集成**两个赛题，设计了一套兼顾重建精度、压缩效率与实时推理帧率的轻量级AI渲染方案。

核心基于极小参数量的MLP网络实现光照信息压缩与解码，通过针对性的训练策略、数据编码优化与UE Shader工程适配，最终实现了动态光照贴图的高质量实时重建，满足离线训练精度与在线渲染性能的双重要求。

---

## 目录
1. [赛题核心目标](#赛题核心目标)
2. [赛题一：光照贴图神经网络重建方案](#赛题一光照贴图神经网络重建方案)
3. [赛题二：UE渲染管线集成方案](#赛题二ue渲染管线集成方案)
4. [性能测试指标](#性能测试指标)
5. [工程文件说明](#工程文件说明)
6. [部署与复现步骤](#部署与复现步骤)
7. [注意事项](#注意事项)

---

## 赛题核心目标
### 赛题一：光照贴图神经网络重建
基于神经网络实现动态光照贴图的压缩与时序重建，核心目标为：
1. 设计可高效存储光照时序信息的FeatureMap，实现高压缩比
2. 训练神经网络实现从压缩特征到任意时间步光照图像的高精度重建
3. 平衡重建质量、压缩率与推理速度，实现三者综合性能最优

### 赛题二：UE渲染管线集成
将赛题一的AI重建算法集成到Unreal Engine渲染管线中，核心目标为：
1. 最小化引擎原生代码修改，保证管线兼容性
2. 基于UE Shader实现神经网络实时推理，保障高渲染帧率
3. 验证算法在实际渲染管线中的落地效果，实现动态光照的实时渲染

---

## 赛题一：光照贴图神经网络重建方案
### 1. 核心网络架构
为兼顾推理速度与UE Shader渲染帧率，设计极简MLP堆栈架构，整体维度为 `8->16->16->16->3`，激活函数采用GeLU，避免过拟合的同时保证非线性拟合能力。

核心网络定义：
```python
import torch
import torch.nn as nn

class MLPStack(nn.Module):
    """
    光照重建MLP堆栈
    输入维度: 8
    隐藏层维度: 16×3
    输出维度: 3 (RGB Gamma变换值)
    """
    def __init__(self, input_combined_dim=8, hidden_dim1=16, hidden_dim2=16, output_dim=3):
        super().__init__()
        self.activation = nn.GELU()
        # 3层隐藏层+输出头
        self.b1 = nn.Linear(input_combined_dim, hidden_dim1)
        self.b2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.b3 = nn.Linear(hidden_dim2, hidden_dim2)
        self.head = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.activation(self.b1(x))
        x = self.activation(self.b2(x))
        x = self.activation(self.b3(x))
        return self.head(x)
