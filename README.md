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
2. 特征与编码设计
2.1 FeatureMap 设计
通道数：4 通道，长宽为原图分辨率的 1/2，平衡压缩效率与信息保留能力
存储格式：推理阶段采用FP16 半精度存储，实测图像重建质量损耗极低，相比 FP32 存储空间减半；FP8 精度会出现不可接受的精度损失，不推荐使用
定义代码：
python
运行
self.featuremap = nn.Parameter(torch.empty((1, 4, feature_width, feature_height), dtype=self.MODEL_DTYPE, device=device))
2.2 输入编码设计
模型输入维度为 8，具体构成：Y坐标、X坐标、T时间坐标、FeatureMap 4通道值、e^t-1时间编码
时间编码：单维e^t-1编码实测效果优于 NeRF 经典的 Sin/Cos 位置编码，避免高频编码易过拟合、中间时间步泛化能力差的问题
位置编码：经大量实验验证，引入位置编码的收益无法覆盖其推理开销，最终方案不引入额外位置编码
2.3 Gamma 变换优化
模型学习目标为 RGB 的 Gamma 变换值，而非原始 RGB 数据，大幅提升模型拟合效率与重建精度
训练时对原始数据执行 Gamma 正变换，推理时执行对应反变换
Gamma 超参数通过网格快速搜索寻优，候选范围：[0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,0.6,0.65, 0.7,0.75,0.8, 0.85,0.9, 0.95,1.0,1.1]
核心变换函数：
python
运行
import torch

def gamma_transform_fixed(x: torch.Tensor, mu_param) -> torch.Tensor:
    """RGB Gamma正变换（训练用）"""
    current_gamma = mu_param.mu
    if torch.isclose(current_gamma, torch.tensor(1.0).to(current_gamma.device)):
        return x
    return torch.pow(x.clamp(min=0.0), current_gamma)

def inverse_gamma_transform_fixed(x_gamma: torch.Tensor, mu_param) -> torch.Tensor:
    """RGB Gamma反变换（推理用）"""
    current_gamma = mu_param.mu
    if torch.isclose(current_gamma, torch.tensor(1.0).to(current_gamma.device)):
        return x_gamma
    inv_gamma = 1.0 / current_gamma
    return torch.pow(x_gamma.clamp(min=0.0), inv_gamma)
3. 训练策略
损失函数：采用 L1 Loss，保证重建图像的像素级精度
python
运行
criterion = nn.L1Loss()
增量训练：分层训练模式，训练前层时固定后层梯度，帮助模型收敛到更优解，显著提升重建效果
自动学习率衰减：初始学习率 3e-5，当 Loss 持续无下降时自动按 0.9 倍率降低学习率，避免针对不同图像频繁调参，提升训练通用性
两阶段训练范式
阶段一（基础训练）：同步训练 FeatureMap 与基础 MLP 网络，完成光照时序信息的压缩与基础解码能力学习
阶段二（混合专家优化）：固定 FeatureMap，为每个时间段训练专属 MLP 专家网络，形成混合专家（MoE）架构；26 个时间步对应 26 个专家模型，总参数量仅约 20MB，推理仅增加极少量分支开销，大幅提升时序重建精度
赛题二：UE 渲染管线集成方案
1. 引擎原生代码最小化修改
仅对 2 个核心文件进行参数级修改，保证引擎管线兼容性：
1.1 MapBuildData.cpp 修改
修改光照贴图长宽，适配 FeatureMap 1/2 分辨率特性：
cpp
运行
#if WITH_EDITORONLY_DATA
int SizeX = LightMapTexture->Source.GetSizeX()/2;
int SizeY = LightMapTexture->Source.GetSizeY() / 2;
#else
int SizeX = LightMapTexture->GetSizeX() / 2;
int SizeY = LightMapTexture->GetSizeY() / 2;
#endif
1.2 LightMap.cpp 修改
设置纹理像素格式为 PF_FloatRGBA，支持 FP16 半精度 FeatureMap 的存储与读取：
cpp
运行
Texture->GetPlatformData()->PixelFormat = PF_FloatRGBA;
2. Shader 核心推理实现
核心修改集中于LightmapCommon.ush，基于 HLSL 实现 MLP 网络的全半精度实时推理，核心优化点如下：
矩阵乘推理优化：将 MLP 权重转换为 4x4 矩阵形式，通过矩阵乘加速推理，配套MLPTran脚本实现训练后权重的自动重排与格式适配
全流程半精度计算：数据读取、模型推理全流程采用 HALF 半精度，降低显存开销，优化渲染帧率
循环展开优化：对全连接层循环执行#pragma unroll，消除循环开销，提升 Shader 执行效率
内置 Gamma 反变换：读取预存的 Gamma 超参数，推理完成后自动执行反变换，输出最终 RGB 光照值
完整 Shader 推理代码（可直接粘贴至LightmapCommon.ush光照渲染位置）：
hlsl
half4 feature= half4(Texture2DSample( LightmapResourceCluster.NeuralLightMapTexture,
LightmapResourceCluster.NeuralLightMapSampler,LightmapUV0));
half4 NeuralLightMapParameters[1024] = GetLightmapData(LightmapDataIndex).NeuralLightMapParameters;
half mu=NeuralLightMapParameters[1023].x;
half inv_gamma = 1.0 / mu;
half time_coord = View.TodCurrentTime / 24.0h;

// 构造模型完整输入(总维度=8)
half4 Output0[2];
Output0[0]=half4(LightmapUV0.y*2.0,LightmapUV0.x, time_coord, feature.r);
Output0[1]=half4(feature.g,feature.b, feature.a, exp(time_coord) - 1.0h);

// 第一层隐藏层(16维)
half4 Output1[4] ;
#pragma unroll
for (int i = 0; i < 4; i++)
    Output1[i] = half4(0.0h, 0.0h, 0.0h, 0.0h);
#pragma unroll
for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
    {
        int BaseIndex = (i * 2 + j) * 4;
        half4x4 WeightMatrix=half4x4(
            NeuralLightMapParameters[BaseIndex + 0],
            NeuralLightMapParameters[BaseIndex + 1],
            NeuralLightMapParameters[BaseIndex + 2],
            NeuralLightMapParameters[BaseIndex + 3]
        );
        Output1[i] += mul(WeightMatrix, Output0[j]);
    }
    Output1[i] += NeuralLightMapParameters[176 + i];
    Output1[i] = (Output1[i] / (1.0h + exp(-1.702h * Output1[i])));

// 第二层隐藏层(16维)
half4 Output2[4] ;
#pragma unroll
for (int i = 0; i < 4; i++)
    Output2[i] = half4(0.0h, 0.0h, 0.0h, 0.0h);
#pragma unroll
for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    {
        int BaseIndex = (8+i * 4 + j) * 4;
        half4x4 WeightMatrix=half4x4(
            NeuralLightMapParameters[BaseIndex + 0],
            NeuralLightMapParameters[BaseIndex + 1],
            NeuralLightMapParameters[BaseIndex + 2],
            NeuralLightMapParameters[BaseIndex + 3]
        );
        Output2[i] += mul(WeightMatrix, Output1[j]);
    }
    Output2[i] += NeuralLightMapParameters[180 + i];
    Output2[i] = (Output2[i] / (1.0h + exp(-1.702h * Output2[i])));

// 第三层隐藏层(16维)
#pragma unroll
for (int i = 0; i <4; i++)
    Output1[i] = half4(0.0h, 0.0h, 0.0h, 0.0h);
#pragma unroll
for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    {
        int BaseIndex = (24+i * 4 + j) * 4;
        half4x4 WeightMatrix=half4x4(
            NeuralLightMapParameters[BaseIndex + 0],
            NeuralLightMapParameters[BaseIndex + 1],
            NeuralLightMapParameters[BaseIndex + 2],
            NeuralLightMapParameters[BaseIndex + 3]
        );
        Output1[i] += mul(WeightMatrix, Output2[j]);
    }
    Output1[i] += NeuralLightMapParameters[184 + i];
    Output1[i] = (Output1[i] / (1.0h + exp(-1.702h * Output1[i])));

// 输出层 + Gamma反变换
half3 Output4 = half3(0.0h, 0.0h, 0.0h);
#pragma unroll
for (int j = 0; j < 4; j++)
{
    int BaseIndex = (40 + j) * 4;
    half4x4 WeightMatrix=half4x4(
        NeuralLightMapParameters[BaseIndex + 0],
        NeuralLightMapParameters[BaseIndex + 1],
        NeuralLightMapParameters[BaseIndex + 2],
        NeuralLightMapParameters[BaseIndex + 3]
    );
    Output4.xyz += mul(WeightMatrix, Output1[j]).xyz;
}
Output4 += NeuralLightMapParameters[188].xyz;
Output4 = max(half3(0.0h, 0.0h, 0.0h), Output4);
Output4 = pow(Output4, inv_gamma);

// 输出最终漫反射光照
OutDiffuseLighting = Output4 * Directionality;
性能测试指标
本地测试环境：NVIDIA RTX 4090 显卡，PyTorch 训练框架
表格
指标项	测试结果
PSNR 重建精度得分	68.41
SSIM 结构相似性得分	97.39
LPIPS 感知相似度得分	94.18
压缩比得分	95.69
推理时间得分	98.26
综合总得分	47.12
补充说明：RTX 4060 显卡 UE 渲染管线测试中，16 维隐藏层模型帧率比 64 维隐藏层高 20 倍，因此 Shader 推理场景下隐藏层维度需限制在 16 以内。
工程文件说明
表格
文件名	用途说明
model_*_featuremap_f16.bin	FP16 精度的 FeatureMap 特征文件，推理用
model_*_featuremap_f32.bin	FP32 精度的 FeatureMap 特征文件，训练备份用
model_*_mlp_f32.bin	MLP 网络权重文件，FP32 精度存储
MLPTran.py	权重重排脚本，自动将训练好的 MLP 权重转换为 UE Shader 适配的格式
LightmapCommon.ush	修改后的 UE Shader 文件，包含完整神经网络推理逻辑
部署与复现步骤
1. 离线训练步骤
准备动态光照贴图数据集，按时间步整理好原始 RGB 图像
运行训练代码，执行阶段一基础训练，得到基础 FeatureMap 与 MLP 模型
（可选）执行阶段二混合专家训练，得到分时间段的专家 MLP 模型
运行MLPTran.py脚本，完成权重重排与格式转换
导出 FP16 精度的 FeatureMap 文件与 FP32 精度的 MLP 权重文件
2. UE 引擎集成步骤
按照上文内容，修改 UE 引擎的MapBuildData.cpp与LightMap.cpp文件，重新编译引擎
将转换好的 FeatureMap 与 MLP 权重文件放入 UE 项目对应目录
把完整的 Shader 推理代码替换到LightmapCommon.ush的对应位置
重启 UE 引擎，重新构建光照贴图，即可实现 AI 动态光照实时渲染
注意事项
推理阶段仅需保留 FP16 精度的 FeatureMap 文件，FP32 版本可删除以节省存储空间
不推荐使用 FP8 精度存储 FeatureMap，会出现严重的图像质量损失
高频 NeRF 时间编码易过拟合到 24 个固定时刻，对中间时间步的泛化能力较差，不推荐使用
UE Shader 推理场景下，隐藏层维度需严格控制在 16 以内，否则会出现帧率大幅下降
Gamma 超参数需与训练阶段保持一致，否则会出现颜色偏差
