高性能AI渲染大赛 - 光照贴图神经网络重建与UE渲染管线集成方案
项目概述
本项目为高性能AI渲染大赛参赛方案，针对光照贴图神经网络时序重建与UE渲染管线工程化集成两个赛题，设计了一套兼顾重建精度、压缩效率与实时推理帧率的轻量级AI渲染方案。
核心基于极小参数量的MLP网络实现光照信息压缩与解码，通过针对性的训练策略、数据编码优化与UE Shader工程适配，最终实现了动态光照贴图的高质量实时重建，满足离线训练精度与在线渲染性能的双重要求。
---
目录
1.赛题核心目标
赛题一：光照贴图神经网络重建方案
赛题二：UE渲染管线集成方案
性能测试指标
工程文件说明
部署与复现步骤
注意事项
---
赛题核心目标
赛题一：光照贴图神经网络重建
基于神经网络实现动态光照贴图的压缩与时序重建，核心目标为：
设计可高效存储光照时序信息的FeatureMap，实现高压缩比
训练神经网络实现从压缩特征到任意时间步光照图像的高精度重建
平衡重建质量、压缩率与推理速度，实现三者综合性能最优
赛题二：UE渲染管线集成
将赛题一的AI重建算法集成到Unreal Engine渲染管线中，核心目标为：
最小化引擎原生代码修改，保证管线兼容性
基于UE Shader实现神经网络实时推理，保障高渲染帧率
验证算法在实际渲染管线中的落地效果，实现动态光照的实时渲染
---
赛题一：光照贴图神经网络重建方案
1. 核心网络架构
为兼顾推理速度与重建精度，采用轻量级MLP网络架构，整体结构为8->16->16->16->3，激活函数选用GeLU，有效平衡模型表达能力与计算开销，适配后续UE Shader实时推理需求。
```python
class MLPStack(nn.Module):
    """MLP 堆栈，实现光照特征解码"""
    def __init__(self, input_combined_dim, output_dim, hidden_dim1, hidden_dim2):
        # 定义所有Block 和Head，确保模型轻量化
        super().__init__()
        self.activation = nn.GELU()
        self.b1 = nn.Linear(input_combined_dim, hidden_dim1)
        self.b2 = nn.Linear(hidden_dim1, hidden_dim2)
```
2. FeatureMap设计
设计4通道FeatureMap，长宽设为原图分辨率的一半，在保证压缩效率的同时，最大限度保留光照贴图的核心时序信息，避免特征压缩导致的重建失真。FeatureMap定义如下：
```python
self.featuremap = nn.Parameter(torch.empty((1, 4, feature_width, feature_height), dtype=self.MODEL_DTYPE, device=device))
```
3. 输入输出编码设计
3.1 输入参数设计
模型输入维度共计8维，具体包括：图像Y坐标、X坐标、时间坐标T、FeatureMap的4个通道值（r、g、b、a），以及1位时间编码e^t-1。
关键优化说明：
时间编码：仅使用1位e^t-1编码时，实测效果优于SIN、COS类NeRF时间编码；若硬件资源充足，采用长段NeRF时间编码可进一步提升性能，但高频NeRF编码易过拟合到24个训练时刻，导致中间时刻推理精度下降。
位置编码：经大量实验验证，位置编码对重建效果提升不明显，且会增加计算开销，因此未引入位置编码。
输入汇总：(Y, X, T, Featuremap.r, Featuremap.g, Featuremap.b, Featuremap.a, E^T-1)
3.2 输出参数设计
模型输出为光照贴图的RGB值GAMMA变换结果，而非原始RGB数据，通过GAMMA变换可有效提升模型训练稳定性与重建精度。
4. 训练策略设计
4.1 训练方法
采用增量训练+自动学习率下降策略，兼顾训练效果与效率：
增量训练：先训练第一层网络，训练过程中固定第二层梯度，实测可使模型收敛到更优解，提升重建精度。
自动学习率下降：初始学习率设为3E-5，当损失函数（LOSS）持续迭代无下降时，自动按0.9的比例降低学习率，避免因不同图像差异频繁手动调整学习率，提升训练通用性。
4.2 损失函数
选用L1Loss作为损失函数，有效降低重建图像与原始图像的像素级误差，适合光照贴图这类连续值图像的重建任务，定义如下：
```python
criterion = nn.L1Loss()
```
4.3 GAMMA变换实现
训练时对原始RGB数据进行GAMMA变换，推理时执行反变换，确保输出结果与原始光照贴图一致。GAMMA超参数通过快速搜索确定，候选值范围为[0.05, 1.1]，通过少量迭代训练筛选最优值。
```python
def gamma_transform_fixed(x: torch.Tensor, mu_param: FixedMu) -> torch.Tensor:
    current_gamma = mu_param.mu
    if torch.isclose(current_gamma, torch.tensor(1.0).to(current_gamma.device)):
        return x
    else:
        return torch.pow(x.clamp(min=0.0), current_gamma)

# 推理时反变换
def inverse_gamma_transform_fixed(x_gamma: torch.Tensor, mu_param: FixedMu) -> torch.Tensor:
    current_gamma = mu_param.mu
    if torch.isclose(current_gamma, torch.tensor(1.0).to(current_gamma.device)):
        return x_gamma
    else:
        inv_gamma = 1.0 / current_gamma
        return torch.pow(x_gamma.clamp(min=0.0), inv_gamma)
```
5. 模型存储优化
为平衡存储开销与重建精度，采用差异化存储策略：
MLP权重采用FP32存储，确保推理精度；
FeatureMap同时存储FP32和FP16格式，推理时可删除FP32格式，仅保留FP16，实测FP16推理对重建质量损耗极低，而FP8会导致损耗急剧上升。
```python
# FeatureMap FP32存储
output_featuremap_f32_filename = f"./Parameters/model_{lightmap['level']}_{id}_featuremap_f32.bin"
featuremap_array_f32.tofile(output_featuremap_f32_filename)

# FeatureMap FP16存储
output_featuremap_f16_filename = f"./Parameters/model_{lightmap['level']}_{id}_featuremap_f16.bin"
featuremap_array_f32.astype(np.float16).tofile(output_featuremap_f16_filename)

# MLP权重FP32存储
output_mlp_filename = f"./Parameters/model_{lightmap['level']}_{id}_mlp_f32.bin"
mlp_final_array_f32.tofile(output_mlp_filename)
```
6. 进阶训练阶段（阶段二）
在基础训练（阶段一）完成后，引入混合专家架构，进一步提升时序重建精度：
阶段一核心：训练出优质FeatureMap，存储光照时序信息，并得到一个基础MLP网络用于通用推理；
阶段二核心：固定阶段一训练得到的FeatureMap，针对不同时间段训练多个MLP（每个MLP作为对应时间段的“专家”），通过时间信息将数据分配给对应MLP解码；
优势：模型参数量可控（26个专家总参数量约20MB），推理速度仅增加一个IF ELSE判断开销，大幅提升不同时间段的重建精度。
阶段二采用与阶段一相同的8->16->16->16->3模型架构，仅训练MLP权重和偏置，固定FeatureMap参数。
---
赛题二：UE渲染管线集成方案
核心原则：最小化UE引擎原生代码修改，基于官方管线集成方法，将赛题一的AI重建算法集成到UE渲染管线中，重点优化Shader推理代码，保障实时渲染帧率。
1. 引擎代码修改
仅对引擎核心文件进行少量修改，适配FeatureMap特性与存储要求：
1.1 MapBuildData.cpp修改
将LightMapTexture长宽减半，适配FeatureMap（原图分辨率一半）的特性，确保FeatureMap正确读取：
```cpp
#if WITH_EDITORONLY_DATA
int SizeX = LightMapTexture->Source.GetSizeX()/2;
int SizeY = LightMapTexture->Source.GetSizeY() / 2;
#else
int SizeX = LightMapTexture->GetSizeX() / 2;
int SizeY = LightMapTexture->GetSizeY() / 2;
#endif
```
1.2 LightMap.cpp修改
设置FeatureMap以FP16格式存储和读取，匹配赛题一的存储优化策略：
```cpp
Texture->GetPlatformData()->PixelFormat = PF_FloatRGBA;
```
2. Shader推理代码修改（核心）
主要修改LightmapCommon.ush文件中的Shader代码，适配MLP推理、GAMMA反变换与半精度优化，基本沿用官方推理框架，仅做针对性调整：
2.1 MLP层数与推理优化
沿用赛题一的8->16->16->16->3 MLP架构，采用矩阵乘优化推理速度，需先通过MLPTrans脚本对MLP权重和偏置进行重排，确保Shader正确读取。
```hlsl
half4x4 WeightMatrix=half4x4(
    WeightMatrix[0] = NeuralLightMapParameters[BaseIndex + 0],
    WeightMatrix[1] = NeuralLightMapParameters[BaseIndex + 1],
    WeightMatrix[2] = NeuralLightMapParameters[BaseIndex + 2],
    WeightMatrix[3] = NeuralLightMapParameters[BaseIndex + 3]);
Output1[i] += mul(WeightMatrix, Output0[j]);
```
2.2 GAMMA反变换实现
读取MLP文件中存储的GAMMA超参数（mu），在推理完成后执行反变换，确保输出光照贴图与原始效果一致：
```hlsl
half mu=NeuralLightMapParameters[1023].x;
half inv_gamma = 1.0 / mu;
Output4 = max(half3(0.0h, 0.0h, 0.0h), Output4);
Output4 = pow(Output4, inv_gamma);
```
2.3 半精度推理优化
将数据读取和模型推理均改为HALF半精度（half4），尝试进一步提升帧率，实测引擎可能已自动将FLOAT优化为HALF，因此帧率提升不明显，但未影响重建精度。
3. 完整Shader推理代码
以下代码可直接复制粘贴到LightmapCommon.ush的光照渲染位置，实现算法复现：
```hlsl
half4 feature= half4(Texture2DSample( LightmapResourceCluster.NeuralLightMapTexture,
LightmapResourceCluster.NeuralLightMapSampler,LightmapUV0));
half4 NeuralLightMapParameters[1024] =
GetLightmapData(LightmapDataIndex).NeuralLightMapParameters;
half mu=NeuralLightMapParameters[1023].x;
half inv_gamma = 1.0 / mu;
half time_coord = View.TodCurrentTime / 24.0h;
// 3. 构造模型完整输入(总维度=3(input)+4(feature)+1(时间编码)=8)
half4 Output0[2];
Output0[0]=half4(LightmapUV0.y*2.0,LightmapUV0.x, time_coord, feature.r);
Output0[1]=half4(feature.g,feature.b, feature.a, exp(time_coord) - 1.0h);

half4 Output1[4] ; // 第一层隐藏层(16维)
#pragma unroll // 展开外层循环,消除循环开销
for (int i = 0; i < 4; i++)
    Output1[i] = half4(0.0h, 0.0h, 0.0h, 0.0h);
#pragma unroll // 展开外层循环,消除循环开销
for (int j = 0; j < 2; j++)
    int BaseIndex = (i * 2 + j) * 4;
    half4x4 WeightMatrix=half4x4(
        WeightMatrix[0] = NeuralLightMapParameters[BaseIndex + 0],
        WeightMatrix[1] = NeuralLightMapParameters[BaseIndex + 1],
        WeightMatrix[2] = NeuralLightMapParameters[BaseIndex + 2],
        WeightMatrix[3] = NeuralLightMapParameters[BaseIndex + 3]);
    Output1[i] += mul(WeightMatrix, Output0[j]);
Output1[i] += NeuralLightMapParameters[176 + i];
Output1[i] = (Output1[i] / (1.0h + exp(-1.702h * Output1[i])));

half4 Output2[4] ;
#pragma unroll // 展开外层循环,消除循环开销
for (int i = 0; i < 4; i++)
    Output2[i] = half4(0.0h, 0.0h, 0.0h, 0.0h);
#pragma unroll // 展开外层循环,消除循环开销
for (int j = 0; j < 4; j++)
    int BaseIndex = (8+i * 4 + j) * 4;
    half4x4 WeightMatrix=half4x4(
        WeightMatrix[0] = NeuralLightMapParameters[BaseIndex + 0],
        WeightMatrix[1] = NeuralLightMapParameters[BaseIndex + 1],
        WeightMatrix[2] = NeuralLightMapParameters[BaseIndex + 2],
        WeightMatrix[3] = NeuralLightMapParameters[BaseIndex + 3]);
    Output2[i] += mul(WeightMatrix, Output1[j]);
Output2[i] += NeuralLightMapParameters[180 + i];
Output2[i] = (Output2[i] / (1.0h + exp(-1.702h * Output2[i])));

#pragma unroll // 展开外层循环,消除循环开销
for (int i = 0; i <4; i++)
    Output1[i] = half4(0.0h, 0.0h, 0.0h, 0.0h);
#pragma unroll // 展开外层循环,消除循环开销
for (int j = 0; j < 4; j++)
    int BaseIndex = (24+i * 4 + j) * 4;
    half4x4 WeightMatrix=half4x4(
        WeightMatrix[0] = NeuralLightMapParameters[BaseIndex + 0],
        WeightMatrix[1] = NeuralLightMapParameters[BaseIndex + 1],
        WeightMatrix[2] = NeuralLightMapParameters[BaseIndex + 2],
        WeightMatrix[3] = NeuralLightMapParameters[BaseIndex + 3]);
    Output1[i] += mul(WeightMatrix, Output2[j]);
Output1[i] += NeuralLightMapParameters[184 + i];
Output1[i] = (Output1[i] / (1.0h + exp(-1.702h * Output1[i])));

half3 Output4 = half3(0.0h, 0.0h, 0.0h);
#pragma unroll
for (int j = 0; j < 4; j++)
    int BaseIndex = (40 + j) * 4;
    half4x4 WeightMatrix=half4x4(
        WeightMatrix[0] = NeuralLightMapParameters[BaseIndex + 0],
        WeightMatrix[1] = NeuralLightMapParameters[BaseIndex + 1],
        WeightMatrix[2] = NeuralLightMapParameters[BaseIndex + 2],
        WeightMatrix[3] = NeuralLightMapParameters[BaseIndex + 3]);
    Output4.xyz += mul(WeightMatrix, Output1[j]).xyz;
Output4 += NeuralLightMapParameters[188].xyz;
Output4 = max(half3(0.0h, 0.0h, 0.0h), Output4);
Output4 = pow(Output4, inv_gamma);
OutDiffuseLighting = Output4 * Directionality;
```
---
性能测试指标
模型综合性能呈凸型函数分布，通过调整超参数可实现重建质量、压缩效率与推理速度的最优平衡，核心测试指标如下（本地4090 GPU环境）：
```json
{
    "PSNR Score": 68.40884319641688,
    "SSIM Score": 97.38861591783102,
    "LPIPS Score": 94.17818999149053,
    "Compression Ratio Score": 95.69013578469222,
    "Inference Time Score": 98.2591621875763,
    "综合得分": 47.115683940361814
}
```
关键说明：
4090 GPU+PyTorch环境下，16维隐藏层与64维隐藏层的推理速度差异不明显；
UE渲染管线（4060 GPU，Shader推理）中，16维隐藏层与64维隐藏层的帧率相差20倍，因此UE Shader推理时，隐藏层维度需限制在16以下（除非通过CUDA和TENSOR CORE优化）。
---
工程文件说明
1. 模型参数文件
存储路径：./Parameters/
model_{level}_{id}_featuremap_f32.bin：FeatureMap FP32格式文件
model_{level}_{id}_featuremap_f16.bin：FeatureMap FP16格式文件（推理首选）
model_{level}_{id}_mlp_f32.bin：MLP权重FP32格式文件
2. 辅助脚本
MLPTrans.py：MLP权重重排脚本，训练完成后运行，将权重和偏置重排为UE Shader可读取的格式。
3. Shader文件
LightmapCommon.ush：修改后的UE光照渲染Shader文件，包含完整MLP推理代码，替换引擎对应文件即可使用。
---
部署与复现步骤
1. 环境准备
离线训练环境：PyTorch、CUDA（建议11.7+）、GPU（建议4090及以上，用于快速训练）；
UE渲染环境：Unreal Engine（版本兼容即可）、4060及以上GPU（用于Shader推理测试）。
2. 模型训练与权重准备
运行训练代码，采用增量训练+自动学习率下降策略，完成阶段一和阶段二训练；
训练完成后，运行MLPTrans.py脚本，对MLP权重进行重排；
将生成的FeatureMap（FP16）和MLP权重文件放入./Parameters/目录。
3. UE管线集成
修改UE引擎文件：MapBuildData.cpp、LightMap.cpp，按上述代码调整参数；
替换LightmapCommon.ush文件，粘贴完整Shader推理代码；
启动UE项目，加载光照贴图资源，验证实时渲染效果。
---
注意事项
模型超参数调整：GAMMA超参数需通过快速搜索确定，避免直接使用默认值导致重建精度下降；
权重重排：MLP训练完成后必须运行MLPTrans脚本，否则UE Shader无法正确读取权重，导致推理失败；
精度选择：FeatureMap推理时优先使用FP16格式，避免使用FP8格式，防止重建质量急剧下降；
UE帧率优化：UE Shader推理时，MLP隐藏层维度需限制在16以下，否则帧率会大幅降低；
环境兼容：确保UE引擎版本与Shader代码兼容，避免因版本差异导致的编译错误。
> （注：文档部分内容可能由 AI 生成）
