# 安全与复现说明

## 密钥管理

公开仓库不能提交真实 API key。请使用 `.env.example` 创建本地 `.env`：

```powershell
Copy-Item .env.example .env
```

然后只在 `.env` 中填写真实密钥。`.env` 已经加入 `.gitignore`。

## 大文件管理

大体积 JSONL、模型权重、训练检查点和完整原始日志不建议直接提交到 git。建议使用：

- GitHub Releases
- Hugging Face Datasets
- 对象存储
- 学校或实验室内部归档

仓库中保留处理后 CSV、论文图和小规模示例数据，用于复现分析流程。

## 复现实验

最小代码测试：

```powershell
pip install -e .[dev]
pytest
```

如果需要运行真实模型实验，需要额外配置模型服务、评估器和数据路径。公开版 `src/ember` 不绑定任何私有供应商。
