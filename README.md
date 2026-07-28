# AI-ML-Finance-Project

NUS summer camp project.

项目标题：基于极端天气特征驱动的 ERCOT 实时电价机器学习套利策略与回测平台

## Project Workflow

本仓库按项目流程拆分为四个主要工作区：

```text
01_data_collection_cleaning/
  raw/          # 原始数据：ERCOT 实时电价、负荷、天气、极端天气事件等
  interim/      # 中间数据：临时合并、初步清洗、特征拼接结果
  processed/    # 建模数据：可直接用于训练、验证、回测的数据集
  scripts/      # 数据下载、清洗、特征工程脚本
  notebooks/    # 数据探索与清洗验证 notebook

02_model_training_validation/
  notebooks/    # 模型实验、特征选择、误差分析 notebook
  src/          # 可复用训练、验证、预测代码
  models/       # 训练好的模型文件
  metrics/      # 验证指标、模型对比结果
  predictions/  # 验证集/测试集/未来窗口预测结果

03_qc_backtesting/
  algorithms/   # QuantConnect / LEAN 策略代码
  config/       # 回测配置、参数、数据映射
  results/      # 回测输出、交易记录、权益曲线
  reports/      # 回测分析报告

04_frontend_dashboard/
  app/          # Dashboard 前端应用代码
  assets/       # 图片、图标、静态资源
  data/         # Dashboard 展示用的轻量结果数据
  docs/         # 前端说明文档

docs/           # 项目计划、方法说明、报告材料
outputs/        # 最终汇总输出：图表、演示材料、可交付文件
```

## 各模块详细文档

各模块的具体文档请参阅各自的 README：
- `01_data_collection_cleaning/` — 数据采集与清洗
- `02_model_training_validation/` — 模型训练与验证（含 C1 XGBoost 交付）
- `03_qc_backtesting/` — [量化回测系统](03_qc_backtesting/README.md)（成员 C 负责）
- `04_frontend_dashboard/` — 前端 Dashboard

