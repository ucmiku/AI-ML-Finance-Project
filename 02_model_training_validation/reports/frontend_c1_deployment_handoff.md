# 前端交付说明：只部署 C1 Prediction Agent

## 1. 部署原则

前端正式产品只接入一个 Prediction Agent：

```text
C1 = XGBoost Regression Head + XGBoost 5/20 Classification Head
```

前端不应让用户在多个模型之间选择。B1-B5、C1-C4、M7-M12等结果可以放在历史回测或研究页面中作为静态展示，但不应作为多个实时预测接口暴露。

## 2. 页面展示建议

单小时预测页面建议展示：

- 预测连续 spread
- INC / No Trade / DEC 概率
- 当前建议动作
- 置信度
- 极端天气风险
- SHAP主要驱动因素
- 模型版本
- 预测时间

## 3. API输出建议

建议API结构：

```json
{
  "delivery_hour_utc": "2025-07-15T18:00:00Z",
  "predicted_spread": 18.4,
  "p_negative": 0.08,
  "p_neutral": 0.21,
  "p_positive": 0.71,
  "predicted_class": "positive",
  "recommended_action": "DEC",
  "confidence": 0.71,
  "extreme_weather_flag": true,
  "top_drivers": [
    "net_load_ramp_1h_mw",
    "cloud_cover_dfw_mean_pct",
    "spread_asof_lag168"
  ],
  "model_name": "C1_XGBoost_Prediction_Agent",
  "model_version": "v3"
}
```

## 4. 字段映射

| API字段 | 后端字段 | 说明 |
|---|---|---|
| `delivery_hour_utc` | `delivery_hour_utc` | 预测交付小时 |
| `predicted_spread` | B2A输出 | 连续spread预测 |
| `p_negative` | `p_c1 + p_c2` | INC概率 |
| `p_neutral` | `p_c3` | No-trade概率 |
| `p_positive` | `p_c4 + p_c5` | DEC概率 |
| `predicted_class` | B2B输出 | 五分类结果，可映射为negative/neutral/positive |
| `recommended_action` | `signal_base`映射 | INC / NO_TRADE / DEC |
| `confidence` | 最大类别概率 | 当前预测强度 |
| `extreme_weather_flag` | `fixed_extreme_weather_flag` | 极端天气风险展示 |
| `top_drivers` | SHAP top features | 模型解释 |
| `model_name` | 固定为C1 | 正式部署模型名 |
| `model_version` | `v3` | 当前版本 |

## 5. 推荐UI文案

动作展示：

- `INC`: 模型认为RT相对DA更可能偏低，适合研究INC方向。
- `DEC`: 模型认为RT相对DA更可能偏高，适合研究DEC方向。
- `NO_TRADE`: 当前概率不足，不建议基准策略交易。

风险展示：

- 当 `extreme_weather_flag=true` 时，显示“极端天气风险：高”。
- 当 `confidence < 0.60` 时，显示“置信度不足：No Trade”。
- 不要把SHAP解释写成因果结论，只写“模型主要参考因素”。

## 6. 历史回测页

历史回测页面可以展示：

- B1-B5单模型统计对比。
- C1-C4完整系统对比。
- Z0-Z3消融结果。
- 2026 weekly walk-forward结果。
- SHAP全局解释。

这些页面只做静态分析展示，不需要部署多个实时模型。

## 7. 当前正式版本

```text
model_name: C1_XGBoost_Prediction_Agent
model_version: v3
regression_head: B2A_XGBoost_regression
classification_head: B2B_XGBoost_5_20_classifier
baseline_threshold: 0.60
```

