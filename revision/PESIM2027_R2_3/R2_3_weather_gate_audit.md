# PES IM 2027 — Reviewer 2.3 Weather Gate 方法核查

**审计结论：有条件通过（PASS with provenance limitation）**

**稳定 run ID：** `R2_3_WG_C64C6FDF13E0FE54`

**统一口径：** 最终投稿 PDF + `section_iv_canonical_v1` + 冻结预测交接表；未重训、未改交易规则、未覆盖输入。

## 中文总结

现存代码能够直接证明最终 Weather Gate 的交易逻辑：先按 `τ=0.60` 产生 B2B 概率方向信号，再要求连续价差预测的符号一致，最后要求 `fixed_extreme_weather_flag=1`；三者同时满足才形成 S5 交易，固定仓位为 1 MWh。S5 没有使用 `|predicted_spread|>=5` 的幅度过滤，没有 `p_outer` 过滤，也没有置信度仓位缩放。天气标记本身是冰冻、高温、强阵风三类小时标记的 OR；降水不参与 Gate。

六个地点和三个物理阈值在最终投稿 PDF 与本地设计说明中一致记载为 Dallas、Fort Worth、Denton、McKinney、Arlington、Wichita Falls，以及温度 `<=0 °C`、温度 `>=35 °C`、阵风 `>=15 m/s`。冻结数据逐小时完全满足这些阈值与 DFW 极值/Wichita Falls 值的等价关系。但是，本地现存源码从已经生成的 `*_city_count` 开始，未保留逐城市原始预报字段，也未找到“逐站比较后生成四个 city-count 字段”的上游脚本；设计说明所指的 `01_data_collection_cleaning` 数据库和处理目录也不在当前项目中。因此，“每个城市逐站执行阈值比较”的实现只能被文档、计数字段语义和聚合值恒等关系共同支持，不能由原始逐站生成代码独立复核。

Canonical 对账无差异：2025 OOF 有 1,871 个天气触发小时、425 笔最终 Weather Gate 交易；2026 H1 OOT 有 925 个天气触发小时、158 笔最终交易。天气小时数和交易数不是同一指标。

## A–G 核查结论

| 问题 | 结论 | 证据强度 | 核查说明 |
|---|---|---|---|
| A. 六个位置 | 是：Dallas、Fort Worth、Denton、McKinney、Arlington、Wichita Falls | 已核实（文档/PDF） | DFW 五城见设计说明第 80–88 行，Wichita Falls 单列见第 90–98 行；PDF Section II-A 同样列出六地。冻结表 `dfw_city_count` 在两期均恒为 5。 |
| B. 每个位置分别检查三个阈值 | 设计定义为六城分别计数；但逐站生成源码缺失 | 部分核实 | 阈值与 city-count 定义见设计说明第 100–109 行。冻结数据中，`count>0` 与 DFW 极值或 Wichita 值越阈值在 12,957 个评价小时全部一致；但逐城市原始列和生成 count 的上游代码不在本地。 |
| C. 地点和条件之间均为 OR | 条件 OR 已由代码直接核实；地点 OR 由 `city_count>0` 及恒等关系核实 | 已核实，带上游限制 | `refresh_data_architecture_v2.py` 第 311–318 行先把各 count 的 `>0` 转为小时标记，再对冰冻/高温/强阵风作 OR。地点层面的具体逐站循环无法检查。 |
| D. DFW mean/min/max 与 Gate | mean 不触发 Gate；Gate 直接读取 city counts。min/max 可无损重构“任一地点触发”布尔值 | 已核实 | 冰冻等价于 `temperature_dfw_min_c<=0 OR temperature_wichita_c<=0`；高温等价于 `temperature_dfw_max_c>=35 OR temperature_wichita_c>=35`；强风等价于 `wind_gust_dfw_max_ms>=15 OR wind_gust_wichita_ms>=15`。两期均 0 个不一致。均值是预测特征，不是 Gate 条件。 |
| E. Rain/precipitation | 不参与 Gate；作为可用预测特征保留 | 已核实 | `rainy_hour_flag` 单独生成，但未进入第 315–317 行的 OR。降水和 `rainy_city_count` 出现在特征白名单；这证明其可作为预测输入，不表示最终模型必然使用每个降水字段。 |
| F. 预测时次与可用性 | 本地交付小时 00:00–08:00 用 `previous_day1`/24 h；09:00–23:00 用 `previous_day2`/48 h | 已核实（映射）；真实发布时间仅由保守假设支持 | 设计说明第 113–124 行；冻结表逐小时映射 0 个不一致。`decision_time_utc`/`weather_decision_cutoff_utc` 为交付日前一日 09:55 CT，项目截止为 10:00 CT；所有评价小时的记录版本时间均不晚于两者。但本地没有 Open-Meteo 实际发布日志，故“截止前真实可获得”仍依赖保存的 previous-run 映射假设。 |
| G. 最终交易条件 | `S5 = S1(τ=.60) AND sign(predicted_spread) agrees AND weather flag` | 已核实 | `canonical_trading.py` 第 70–76、92–100、121–125 行。无幅度过滤、无额外尾部概率过滤、无仓位缩放；S5 为固定 1 MWh。统一经济结算参数只作用于成交后的 PnL，不增加触发条件。 |

## 已核实的完整触发逻辑

对每个交付小时 `t`：

1. 基础概率信号：若 `p_positive>=0.60` 且严格大于 `p_negative`，则 `S1=+1`；若 `p_negative>=0.60` 且严格大于 `p_positive`，则 `S1=-1`；否则不交易。
2. 方向确认：`S1=+1` 时要求 `predicted_spread>0`；`S1=-1` 时要求 `predicted_spread<0`。等于 0 不通过。
3. 天气标记：
   `fixed_extreme_weather_flag = I(freezing_city_count>0 OR extreme_heat_city_count>0 OR high_wind_city_count>0)`。
4. 最终 S5：`S5=S1` 仅当第 2 步和第 3 步同时通过，否则为 0；活动小时数量固定为 1 MWh。

`rainy_city_count>0` 会生成 `rainy_hour_flag`，但不进入第 3 步。`|predicted_spread|>=5` 属于独立策略 S4；`p_c1+p_c5>=0.15` 属于 S6；置信度缩放属于 S7，均不属于 Weather Gate。

## 数据字段、单位与时间约束

| 字段 | 单位/含义 | Gate 中的作用 |
|---|---|---|
| `freezing_city_count` | 六城中温度 `<=0 °C` 的城市数 | `>0` 触发冰冻小时 |
| `extreme_heat_city_count` | 六城中温度 `>=35 °C` 的城市数 | `>0` 触发高温小时 |
| `high_wind_city_count` | 六城中阵风 `>=15 m/s` 的城市数 | `>0` 触发强风小时 |
| `rainy_city_count` | 六城中降水 `>0.1 mm` 的城市数 | 不参与 Gate |
| `temperature_dfw_mean/min/max_c` | DFW 五城摄氏温度聚合 | mean 为预测特征；min/max 可复核任一 DFW 城越温度阈值 |
| `wind_gust_dfw_mean/max_ms` | DFW 五城阵风聚合，m/s | mean 为预测特征；max 可复核任一 DFW 城越风阈值 |
| `weather_forecast_run_time_utc` | 该交付小时采用的 previous-run 版本时间 | 可用性审计，不作为 Gate 条件 |
| `weather_forecast_lead_hours` | 24 或 48 h | 00:00–08:00 CT 为 24 h；09:00–23:00 CT 为 48 h |

时次检查结果：2025 的 8,760 小时中，24 h 映射 3,285 小时、48 h 映射 5,475 小时；2026 H1 的 4,197 小时中分别为 1,574 和 2,623 小时。两期对 hybrid 本地小时规则、09:55 CT 决策时点和 10:00 CT 项目截止的检查均为 0 个不一致。这里的 `weather_forecast_run_time_utc` 实际按交付小时减 24/48 h 记录为“所用预测版本时间”，并非独立保存的供应商文件发布时间日志。

## 与 canonical 逐小时结果对账

| 期间 | 评价小时 | 冰冻小时 | 高温小时 | 强风小时 | 天气 OR 触发小时 | S1 基础交易 | S3 方向确认交易 | S5 Weather Gate 交易 | canonical S5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 OOF | 8,760 | 440 | 698 | 759 | **1,871** | 1,174 | 933 | **425** | **425** |
| 2026 H1 OOT | 4,197 | 221 | 163 | 553 | **925** | 441 | 353 | **158** | **158** |

逐小时三层对账均通过：冻结输入到预测交接表的时间戳/天气标记 0 个不一致；由 `τ=.60 + direction + weather` 重算的 S5 与 canonical hourly S5 在两期均 0 个不一致。雨天小时另为 2025 年 936、2026 H1 488，仅作旁证，不计入天气 OR。

## 源代码与证据定位

- 最终投稿 PDF：`PESIM2027_ERCOT_ExtremeWeather_VirtualBidding.pdf`，Section II-A/II-C（六地、阈值、previous-run 映射、Weather Gate 层级）；SHA256 `47dfc0b8...c10a6f4`。
- 宽表来源绑定：`02_prediction_agent_v2/data_audit/refresh_data_architecture_v2.py:33-35`。
- 截止时点构造与天气可用性检查：同文件 `:71-76`、`:112-120`。
- 小时标记和最终天气 OR：同文件 `:311-318`。
- 六城、阈值、预测版本设计：`02_prediction_agent_v2/model_wide_explanation_zh_2024_2026(1).md:76-124`；字段和单位见 `:180-245`。
- 冻结与白名单生成：`02_prediction_agent_v2/data_audit/freeze_model_dataset_v2.py:16-31`、`:141-176`、`:184-205`。
- 最终 S1/S3/S5：`04_trading_strategy_evaluation/section_iv_canonical_v1/src/canonical_trading.py:70-76`、`:91-125`。
- canonical 嵌套关系断言：`04_trading_strategy_evaluation/section_iv_canonical_v1/src/run_section_iv.py:84-88`。
- 策略声明：`04_trading_strategy_evaluation/section_iv_canonical_v1/config/strategy_implementation_spec.csv:6`。
- 完整输入绝对路径和 SHA256：`results/evidence_inventory_sha256.csv`；逐小时身份检查和层间对账分别见 `results/weather_gate_field_identity.csv` 与 `results/weather_gate_layer_reconciliation.csv`。

## 尚无法核实及与描述的边界

1. 设计说明第 18–20 行引用的 `01_data_collection_cleaning/interim/ercot_analytics.sqlite`、对应表以及 processed CSV 不在当前本地项目；当前 refresh 直接读取 `model_wide_hourly_2024_2026_final(2).csv`，其中 city-count 字段已预先生成。
2. 未找到逐城市 Open-Meteo 原始列或将六站值转换为 `freezing_city_count`、`extreme_heat_city_count`、`high_wind_city_count`、`rainy_city_count` 的生成脚本。因此不能宣称已从原始逐站 API 响应复跑该步骤。
3. 保存的聚合值和 city counts 在所有评价小时完全自洽，足以复现最终布尔 Gate，但不能恢复“具体哪一座城市”触发或验证每个站点的 API 元数据。
4. 截止前可用性由保存的 previous-run 版本时间和保守 hybrid 映射支持；没有供应商真实发布时间日志，不能把该假设升级为独立观测事实。
5. 原论文方法描述与最终 Gate 规则未发现数值冲突；需要在返修文字中避免把 DFW mean 说成 Gate 阈值输入，也应把天气小时数与 Gate 后交易数分开。

## 可直接用于 Section II-A 的英文方法说明

> Forecast weather was represented over Dallas, Fort Worth, Denton, McKinney, Arlington, and Wichita Falls. The frozen dataset supplies hourly counts of locations meeting fixed physical thresholds for freezing (temperature <= 0 degC), extreme heat (temperature >= 35 degC), and high wind (gust >= 15 m/s). The weather indicator equals one when any of these three counts is nonzero; precipitation and DFW mean/min/max summaries remain predictive features but do not independently activate the gate. The trading gate retains a 1-MWh B2B probability signal at tau = 0.60 only when the continuous spread forecast agrees in sign and the fixed weather indicator is one. Forecasts are aligned to delivery hours using a conservative previous-run mapping (24-h versions for local hours 00:00–08:00 and 48-h versions otherwise), so the timing claim is conditional on this stored availability assumption.

## 可复现命令

```powershell
python revision\PESIM2027_R2_3\audit_weather_gate.py `
  --project-root "D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project" `
  --submitted-pdf "D:\NUS SOC2026\会议论文\PESIM2027_ERCOT_ExtremeWeather_VirtualBidding.pdf"
```

脚本只读输入并在本目录的 `results/` 中写审计结果。Run ID 由审计版本及全部关键输入 SHA256 稳定生成。

## Claim–evidence matrix

| Claim ID | Section | Claim text | Claim type | Evidence status | Run ID | Artifact | Citation keys | Audit decision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R2.3-A | II-A | Six weather locations are Dallas, Fort Worth, Denton, McKinney, Arlington, and Wichita Falls. | method | Supported | `R2_3_WG_C64C6FDF13E0FE54` | submitted PDF; design document lines 76–111 | N/A | PASS | DFW city count is 5 in all audited rows. |
| R2.3-B | II-A | Each location is thresholded at 0/35 degC and 15 m/s before city counting. | method/provenance | Supported | `R2_3_WG_C64C6FDF13E0FE54` | design document lines 100–109; field-identity CSV | N/A | REVISE | Qualify the wording: upstream per-city generation code and raw station columns are missing. |
| R2.3-C | II-A | Any qualifying location or included condition activates the weather indicator. | method | Supported | `R2_3_WG_C64C6FDF13E0FE54` | refresh code lines 311–318; field-identity CSV | N/A | PASS | Condition OR is direct code; location OR is count>0 and aggregate-equivalent. |
| R2.3-D | II-A | Rain does not activate the Weather Gate. | method | Supported | `R2_3_WG_C64C6FDF13E0FE54` | refresh code lines 311–318; feature whitelist | N/A | PASS | Rain/precipitation remains available to prediction features. |
| R2.3-E | II-C | S5 is tau=.60 B2B signal plus sign confirmation plus weather indicator, at 1 MWh. | method | Supported | `R2_3_WG_C64C6FDF13E0FE54` | canonical trading lines 70–125; layer reconciliation CSV | N/A | PASS | Hour-by-hour reproduced; no magnitude, tail-probability, or scaling filter in S5. |
| R2.3-F | IV | Weather Gate executes 425 trades in 2025 OOF and 158 in 2026 H1 OOT. | empirical result | Supported | `R2_3_WG_C64C6FDF13E0FE54` | weather_gate_counts.csv; canonical hourly parquets | N/A | PASS | Hour-by-hour reproduced; weather-trigger hours are 1,871 and 925, respectively. |
| R2.3-G | II-A | All forecast versions were available before the DAM deadline. | timing/provenance | Supported | `R2_3_WG_C64C6FDF13E0FE54` | time-audit CSV; design document lines 113–124 | N/A | REVISE | State that availability is conditional on the stored conservative mapping; no independent provider publication log survives locally. |

质量检查：`check_result_consistency.py --audit` 通过。`validate_bib.py` 报告授权文件 `references/gridwise.bib` 不存在；本报告未新增或替代任何文献条目，且本次方法核查不依赖外部引文。
