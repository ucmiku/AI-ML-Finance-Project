# PES IM 2027 — Reviewer 3.2 预测可用性核查

**审计结论：24/48 h 映射在代码和冻结数据中完全一致，但不足以独立证明 DAM 截止前真实可用。**

**Run ID：** `R3_2_PROV_F78ADB0C81CB7F16`

## 五类时间必须分开

| 时间概念 | 现存字段/证据 | 是否真实观测 | 结论 |
|---|---|---|---|
| 1. 预测目标小时 | API `hourly.time` → `delivery_hour_utc` | 是 | 每条天气值对应的 ERCOT 交付小时 |
| 2. 模型初始化或参考时间 | `weather_forecast_run_time_utc` | **否，是派生值** | Previous Runs collector 用 `target - 24/48 h` 人工生成；不能视为 API 返回的模型初始化时间 |
| 3. 预测结果实际发布时间 | 无字段、无供应商日志 | 否 | 无法逐小时确认 |
| 4. 市场参与者侧可获得时间 | 无抓取日志或接收时间 | 否 | 无法逐小时确认 |
| 5. DAM 决策冻结时间 | `weather_decision_cutoff_utc` | 项目规则 | 交付日前一天 09:55 America/Chicago |

关键代码是 `openmeteo_collector.py:246-270`：先根据目标小时的 CT 小时选择 24 或 48，再把 `target - lead_hours` 写入 `forecast_run_time_utc`。该字段不是 Open-Meteo 响应中的 `issued_at`、`publish_time` 或参与者接收时间。

## 24/48 h 规则核查

| 期间 | 评价小时 | 24 h (`previous_day1`) | 48 h (`previous_day2`) | 派生参考时间不早于截止 | 目标减参考时差错误 |
|---|---:|---:|---:|---:|---:|
| 2025 OOF | 8,760 | 3,285 | 5,475 | 0 | 0 |
| 2026 H1 OOT | 4,197 | 1,574 | 2,623 | 0 | 0 |

这只能证明代码实施了预定映射以及派生参考时间早于 09:55 CT，不能证明预测结果已在该参考时间发布。

## 截止裕量与风险小时

按照派生参考时间计算：

| 期间 | 最小名义裕量 | `<=120 min` 小时 | `<=60 min` 小时 | 这些小时中的天气触发 | 这些小时中的 S5 交易 |
|---|---:|---:|---:|---:|---:|
| 2025 OOF | **55 min** | 365 | 1 | 61 | 22 |
| 2026 H1 OOT | **115 min** | 174 | 0 | 43 | 8 |

2025 年最小值发生在 `2025-11-02 08:00 CST`：目标为 `2025-11-02T14:00Z`，代码生成的参考时点为 `2025-11-01T14:00Z`，而截止为 `2025-11-01T14:55Z`，只剩 55 分钟。原因是代码按目标本地小时选择 `previous_day1`，随后在 UTC 中固定减 24 小时；DST 回拨使“前一天相同 UTC 时刻”不等于“前一天相同本地时刻”。该小时仍没有实际发布时间日志，所以不能判断是否使用了截止后数据。

所有 24 h 行也存在程度不同的发布延迟不确定性：

| 期间 | 24 h 天气触发小时 | 24 h S5 交易 |
|---|---:|---:|
| 2025 OOF | 516 | 54 |
| 2026 H1 OOT | 331 | 21 |

这些数字表示潜在受“发布时间未验证”影响的范围，不表示已经发生泄漏。

## 是否存在截止后信息？

- **已核实：** 没有任何小时的人工生成 `forecast_run_time_utc` 晚于或等于 09:55 CT 截止。
- **无法确认：** 实际模型初始化时间、生成完成时间、Open-Meteo 发布时间和参与者接收时间均未保存。
- **因此：** 不能证明存在截止后信息，也不能排除。论文中“a run issued after the submission cutoff cannot enter”强于现存证据，应改成“a conservative lag mapping was imposed; actual publication times were unavailable and remain a limitation”。

## 更保守替代检验是否可行

可行的最小敏感性设计是：对本地 00:00–08:00 的目标小时也统一使用 `_previous_day2`，从而把全部天气输入至少再后移一天，并保持 2025/2026 规则不变。

采集请求当时同时请求了 day1 和 day2，但 collector 写入磁盘的是筛选后的统一变量，未保留未选中的另一套值；当前本地也没有原始 HTTP 响应或 SQLite。因此，**现有冻结文件不能直接完成替代检验**。需要重新查询 Previous Runs API到新的只读审计目录，重新生成替代天气特征，然后用冻结模型/规则开展敏感性评分；本轮未执行该实验，也未重训模型。

另一种更强的未来证据是实时记录每次 API 响应的获取时间、响应 SHA256、模型运行标识和参与者侧可见时间。只有这种日志才能直接证明 DAM 截止前可用。

## 与最终样本的一致性

| 期间 | 样本小时 | 天气触发小时 | Weather Gate 交易 |
|---|---:|---:|---:|
| 2025 OOF | 8,760 | 1,871 | 425 |
| 2026 H1 OOT | 4,197 | 925 | 158 |

以上数字与 canonical 结果一致。它们验证的是下游数据和交易规则，不构成原始发布时间证明。

## Section II-A 英文修改建议

> Weather covariates were retrieved retrospectively from the Open-Meteo Previous Runs API for Dallas, Fort Worth, Denton, McKinney, Arlington, and Wichita Falls. The request did not specify a `models` parameter; therefore, the data represent the provider's default model configuration rather than an identified ECMWF or GFS product. For each delivery hour, the archived `_previous_day1` values were selected for local hours 00:00–08:00 and `_previous_day2` values otherwise. This 24/48-h lag rule was imposed before modeling and is reproduced exactly in the frozen input. However, the retained artifacts do not contain provider publication timestamps or participant-side retrieval logs; consequently, pre-DAM availability is treated as a conservative mapping assumption rather than a directly observed publication-time fact.

## 建议给审稿人的结论

Reviewer concern → added evidence：补充 `datacollection` 分支的 API、请求参数、换源提交历史，以及 final model-wide 到 frozen input 的逐字段对账。

Manuscript change：把“截止后 run 不可能进入”收窄为“实施了固定 24/48 h 滞后映射，但真实发布时间未被原始日志验证”，并明确默认模型不可识别。

Remaining limitation：原始响应、metadata SHA256、实际发布时间和参与者接收时间缺失；建议将统一 previous_day2 作为后续保守敏感性检验，而不是在本轮声称其结果。

## Claim–evidence matrix

| Claim ID | Section | Claim text | Claim type | Evidence status | Run ID | Artifact | Citation keys | Audit decision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R3.2-A1 | II-A | The frozen input implements the 24/48-h Previous Runs mapping exactly. | numerical empirical | supported | `R3_2_PROV_F78ADB0C81CB7F16` | `forecast_availability_summary.csv`; frozen parquet | N/A | pass | Zero mapping or target-minus-reference mismatches. |
| R3.2-A2 | II-A | `forecast_run_time_utc` is an observed supplier publication timestamp. | timing/provenance | unsupported | `R3_2_PROV_F78ADB0C81CB7F16` | collector lines 246-270; source assertions | N/A | stop | It is constructed as target minus 24/48 h. |
| R3.2-A3 | II-A | Every weather forecast was available to a market participant before 09:55 CT. | operational availability | unsupported | `R3_2_PROV_F78ADB0C81CB7F16` | availability summary; local inventory | N/A | stop | No publication or participant-side retrieval log survives. |
| R3.2-A4 | II-A | No constructed reference timestamp is at or after the cutoff. | numerical empirical | supported | `R3_2_PROV_F78ADB0C81CB7F16` | `forecast_availability_summary.csv` | N/A | pass | This does not prove actual publication availability. |
| R3.2-A5 | Limitation | The minimum nominal cutoff margin is 55 min in 2025 and 115 min in 2026 H1. | numerical empirical | supported | `R3_2_PROV_F78ADB0C81CB7F16` | availability summary; edge-hours CSV | N/A | pass | The 55-min case is the 2025 DST fall-back day. |
| R3.2-A6 | Robustness | Replacing day1 with day2 for local hours 00:00-08:00 has been evaluated. | proposed |  |  | N/A | revise | Feasible future sensitivity; not executed in this audit. |
