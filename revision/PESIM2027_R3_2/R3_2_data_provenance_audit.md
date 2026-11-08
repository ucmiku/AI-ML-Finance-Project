# PES IM 2027 — Reviewer 3.2 数据来源核查

**审计结论：最终天气字段来自 Open-Meteo Previous Runs API；具体数值天气模型无法确认。**

**Run ID：** `R3_2_PROV_F78ADB0C81CB7F16`

**Git 证据基准：** `origin/datacollection` commit `82da6f7dc81e66433dad7d98885e1d43d5d8db02`

## 直接回答

1. **最终天气数据来自哪个 API？** 最终 `model_wide_hourly_2024_2026_final(2).csv` 和 `model_input_frozen_v2.parquet` 中的天气字段来自 `https://previous-runs-api.open-meteo.com/v1/forecast`，采用 `previous-runs-hybrid` 数据集。最终清洗 SQL 只接受 `f.dataset='previous-runs-hybrid'`，因此早期 Historical Forecast 和实验性的 Single Runs 均未进入最终冻结输入。
2. **能确认具体天气模型吗？** 不能。Previous Runs 请求没有发送 `models` 参数；`openmeteo_previous_runs_default_model` 是项目代码写入的标签，不是响应中恢复出的 ECMWF、GFS 或其他具体模型标识。现存过滤后数据和冻结表也没有保留供应商返回的具体底层模型信息。
3. **市场参与者如何获取同类预测？** 可通过 Open-Meteo Previous Runs API，使用本报告列出的六地坐标、变量、单位和 `_previous_day1/_previous_day2` 参数复现归档预测。该 API 的归档查询可用于事后复现；要证明交易当时的可获得性，市场参与者还需要在 D-1 实时抓取并保存 HTTP 获取时间、响应、模型运行标识及发布日志。本项目没有保留这类参与者侧日志。
4. **原始响应和 metadata 是否仍在？** 当前 `D:\NUS SOC2026` 搜索范围内未找到 Open-Meteo raw response、`.metadata.json`、`ercot_data.sqlite` 或 `ercot_analytics.sqlite`。因此无法恢复原始响应 SHA256、真实采集时间或响应级模型元数据；现存可复现链从最终 model-wide CSV 开始。

## 采集实现与请求参数

`datacollection` 分支的 `01_data_collection_cleaning/scripts/collectors/openmeteo_collector.py` 明确保存了三类端点：

| 模式 | Endpoint | 本项目阶段 |
|---|---|---|
| Historical Forecast | `https://historical-forecast-api.open-meteo.com/v1/forecast` | 早期数据阶段；进入旧版 model-wide，但未进入最终冻结输入 |
| Previous Runs | `https://previous-runs-api.open-meteo.com/v1/forecast` | **最终采用**；`previous-runs-hybrid` |
| Single Runs | `https://single-runs-api.open-meteo.com/v1/forecast` | 曾实现 ECMWF single-run 采集与恢复，但最终清洗 SQL 未选用 |

最终 Previous Runs 请求由代码第 369–383 行构造：

| 参数 | 实际值 |
|---|---|
| `latitude` | 六个纬度以逗号连接 |
| `longitude` | 六个经度以逗号连接 |
| `start_date`, `end_date` | 按 31 日分块；查询窗口在研究区间两侧各扩展一天 |
| `hourly` | 七个基础变量分别请求 `_previous_day1` 和 `_previous_day2` |
| `timezone` | `UTC` |
| `temperature_unit` | `celsius` |
| `wind_speed_unit` | `ms` |
| `precipitation_unit` | `mm` |
| `models` | **未设置** |
| `apikey` | 可选；保存到 metadata 前会移除 |

七个基础变量为 `temperature_2m`、`relative_humidity_2m`、`wind_speed_10m`、`wind_gusts_10m`、`cloud_cover`、`shortwave_radiation` 和 `precipitation`。代码定义见 `openmeteo_collector.py:53-61`。

### 六个请求地点

| 地点 | 纬度 | 经度 |
|---|---:|---:|
| Dallas | 32.7767 | -96.7970 |
| Fort Worth | 32.7555 | -97.3308 |
| Denton | 33.2148 | -97.1331 |
| McKinney | 33.1972 | -96.6397 |
| Arlington | 32.7357 | -97.1081 |
| Wichita Falls | 33.9137 | -98.4934 |

坐标直接来自 `openmeteo_collector.py:71-78`，不是从论文文字反推。

## Historical → Previous Runs 的切换证据

`DATA_COLLECTION_STATUS.md` 和 `DATA_INVENTORY.md` 生成于 2026-07-18，描述的是早期 Historical Forecast 数据，并明确警告原始响应没有精确 `issued_at`。`datacollection` 历史显示：

- `880c7ad`（2026-07-19）：早期模型宽表，天气清洗选择六地 Historical Forecast。
- `8e0734e`（2026-07-22，提交说明“最新宽表”）：把 `build_clean.py` 的天气来源改为精确匹配 `previous-runs-hybrid`，加入 24/48 h 规则和天气截止检查。
- 当前远端 `datacollection` 为 `82da6f7`；后续两个提交未改变上述数据采集逻辑。

因此，状态文档不能代表最终输入来源；最终清洗代码和最终 CSV/frozen 字段优先级更高。

本地也保留了两代结果：

| 层 | 行数 | UTC 覆盖 | 来源标记 |
|---|---:|---|---|
| 旧版 model-wide | 21,855 | 2024-01-02 07:00 至 2026-06-30 23:00 | `historical_forecast_without_publish_time` |
| 最终 model-wide | 21,432 | 2024-01-19 12:00 至 2026-07-01 04:00 | `openmeteo_previous_day1_local_hours_00_08_else_day2_before_pre_dam_cutoff` |
| 冻结模型输入 | 21,093 | 2024-01-28 06:00 至 2026-07-01 04:00 | 同上 |

两代天气值并非只改了标签。在 21,427 个共同小时中，DFW 平均温度有 21,334 小时不同，DFW 最大阵风有 21,005 小时不同，DFW 平均降水有 3,149 小时不同。这支持“实际换源”，而非事后改名。

## 从 API 到最终模型的链条

1. `openmeteo_collector.py:298-485` 请求 Previous Runs API，并按目标小时选择 `_previous_day1` 或 `_previous_day2`。
2. `build_clean.py:420-535` 只读取 `previous-runs-hybrid` 六地记录，提取温度、湿度、风速、阵风、云量、辐射和降水。
3. `build_clean.py:538-638` 对 DFW 五城计算 mean/min/max，单独保留 Wichita Falls，并在全部六城上计算极端天气计数。
4. `build_model_wide.py:121-160` 将天气字段、计数、来源标签和 24/48 h 字段写入最终 model-wide 表。
5. 本次逐字段对账显示：最终 model-wide 到 `model_input_frozen_v2.parquet` 的 40 个天气来源、聚合、计数及审计字段，在全部 21,093 个冻结小时中均为 **0 个不一致**。
6. 2025 OOF 的 8,760 小时和 2026 H1 OOT 的 4,197 小时继续复现天气触发 1,871/925 小时以及 Weather Gate 交易 425/158 笔。

这条链能够证明最终模型使用 Previous Runs 数据，也能够从最终 model-wide 层复现冻结输入；但由于原始 API 文件、SQLite 和 metadata 不在本地或 Git 中，不能把下游一致性提升为“原始响应已验证”。

## Single Runs 为什么不是最终来源

`openmeteo_collector.py:576-857` 确实实现了 Single Runs：endpoint 为 `single-runs-api`，默认 `models=ecmwf_ifs`，并以固定运行时次抓取六地天气；`recover_single_run_weather.py` 也用于补抓不完整日期。但最终 `build_clean.py:481-487` 采用精确条件 `f.dataset='previous-runs-hybrid'`，不会读入 `single-run-ecmwf`。所以不能因为仓库中存在 Single Runs/ECMWF 代码，就把最终冻结数据描述为 ECMWF Single Runs。

`import_pre_dam_forecast_features.py` 处理的是 ERCOT load/wind/solar 预测字段，并不决定天气来源。

## 原始证据缺失范围

采集框架原本会给每个 raw 文件生成 sibling metadata，保存 endpoint、无密钥请求参数、`collected_at_utc`、行数和 raw 文件 SHA256（`collectors/common.py:128-178`）。然而当前本地与 Git 分支仅保留 `.gitkeep`，没有原始天气响应、metadata 或 SQLite。因此：

- 原始 Previous Runs 响应 SHA256：**无法确认**；
- 实际采集日期：**无法确认**；
- API 返回的网格坐标和单位 metadata：代码有保存设计，但原始值无法复核；
- 具体底层天气模型：请求未指定，过滤后响应也未保存可识别信息，**无法确认**；
- 从原始逐站响应重新构建最终 CSV：当前材料不足；
- 从最终 model-wide CSV重新冻结天气字段及 Gate：已逐字段核实可复现。

完整 Git blob SHA256、最终输入 SHA256 和本地搜索清单见 `results/branch_evidence_sha256.csv`、`results/local_evidence_sha256.csv` 与 `results/local_provenance_file_inventory.csv`。

## Section II-A 的准确表述边界

可以写：最终天气协变量由 Open-Meteo Previous Runs API 的归档预测构建；六地使用 provider default configuration；00:00–08:00 CT 选择 `_previous_day1`，其余小时选择 `_previous_day2`；请求未指定具体模型。

不能写：最终数据来自 ECMWF/GFS；原始响应包含精确发布时间；或者所有预测均已由供应商发布时间日志证明在 DAM 截止前可获得。

## Claim–evidence matrix

| Claim ID | Section | Claim text | Claim type | Evidence status | Run ID | Artifact | Citation keys | Audit decision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R3.2-P1 | II-A | Final frozen weather covariates originate from the Open-Meteo Previous Runs API. | method/provenance | supported | `R3_2_PROV_F78ADB0C81CB7F16` | datacollection `build_clean.py:420-535`; provenance layers; field reconciliation | N/A | pass | Final SQL selects only `previous-runs-hybrid`. |
| R3.2-P2 | II-A | The exact underlying NWP model is ECMWF or GFS. | provenance | unsupported | `R3_2_PROV_F78ADB0C81CB7F16` | recovered request spec; collector lines 369-383 | N/A | stop | Previous Runs request omits `models`; do not name a model. |
| R3.2-P3 | II-A | Historical Forecast data were superseded before the final freeze. | provenance | supported | `R3_2_PROV_F78ADB0C81CB7F16` | git history; source transition comparison | N/A | pass | Old and final weather values differ materially. |
| R3.2-P4 | II-A | Single Runs/ECMWF entered the final frozen input. | provenance | unsupported | `R3_2_PROV_F78ADB0C81CB7F16` | collector Single Runs code; final clean SQL | N/A | stop | Code exists, but final SQL excludes its dataset. |
| R3.2-P5 | II-A | Final model-wide and frozen weather fields are identical for 21,093 hours. | numerical empirical | supported | `R3_2_PROV_F78ADB0C81CB7F16` | `weather_layer_reconciliation.csv` | N/A | pass | Forty compared fields; zero mismatches. |
| R3.2-P6 | Data availability | Original raw responses, metadata SHA256, and collection SQLite are locally verifiable. | provenance | unsupported | `R3_2_PROV_F78ADB0C81CB7F16` | local provenance file inventory | N/A | stop | No such artifacts found under `D:\NUS SOC2026`. |

## 可复现命令

```powershell
python revision\PESIM2027_R3_2\audit_data_provenance.py `
  --repo-root "D:\NUS SOC2026\AI-ML-Finance-Project" `
  --project-root "D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project" `
  --submitted-pdf "D:\NUS SOC2026\会议论文\PESIM2027_ERCOT_ExtremeWeather_VirtualBidding.pdf" `
  --search-root "D:\NUS SOC2026"
```

质量检查：两份 R3.2 报告均通过 `check_result_consistency.py --audit`。`validate_bib.py` 仍报告授权文件 `references/gridwise.bib` 不存在；本次证据核查未新增或替代任何文献条目。
