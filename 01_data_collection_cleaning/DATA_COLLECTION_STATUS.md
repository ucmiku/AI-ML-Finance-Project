# 数据获取状态总结

生成日期：2026-07-18  
统计目录：`01_data_collection_cleaning/raw/`  
目标时间范围：2022-01-01 至 2025-12-31  
当前阶段：raw 数据已落盘，SQLite raw 层和 clean 层均已建成

## 总览

当前 raw 层共有：

| 指标 | 数值 |
|---|---:|
| 原始数据文件 | 376 |
| 配套 metadata 文件 | 376 |
| 压缩后总大小 | 14,702,249 bytes，约 14.02 MiB |
| 数据格式 | `csv.gz`、`json.gz` |
| 元数据格式 | `.metadata.json` |

## Clean 层

- 数据库：`01_data_collection_cleaning/interim/ercot_analytics.sqlite`
- Schema：v3
- 大小：165,588,992 bytes，约 157.70 MiB
- 统一时区：UTC，时间格式为 `YYYY-MM-DDTHH:MM:SSZ`
- 天气范围：仅 Dallas、Fort Worth、Denton、McKinney、Arlington、Wichita Falls
- 天气去重键：`city + target_hour_utc`
- 暂不纳入：风电和光伏预测
- 完整性检查：`ok`

详细表结构及行数见 `CLEAN_DATA_SCHEMA.md`。

## 已获取数据清单

| 状态 | 数据集 | 来源 | API / Series ID | 频率 | 文件数 | 当前行数 | 当前覆盖范围 |
|---|---|---|---|---|---:|---:|---|
| 已完成 | ERCOT North Hub 日前电价 | GridStatus.io | `ercot_spp_day_ahead_hourly` | 小时 | 48 | 35,064 | 2022-01-01 至 2025-12-31 |
| 已完成 | ERCOT North Hub 实时电价 | GridStatus.io | `ercot_spp_real_time_15_min` | 15 分钟 | 48 | 140,256 | 2022-01-01 至 2025-12-31 |
| 已完成 | ERCOT Seven-Day Load Forecast | GridStatus.io | `ercot_load_forecast` | 5 分钟 | 209 | 420,768 | 2022-01-01 至 2025-12-31 |
| 部分获取 | ERCOT Wind Production Forecast | GridStatus.io | `ercot_wind_actual_and_forecast_hourly` | 小时，多 forecast vintage | 2 | 72,575 | 2022-01-01 至 2022-01-14 |
| 未获取 | ERCOT Solar Production Forecast | GridStatus.io | `ercot_solar_actual_and_forecast_hourly` | 小时，多 forecast vintage | 0 | 0 | 尚未开始 |
| 已完成（旧配置） | Austin 历史天气预报 | Open-Meteo | Historical Forecast API | 小时 | 17 | 35,088 | 2022-01-01 至 2025-12-31 |
| 已完成（旧配置） | Dallas 历史天气预报 | Open-Meteo | Historical Forecast API | 小时 | 17 | 35,088 | 2022-01-01 至 2025-12-31 |
| 已完成（旧配置） | Houston 历史天气预报 | Open-Meteo | Historical Forecast API | 小时 | 17 | 35,088 | 2022-01-01 至 2025-12-31 |
| 已完成（旧配置） | San Antonio 历史天气预报 | Open-Meteo | Historical Forecast API | 小时 | 17 | 35,088 | 2022-01-01 至 2025-12-31 |
| 已完成并入库 | Dallas、Fort Worth、Denton、McKinney、Arlington、Wichita Falls 北德州天气 | Open-Meteo | Historical Forecast API | 小时 | 324 | 236,592 个唯一小时 | 2022-01-01 至 2026-07-01 UTC |
| 已完成 | Henry Hub 天然气现货价格 | FRED | `DHHNGSP` | 工作日日度 | 2 | 1,172 | 2022-01-03 至 2026-06-30 |

## GridStatus.io 数据

### 1. ERCOT North Hub 日前电价

- 本地 raw 目录：`01_data_collection_cleaning/raw/gridstatusio/ercot_spp_day_ahead_hourly/`
- API 数据集：`ercot_spp_day_ahead_hourly`
- 筛选条件：`location = HB_NORTH`
- 市场：`DAY_AHEAD_HOURLY`
- 当前状态：完整覆盖项目目标区间
- 主要字段：

| 字段 | 含义 |
|---|---|
| `interval_start_local` | ERCOT 本地交付小时开始时间 |
| `interval_start_utc` | UTC 交付小时开始时间 |
| `interval_end_local` | ERCOT 本地交付小时结束时间 |
| `interval_end_utc` | UTC 交付小时结束时间 |
| `location` | 结算点，本项目为 `HB_NORTH` |
| `location_type` | 地点类型，本项目为 `Trading Hub` |
| `market` | 市场类型，本项目为 `DAY_AHEAD_HOURLY` |
| `spp` | Settlement Point Price，USD/MWh |

### 2. ERCOT North Hub 实时电价

- 本地 raw 目录：`01_data_collection_cleaning/raw/gridstatusio/ercot_spp_real_time_15_min/`
- API 数据集：`ercot_spp_real_time_15_min`
- 筛选条件：`location = HB_NORTH`
- 市场：`REAL_TIME_15_MIN`
- 当前状态：完整覆盖项目目标区间
- 后续清洗要求：需要将 15 分钟实时价格聚合为小时实时价格，再计算：

```text
spread = rt_price_hourly - da_price
```

### 3. ERCOT Seven-Day Load Forecast

- 本地 raw 目录：`01_data_collection_cleaning/raw/gridstatusio/ercot_seven_day_load_forecast/`
- API 数据集：`ercot_load_forecast`
- 当前状态：完整覆盖项目目标区间
- 当前行数：420,768
- 主要字段：

| 字段 | 含义 |
|---|---|
| `interval_start_local` | 预测目标时间，本地时间 |
| `interval_start_utc` | 预测目标时间，UTC |
| `interval_end_local` | 预测目标结束时间，本地时间 |
| `interval_end_utc` | 预测目标结束时间，UTC |
| `publish_time_local` | 预测发布时间，本地时间 |
| `publish_time_utc` | 预测发布时间，UTC |
| `load_forecast` | ERCOT load forecast |

### 4. ERCOT Wind Production Forecast

- 本地 raw 目录：`01_data_collection_cleaning/raw/gridstatusio/ercot_wind_production_forecast/`
- API 数据集：`ercot_wind_actual_and_forecast_hourly`
- 当前状态：部分获取
- 当前覆盖：2022-01-01 至 2022-01-14
- 当前行数：72,575
- 说明：该数据集包含多个 forecast vintage，数据量明显大于普通小时序列。
- 主要字段包括：

| 字段 | 含义 |
|---|---|
| `interval_start_utc` | 预测目标小时 |
| `publish_time_utc` | 预测发布时间 |
| `gen_system_wide` | 系统级实际风电出力 |
| `cop_hsl_system_wide` | 系统级 COP HSL |
| `stwpf_system_wide` | Short-Term Wind Power Forecast |
| `wgrpp_system_wide` | Wind Generation Resource Production Potential |
| `gen_lz_*` | 分负荷区实际风电出力 |
| `stwpf_lz_*` | 分负荷区短期风电预测 |
| `wgrpp_lz_*` | 分负荷区风电潜力预测 |

### 5. ERCOT Solar Production Forecast

- 本地 raw 目录：尚未生成
- API 数据集：`ercot_solar_actual_and_forecast_hourly`
- 当前状态：尚未获取
- 主要字段预计包括：

| 字段 | 含义 |
|---|---|
| `interval_start_utc` | 预测目标小时 |
| `publish_time_utc` | 预测发布时间 |
| `gen_system_wide` | 系统级实际光伏出力 |
| `cop_hsl_system_wide` | 系统级 COP HSL |
| `stppf_system_wide` | Short-Term Photovoltaic Power Forecast |
| `pvgrpp_system_wide` | PV Generation Resource Production Potential |

## Open-Meteo 天气数据

- 本地 raw 目录：`01_data_collection_cleaning/raw/openmeteo/`
- 来源：Open-Meteo Historical Forecast API
- 保留的旧 raw 地点：Austin、Dallas、Houston、San Antonio
- 新默认地点：Dallas、Fort Worth、Denton、McKinney、Arlington、Wichita Falls
- 当前状态：新六地点均已完成采集并写入 SQLite raw 层
- 覆盖范围：2022-01-01 至 2026-07-01 UTC
- 每个新地点唯一小时数：39,432
- 六地点合计唯一小时数：236,592
- 时间频率：小时
- 时区：UTC

Dallas 的新旧分块存在时间重叠，旧四地点中还保留了早期单日测试记录。raw 层不删除来源记录；后续 cleaned 表应按：

```text
location + interval_start_utc
```

去重。

天气变量包括：

| 字段 | 单位 | 含义 |
|---|---|---|
| `temperature_2m` | Celsius | 2 米气温 |
| `relative_humidity_2m` | % | 2 米相对湿度 |
| `wind_speed_10m` | m/s | 10 米风速 |
| `wind_gusts_10m` | m/s | 10 米阵风 |
| `cloud_cover` | % | 云量 |
| `shortwave_radiation` | W/m^2 | 短波辐射 |
| `precipitation` | mm | 降水量 |

注意：Open-Meteo Historical Forecast 原始响应没有精确的历史发布时间 `issued_at`。建模时不能声称这些天气数据一定是日前投标截止时刻真实可见的 forecast vintage，需要在报告中说明该限制。

## FRED 天然气数据

- 本地 raw 目录：`01_data_collection_cleaning/raw/fred/DHHNGSP/`
- Series ID：`DHHNGSP`
- 含义：Henry Hub Natural Gas Spot Price
- 频率：工作日日度
- 当前状态：完整覆盖目标区间内可用观测
- 当前行数：1,172
- 单位：USD/MMBtu

当前 clean 处理：

- FRED 中的 `.` 已转换为 SQL `NULL`，共保留 97 个缺失值标记。
- `realtime_start` 只代表两次采集快照版本，不是原始发布日期。
- 合并到小时级模型表前仍需定义保守的发布滞后规则。

## 数据库状态与剩余项

| 项目 | 状态 | 说明 |
|---|---|---|
| ERCOT Wind/Solar Forecast | 暂缓 | 已决定当前 baseline 不纳入风光预测 |
| raw 数据写入 SQLite | 已完成 | `ercot_data.sqlite` |
| clean 数据库 | 已完成 | `ercot_analytics.sqlite` schema v3 |
| RT 15 分钟聚合及 `RT - DA` | 已完成 | 39,405 个完整标签 |
| 六城市天气去重 | 已完成 | 236,592 个唯一 city-hour |
| 天气区域特征 | 已完成 | 39,432 个 DFW/Wichita 小时特征 |
| FRED 缺失值处理 | 已完成 | `.` 转换为 SQL `NULL` |
| Load Forecast as-of 特征 | 已完成但覆盖不足 | 09:55 CT规则，当前仅168小时 |
| Gas 可用时间规则 | 已完成 | 次一工作日可用，仅向前填充 |
| 时间顺序划分 | 已完成 | 70% train / 15% validation / 15% test |
| 模型视图 | 已完成但load稀疏 | `vw_model_dataset_hourly`，39,405小时 |

## 建议的下一步

1. 重新获取完整的 ERCOT load forecast historical vintages，扩大当前168小时as-of覆盖。
2. 由模型训练部分决定是否使用稀疏load特征，baseline应先使用时间、天气和gas。
3. 进行训练、验证和测试集上的特征消融实验。
4. 导出模型训练数据，核心目标变量为：

```text
spread = rt_price_hourly - da_price
```
