# 原始数据清单

生成日期：2026-07-18  
统计目录：`01_data_collection_cleaning/raw/`  
目标时间范围：2022-01-01 至 2025-12-31

## 总览

| 状态 | 数据 | 来源 | 频率 | 覆盖范围 | 有效记录数 |
|---|---|---|---|---|---:|
| 已获取 | North Hub 日前电价 | GridStatus.io | 小时 | 2022-01-01 至 2025-12-31 | 35,064 |
| 已获取 | North Hub 实时电价 | GridStatus.io | 15 分钟 | 2022-01-01 至 2025-12-31 | 140,256 |
| 已获取并入库 | 北德州六地点天气历史预报 | Open-Meteo | 小时 | 2022-01-01 至 2026-07-01 UTC | 236,592 |
| 已获取 | Henry Hub 天然气现货价格 | FRED | 工作日日度 | 2022-01-03 至 2025-12-31 | 1,043 |

天气及其他已有 raw 数据已写入 `interim/ercot_data.sqlite` 的 raw 层。

## 1. ERCOT North Hub 日前电价

- 来源：GridStatus.io Hosted API
- API 数据集：`ercot_spp_day_ahead_hourly`
- 地点：`HB_NORTH`
- 地点类型：`Trading Hub`
- 市场：`DAY_AHEAD_HOURLY`
- 原始文件：48 个 `csv.gz`
- 唯一记录：35,064 行
- 重复记录：0 行
- 缺失 SPP：0 行
- UTC 区间：`2022-01-01 06:00:00+00:00` 至 `2026-01-01 06:00:00+00:00`
- 本地交付区间：2022-01-01 至 2025-12-31，ERCOT `US/Central`
- SPP 范围：-4.00 至 4,200.00 USD/MWh

字段：

| 字段 | 含义 |
|---|---|
| `interval_start_local` | ERCOT 本地时段开始时间 |
| `interval_start_utc` | UTC 时段开始时间 |
| `interval_end_local` | ERCOT 本地时段结束时间 |
| `interval_end_utc` | UTC 时段结束时间 |
| `location` | 结算点，本数据均为 `HB_NORTH` |
| `location_type` | 地点类型，本数据均为 `Trading Hub` |
| `market` | 市场类型，本数据均为 `DAY_AHEAD_HOURLY` |
| `spp` | Settlement Point Price，USD/MWh |

## 2. ERCOT North Hub 实时电价

- 来源：GridStatus.io Hosted API
- API 数据集：`ercot_spp_real_time_15_min`
- 地点：`HB_NORTH`
- 地点类型：`Trading Hub`
- 市场：`REAL_TIME_15_MIN`
- 原始文件：48 个 `csv.gz`
- 唯一记录：140,256 行
- 重复记录：0 行
- 缺失 SPP：0 行
- UTC 区间：`2022-01-01 06:00:00+00:00` 至 `2026-01-01 06:00:00+00:00`
- 本地交付区间：2022-01-01 至 2025-12-31，ERCOT `US/Central`
- SPP 范围：-251.00 至 5,409.28 USD/MWh

字段与日前电价相同，`market` 为 `REAL_TIME_15_MIN`。后续清洗时需将每小时四个 15 分钟区间聚合为小时实时价格，再计算：

```text
spread = rt_price_hourly - da_price
```

## 3. Texas 天气历史预报

- 来源：Open-Meteo Historical Forecast API
- 频率：小时
- 时区：UTC/GMT
- 北德州六地点覆盖范围：`2022-01-01T00:00` 至 `2026-07-01T23:00Z`
- 北德州每个地点唯一记录：39,432 行
- 北德州六地点合计唯一记录：236,592 行
- 所有天气变量缺失值：0 行

六个北德州地点均已完成采集并写入 SQLite。旧配置的 Austin、Houston、San Antonio 及 Dallas 历史分块继续保留在 raw 层，以保证原始数据可追溯。

当前已保存地点：

| 地点 | API 网格纬度 | API 网格经度 | 原始文件 | 原始行数 | 唯一行数 | 重复行数 |
|---|---:|---:|---:|---:|---:|---:|
| Austin | 30.269146 | -97.753380 | 17 | 35,088 | 35,064 | 24 |
| Dallas | 32.784855 | -96.803590 | 17 | 35,088 | 35,064 | 24 |
| Houston | 29.767237 | -95.354450 | 17 | 35,088 | 35,064 | 24 |
| San Antonio | 29.411030 | -98.485000 | 17 | 35,088 | 35,064 | 24 |

当前默认采集地点：

| 地点 | 请求纬度 | 请求经度 | 新采集文件 | 唯一小时记录 |
|---|---:|---:|---:|---:|
| Dallas | 32.7767 | -96.7970 | 54 | 39,432 |
| Fort Worth | 32.7555 | -97.3308 | 54 | 39,432 |
| Denton | 33.2148 | -97.1331 | 54 | 39,432 |
| McKinney | 33.1972 | -96.6397 | 54 | 39,432 |
| Arlington | 32.7357 | -97.1081 | 54 | 39,432 |
| Wichita Falls | 33.9137 | -98.4934 | 54 | 39,432 |

Dallas 的旧 17 个分块与新 53 个分块在 2022 至 2025 年重叠；旧四地点中还各有 24 行早期单日测试记录。raw 层有意保留这些来源记录，cleaned 层必须以 `location + interval_start_utc` 去重。

天气字段：

| 字段 | 单位 | 含义 |
|---|---|---|
| `time` | UTC | 目标小时 |
| `temperature_2m` | °C | 2 米气温 |
| `relative_humidity_2m` | % | 2 米相对湿度 |
| `wind_speed_10m` | m/s | 10 米风速 |
| `wind_gusts_10m` | m/s | 10 米阵风风速 |
| `cloud_cover` | % | 云量 |
| `shortwave_radiation` | W/m² | 短波辐射 |
| `precipitation` | mm | 降水量 |

注意：这些数据来自 Open-Meteo Historical Forecast 端点。原始响应没有精确的历史发布时间 `issued_at`，不能声称它们是某一日前投标截止时刻实际可见的 forecast vintage。建模报告中应记录这一限制。

## 4. Henry Hub 天然气价格

- 来源：FRED
- Series ID：`DHHNGSP`
- 含义：Henry Hub Natural Gas Spot Price
- 单位：USD/MMBtu
- 原始文件：1 个 `json.gz`
- API 返回记录：1,043 行
- 日期范围：2022-01-03 至 2025-12-31
- 有效数值：998 行
- FRED 缺失标记 `.`：45 行
- 有效价格范围：1.21 至 13.20 USD/MMBtu

字段：

| 字段 | 含义 |
|---|---|
| `date` | 观测日期 |
| `value` | 天然气价格；`.` 表示缺失 |
| `realtime_start` | FRED vintage 实时开始日期 |
| `realtime_end` | FRED vintage 实时结束日期 |

清洗时需将 `.` 转换为 SQL `NULL`。该序列不是每日都有观测，模型小时表连接时需采用明确的 as-of 规则，并避免使用在交易决策时尚未发布的价格。

## 5. 元数据文件

每个 `csv.gz` 或 `json.gz` 都有一个同名的 `.metadata.json`，记录：

- 数据来源与数据集名称
- 不含 API Key 的请求参数
- 采集时间
- 原始文件行数与字段
- 文件大小
- SHA-256 校验值

当前检查结果：所有原始文件都存在对应元数据，全部 SHA-256 校验通过，元数据未包含 API Key。

## 6. 数据库建设状态

- [x] ERCOT Seven-Day Load Forecast 已写入 raw 和 clean 数据库
- [x] raw 数据已写入 `ercot_data.sqlite`
- [x] typed clean 数据已写入 `ercot_analytics.sqlite`
- [x] RT 15 分钟价格已聚合为小时价格
- [x] 所有时间字段统一为规范 UTC
- [x] 六个目标天气地区已去重
- [x] FRED 的 `.` 已转换为 SQL `NULL`
- [x] 已生成 `spread = RT - DA` 和方向标签
- [x] 已生成 DFW 聚合与 Wichita 独立天气特征
- [x] 已按前一日09:55 CT生成 load as-of 特征，但现有历史仅覆盖168小时
- [x] Henry Hub按次一工作日可用并且只向前填充
- [x] 已生成包含时间、天气、gas和load覆盖标记的模型视图
- [ ] 重新获取完整 load forecast vintages 后扩大负荷特征覆盖

## 7. 原始文件位置

```text
01_data_collection_cleaning/raw/
├── gridstatusio/
│   ├── ercot_spp_day_ahead_hourly/
│   └── ercot_spp_real_time_15_min/
├── openmeteo/
│   ├── historical-forecast_Arlington/
│   ├── historical-forecast_Dallas/
│   ├── historical-forecast_Denton/
│   ├── historical-forecast_Fort_Worth/
│   ├── historical-forecast_McKinney/
│   └── historical-forecast_Wichita_Falls/
└── fred/
    └── DHHNGSP/
```

当前 SQLite 数据库路径为：

```text
01_data_collection_cleaning/interim/ercot_data.sqlite
01_data_collection_cleaning/interim/ercot_analytics.sqlite
```
