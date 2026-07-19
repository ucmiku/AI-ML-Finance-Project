# Clean 数据库 Schema v3

生成日期：2026-07-18  
数据库：`01_data_collection_cleaning/interim/ercot_analytics.sqlite`  
上游 raw 数据库：`01_data_collection_cleaning/interim/ercot_data.sqlite`

## 设计原则

- raw 数据库保持不可变，clean 数据库可由脚本完整重建。
- 所有带时间的字段统一为 UTC，格式为 `YYYY-MM-DDTHH:MM:SSZ`。
- FRED 只提供日期精度，因此保留 `YYYY-MM-DD`，不人为添加午夜时间。
- 每张 clean 表使用明确的业务主键去重，并保留 raw 的 `source_file_id` 和 `source_record_id`。
- 天气仅使用 Dallas、Fort Worth、Denton、McKinney、Arlington、Wichita Falls。
- 当前不建设风电和光伏 clean 表。

重建命令：

```powershell
python 01_data_collection_cleaning/scripts/build_clean.py
```

## 表清单

| 表 | 主键 | 行数 | 用途 |
|---|---|---:|---|
| `clean_build_info` | `key` | 13 | 构建时间、版本和关键可用性假设 |
| `clean_da_price_hourly` | `delivery_hour_utc, location` | 39,407 | North Hub 日前小时价格 |
| `clean_rt_price_15min` | `interval_start_utc, location` | 157,626 | North Hub 实时 15 分钟价格 |
| `clean_rt_price_hourly` | `delivery_hour_utc, location` | 39,407 | 实时价格小时平均值 |
| `clean_price_hourly` | `delivery_hour_utc, location` | 39,407 | 日前、实时、价差及方向标签 |
| `clean_weather_hourly` | `city, target_hour_utc` | 236,592 | 六城市小时天气 |
| `clean_gas_daily` | `observation_date, vintage_start_date` | 2,215 | Henry Hub 日价格及 FRED vintage |
| `clean_load_forecast` | `target_time_utc, publish_time_utc` | 472,884 | ERCOT 负荷预测全部版本 |
| `feature_weather_hourly` | `target_hour_utc` | 39,432 | DFW 聚合、Wichita 差值和极端天气特征 |
| `feature_time_hourly` | `delivery_hour_utc` | 39,407 | UTC主键和ERCOT本地时间特征 |
| `feature_load_da_hourly` | `delivery_hour_utc` | 168 | 前一日09:55 CT as-of负荷小时特征 |
| `feature_gas_da_daily` | `decision_date_local` | 1,642 | 次一工作日可用、仅向前填充的天然气特征 |
| `model_split_assignments` | `delivery_hour_utc` | 39,405 | 时间顺序 train/validation/test 划分 |
| `quality_check_results` | `check_name` | 12 | 数据库内 QC 结果 |

只读视图：

| 视图 | 行数 | 用途 |
|---|---:|---|
| `vw_complete_price_labels` | 39,405 | 排除不完整 RT 小时的价格标签 |
| `vw_model_price_weather_hourly` | 39,405 | 已对齐的完整价格标签和天气特征 |
| `vw_model_dataset_hourly` | 39,405 | 含时间、天气、gas和load覆盖标记的模型视图 |

## 价格表

`clean_da_price_hourly` 和 `clean_rt_price_15min` 对重复 raw 记录采用以下优先级：

```text
collected_at_utc DESC
imported_at_utc DESC
record_id DESC
```

`clean_rt_price_hourly` 使用同一 UTC 小时内的 15 分钟 SPP 算术平均值，并记录：

- `interval_count`
- `is_complete_hour`
- 小时最小、最大和平均实时价格

`clean_price_hourly` 定义：

```text
spread_usd_per_mwh = rt_price_usd_per_mwh - da_price_usd_per_mwh
rt_above_da = 1 if spread > 0 else 0
spread_sign = -1, 0, or 1
```

共有 39,405 个完整标签。以下两个小时各缺少一个实时 15 分钟区间，`is_label_complete = 0`：

- `2026-01-08T21:00:00Z`
- `2026-06-04T20:00:00Z`

训练和回测默认应筛选 `is_label_complete = 1`。

## 天气表

每个城市有 39,432 个唯一小时，覆盖：

```text
2022-01-01T00:00:00Z through 2026-07-01T23:00:00Z
```

Dallas 新旧 raw 文件的重叠记录已按 `city + target_hour_utc` 去重。Austin、Houston 和 San Antonio 未进入 clean 表。

天气字段包括温度、相对湿度、风速、阵风、云量、短波辐射和降水。Open-Meteo Historical Forecast 没有历史 `publish_time`，因此每行标记：

```text
availability_assumption = historical_forecast_without_publish_time
```

后续模型报告必须披露这个数据可用性限制。

## 天气特征表

`feature_weather_hourly` 将 Dallas、Fort Worth、Denton、McKinney 和 Arlington 聚合为 DFW 城市群，同时单独保留 Wichita Falls。字段包括：

- DFW 五城的均值、温度范围、最大风速/阵风和最大降水
- Wichita Falls 的独立值
- Wichita Falls 与 DFW 均值的差值
- 六城最低/最高温度、最大阵风和最大降水
- 冰冻、高温、强风和降雨城市数

极端阈值定义为：温度不高于 0°C、温度不低于 35°C、阵风不低于 15 m/s、降水大于 0.1 mm。所有 39,432 个天气小时都包含完整六城数据。

## 天然气表

`clean_gas_daily` 保留 FRED 响应中的 `realtime_start` 和 `realtime_end`，分别写为 `vintage_start_date` 和 `vintage_end_date`。当前两个版本实际对应 2026-07-16 和 2026-07-17 两次采集快照，并不表示每个历史价格的原始发布日期。`.` 或 JSON `null` 被转换为 SQL `NULL`，并设置 `is_missing = 1`。

当前有 1,172 个观测日期、2,215 个快照版本记录和 97 个缺失值。模型特征使用最新快照中的观测值，并假设观测日期的次一工作日才可用；缺失或尚未可用的值只向前填充，绝不使用未来值。

## 负荷预测表

`clean_load_forecast` 保留全部预测版本。构建日前特征时必须满足：

```text
publish_time_utc <= decision_time_utc
```

再对每个 `target_time_utc` 选择截止时刻前最新的 `publish_time_utc`。当前表不包含风电和光伏预测。

`feature_load_da_hourly` 使用前一日 `09:55 America/Chicago` 作为截止时刻，先选择最新5分钟预测，再聚合为小时。当前原始负荷历史并非完整 forecast vintage，因此只有168个小时通过 as-of筛选，模型训练时不应把缺失的load当作已观测值。

## 时间特征和数据划分

`delivery_hour_utc` 是所有连接的主键；`feature_time_hourly` 额外保留ERCOT本地日期、当地小时、星期、月份、周末和夏令时标记。决策时间规则为：

```text
delivery local date - 1 day at 09:55 America/Chicago
```

39,405个完整价格-天气小时按UTC时间顺序划分为70% train、15% validation和15% test，不使用随机划分。

## QC 查询

数据库内置 7 个 `PASS` 和 5 个 `WARN`。警告包括97个FRED缺失值、2个不完整RT小时、gas前向填充初始空值和负荷as-of覆盖不足。警告记录不会被静默填补。

```sql
SELECT *
FROM quality_check_results
ORDER BY status, check_name;

SELECT city, COUNT(*)
FROM clean_weather_hourly
GROUP BY city ORDER BY city;

SELECT is_complete_hour, COUNT(*)
FROM clean_rt_price_hourly
GROUP BY is_complete_hour;

SELECT is_label_complete, COUNT(*)
FROM clean_price_hourly
GROUP BY is_label_complete;
```
