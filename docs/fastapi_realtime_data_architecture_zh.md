# FastAPI 实时数据接入与预测服务方案

## 1. 建设目标

后期系统使用 FastAPI 对外提供 ERCOT 日前预测、数据状态和历史结果接口。系统在日前市场截止前接入天气、天然气、负荷、风电和光伏预测，生成次日逐小时价差预测和交易建议；交付后再获取真实日前与实时价格，用于结算、回测和模型更新。

FastAPI 主要承担服务层职责。外部数据应由定时任务提前抓取并写入数据库，而不是等 Dashboard 发起请求时临时调用全部上游 API。

## 2. 总体架构

```text
GridStatus.io ─┐
Open-Meteo ────┤
FRED ──────────┤ -> 定时采集 -> Raw/Online DB -> 特征快照 -> 模型推理
ERCOT 价格 ────┘                                      |
                                                       v
Dashboard <- FastAPI REST/SSE <- 预测、状态、实际结算结果
```

系统分成两条数据流：

1. **日前预测流**：截止时间前冻结天气、天然气、负荷、风电和光伏预测，生成次日预测。
2. **交付后结算流**：获取真实 DA、RT 价格，计算 `RT - DA`，用于回测和后续训练。

真实 DA 和 RT 价格不能进入当次日前模型输入。

## 3. 上游数据接口

### 3.1 ERCOT：GridStatus.io

继续使用 `gridstatusio.GridStatusClient`：

```python
from gridstatusio import GridStatusClient

client = GridStatusClient(
    api_key=GRIDSTATUS_API_KEY,
    return_format="pandas",
)
```

| 数据 | GridStatus.io dataset | 用途 |
|---|---|---|
| 日前价格 | `ercot_spp_day_ahead_hourly` | 交付后标签与回测 |
| 实时价格 | `ercot_spp_real_time_15_min` | 聚合小时 RT 标签 |
| 负荷预测 | `ercot_load_forecast` | 日前模型输入 |
| 风电预测 | `ercot_wind_actual_and_forecast_hourly` | 日前模型输入 |
| 光伏预测 | `ercot_solar_actual_and_forecast_hourly` | 日前模型输入 |

示例：

```python
frame = client.get_dataset(
    dataset="ercot_load_forecast",
    start="2026-07-23",
    end="2026-07-24",
    publish_time_end="2026-07-22T09:55:00-05:00",
    timezone="US/Central",
    limit=5000,
    filter_value="",
    verbose=False,
)
```

数据返回后仍需在本地根据 `publish_time_utc` 筛选10:00 CT 前可用的最新版本。

当前模型同时使用负荷、风电和光伏预测。线上不能只接入 load，否则输入 schema 将与训练时不一致。

### 3.2 天气：Open-Meteo

为了与当前 hybrid 训练数据保持一致，建议使用：

```text
https://previous-runs-api.open-meteo.com/v1/forecast
```

六个地区：

- Dallas
- Fort Worth
- Denton
- McKinney
- Arlington
- Wichita Falls

请求变量：

```text
temperature_2m
relative_humidity_2m
wind_speed_10m
wind_gusts_10m
cloud_cover
shortwave_radiation
precipitation
```

同时请求每个变量的 `previous_day1` 和 `previous_day2` 版本，并沿用当前规则：

```text
ERCOT 本地 00:00-08:00 -> previous_day1
ERCOT 本地 09:00-23:00 -> previous_day2
```

不要直接改为普通 Forecast API 的最新预报，否则天气预测提前量会与训练数据不同。

### 3.3 天然气：FRED

接口：

```text
GET https://api.stlouisfed.org/fred/series/observations
```

主要参数：

```text
series_id=DHHNGSP
api_key=...
file_type=json
observation_start=...
observation_end=...
sort_order=asc
```

FRED Henry Hub 是日频现货数据，不是实时天然气行情。建议每天更新一至两次，并沿用当前数据可用性规则：

```text
观测日后一个工作日才视为可用
只能向前填充
```

## 4. FastAPI 对外接口

### 4.1 日前预测

```http
GET /v1/forecasts/day-ahead/{delivery_date}
```

示例响应：

```json
{
  "delivery_date": "2026-07-23",
  "as_of_utc": "2026-07-22T14:50:00Z",
  "model_version": "lightgbm-v2",
  "feature_schema_version": 6,
  "hours": [
    {
      "delivery_hour_utc": "2026-07-23T05:00:00Z",
      "local_hour": 0,
      "predicted_spread": -12.4,
      "probability_rt_above_da": 0.28,
      "signal": "short_spread",
      "confidence": 0.72
    }
  ]
}
```

### 4.2 Dashboard 查询接口

```http
GET /v1/features/day-ahead/{delivery_date}
GET /v1/data-status/{delivery_date}
GET /v1/prices/latest?location=HB_NORTH
GET /v1/predictions/history?start=...&end=...
GET /v1/model/info
GET /health/live
GET /health/ready
```

Dashboard 如需自动更新，可增加 SSE：

```http
GET /v1/stream/predictions
```

### 4.3 内部任务接口

以下接口只供管理员或调度服务使用，并需要认证：

```http
POST /internal/jobs/collect/pre-dam/{delivery_date}
POST /internal/jobs/build-features/{delivery_date}
POST /internal/jobs/predict/{delivery_date}
POST /internal/jobs/settle/{delivery_date}
POST /internal/jobs/retry/{job_id}
```

生产环境中更建议调度器直接调用 Python 服务函数，而不是通过公开 HTTP 触发采集。

## 5. 推荐调度流程

所有调度时间使用 `America/Chicago`：

```text
08:30  第一次抓取天气、天然气、负荷、风电和光伏
09:35  第二次刷新
09:50  冻结数据快照
09:52  构建特征并运行模型
09:55  向 Dashboard 发布最终建议
交付后 获取 DA/RT 结果并形成标签
```

每次预测应保存：

- `as_of_utc`；
- 各数据源的 `issue_time` 或 `run_time`；
- 决策截止时间；
- 模型版本；
- 特征 schema 版本；
- 完整输入快照；
- 每小时预测和交易信号。

## 6. 数据库设计

建议增加以下在线表：

```text
weather_forecast_vintages
ercot_forecast_vintages
gas_observations
price_actuals
feature_snapshots
model_predictions
collection_jobs
```

每个 forecast 表必须保留：

- 目标交付小时；
- 发布时间或模型运行时间；
- 实际采集时间；
- 数据源和产品 ID；
- 原始响应文件位置；
- 数据质量状态。

演示阶段可以继续使用 SQLite，并启用：

```sql
PRAGMA journal_mode=WAL;
```

SQLite 模式下应只保留一个写入进程，不要让多个 Uvicorn worker 同时写库。正式部署建议使用 PostgreSQL；Redis 只用于缓存和任务队列。

## 7. 推荐代码结构

```text
04_frontend_dashboard/
05_realtime_service/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── forecasts.py
│   │   ├── market.py
│   │   └── health.py
│   ├── connectors/
│   │   ├── gridstatusio.py
│   │   ├── openmeteo.py
│   │   └── fred.py
│   ├── services/
│   │   ├── feature_service.py
│   │   ├── inference_service.py
│   │   └── settlement_service.py
│   ├── repositories/
│   ├── schemas/
│   └── jobs/
├── tests/
├── Dockerfile
└── requirements.txt
```

外部 HTTP 接口可使用共享的 `httpx.AsyncClient`。同步的 GridStatus.io SDK 应放入线程池运行，避免阻塞 FastAPI 事件循环。

## 8. 数据质量和安全规则

- FastAPI 查询不能临时同步抓取全部上游数据。
- 每次预测必须基于冻结的 `as-of` 特征快照。
- DA、RT 真实价格只能用于标签、监控和结算。
- 线上特征名称、单位和聚合方式必须与训练宽表一致。
- 使用明确的特征白名单和 schema hash。
- 任一必要数据缺失时返回 `not_ready`，不能静默填0。
- ERCOT 本地交付日可能有23、24或25个小时。
- 所有连接键使用 UTC，本地时间只作特征和展示。
- API Key 只存放在环境变量或密钥管理系统中。
- 为上游请求设置 timeout、retry、指数退避和 rate limit。
- 保存原始响应和 source timestamp，保证预测可复现。
- 内部任务接口必须鉴权，并记录调用审计日志。

## 9. 推荐实施顺序

1. 建立 `05_realtime_service` FastAPI 工作区和健康检查接口。
2. 将现有三个 collector 封装成可复用 connector。
3. 建立在线 vintage 表、特征快照表和预测表。
4. 实现日前定时采集和 `data-status` 完整性检查。
5. 固化模型特征白名单和模型加载流程。
6. 实现日前预测接口。
7. 实现交付后价格结算和标签回填。
8. 最后连接 Dashboard 和 SSE 更新。

正式上线前，应使用历史日期执行一次完整 replay，确认在线特征构建结果与离线宽表在相同时间点、单位和字段定义下保持一致。
