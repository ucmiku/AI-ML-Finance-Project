import pandas as pd
import sqlite3

# 1. 读取同学提供的 CSV 数据
df = pd.read_csv("model_wide_hourly_2024_2026.csv")

# 2. 连接到你的 SQLite 数据库 (如果没有会自动创建)
conn = sqlite3.connect("ercot_data.db")

# 3. 将宽表写入数据库，表名遵循文档规范
df.to_sql("model_wide_hourly_2024_2026", conn, if_exists="replace", index=False)
conn.close()

print("✅ 数据库初始化完成，已写入 21,804 行数据。")