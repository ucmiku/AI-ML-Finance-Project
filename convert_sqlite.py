import pandas as pd
import sqlite3
import os

# 1. 配置你的文件路径
csv_file_path = "model_wide_hourly_2024_2026.csv" # 替换为你的 CSV 实际路径
db_directory = "01_data_collection_cleaning/interim"
db_file_path = f"{db_directory}/ercot_analytics.sqlite"
table_name = "model_wide_hourly_2024_2026"

# 2. 确保目标文件夹存在（如果没有，程序会自动创建）
os.makedirs(db_directory, exist_ok=True)

print(f"⏳ 正在读取 CSV 文件: {csv_file_path} ...")
# 3. 读取 CSV 文件
try:
    df = pd.read_csv(csv_file_path)
    print(f"✅ 成功读取数据，共 {len(df)} 行，{len(df.columns)} 列。")
except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {csv_file_path}，请检查路径是否正确。")
    exit()

# 4. 连接到 SQLite 数据库（如果文件不存在，会自动创建这个文件）
print(f"⏳ 正在将数据写入 SQLite 数据库: {db_file_path} ...")
conn = sqlite3.connect(db_file_path)

# 5. 将数据框 (DataFrame) 转换为 SQLite 表
# if_exists="replace" 表示如果表已经存在，就覆盖它；index=False 表示不要把 pandas 的行号写进数据库
df.to_sql(table_name, conn, if_exists="replace", index=False)

# 6. 关闭连接
conn.close()

print(f"🎉 转换完成！你的数据库文件已准备好：{db_file_path}")