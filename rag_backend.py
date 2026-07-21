import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent  # <--- 修改了这里
from langchain_community.agent_toolkits import SQLDatabaseToolkit # <--- 建议这里也改一下

# 将 api_key 作为参数传入
def get_sql_agent_response(user_question, api_key, db_path="sqlite:///ercot_data.db"):
    db = SQLDatabase.from_uri(db_path)
    
    # 动态使用传入的 api_key
    llm = ChatOpenAI(
        model="deepseek-chat", 
        api_key=api_key, 
        base_url="https://api.deepseek.com", 
        temperature=0
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    system_prefix = """
    You are an expert quantitative power trader and a highly skilled SQL assistant for the ERCOT market.
    You query a SQLite table named 'model_wide_hourly_2024_2026'.
    
    Business Rules based on data dictionary:
    1. Target metric: 'spread_usd_per_mwh' is the price spread (RT - DA). A positive value means Real-Time price is higher.
    2. Time: Use 'ercot_local_hour' (values 0-23) to group or filter by hours of the day. Do not parse raw UTC strings for hour grouping.
    3. Weather Events: For freezing weather, check if 'freezing_city_count' > 0. For heatwaves, use 'extreme_heat_city_count'.
    4. Fundamentals: Gas price is 'gas_price_usd_per_mmbtu'. Load proxy is 'load_hb_north_proxy_mw'.
    
    Always write strictly valid SQLite queries. Output natural language explanations along with the data findings.
    """

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        prefix=system_prefix 
    )
    
    response = agent_executor.invoke({"input": user_question})
    return response['output']