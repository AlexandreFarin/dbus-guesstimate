import os
import sys
import requests
import pandas as pd
from databricks import sql
from pathlib import Path
from dotenv import load_dotenv


path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

load_dotenv()

def get_dataset(table_name):
    with sql.connect(server_hostname=os.getenv("db_host"),
                    http_path=os.getenv("http_path"),
                    access_token=os.getenv("token")) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM users.alexandre_farin.{table_name}")
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] 
    return pd.DataFrame(result, columns=columns)

def score_model(age, active_users, industry, etl_pct, data_volume, jobs, delta, ml_pct, model_serving, sql_pct, serverless):
    headers = {
       "Authorization": f'Bearer {os.getenv("token")}',
       "Content-Type": "application/json",
    }
    input_data = {
      "dataframe_split": {
        "columns": [
          "pct_eda",
          "pct_etl",
          "pct_edw",
          "pct_jobs",
          "DailyGbProcessCatOrd",
          "DeltaPercent",
          "activeUsers",
          "ServerlessSqlPercent",
          "ModelServing",
          "Industry",
          "customerAgeMonths"
        ],
        "data": [
          [
            ml_pct,
            etl_pct,
            sql_pct,
            jobs,
            data_volume,
            delta,
            active_users,
            serverless,
            model_serving,
            industry,
            age
          ]
        ]
      }
    }
    model_uri=f'https://{os.getenv("db_host")}/serving-endpoints/dus-guesstimate/invocations'
    response = requests.post(headers=headers, url=model_uri, json=input_data, timeout=20)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    
    dbus = response.json()["predictions"][0]
    return int(dbus)