import os
import sys
import requests
import pandas as pd
from databricks import sql
from pathlib import Path
from dotenv import load_dotenv
from dash import html
import dash_bootstrap_components as dbc


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

def call_models(cloudType,marketSegment,industryVertical,customerStatus,pct_ml,pct_de,pct_bi,pct_automation,DailyGbProcessCatOrd,DeltaPercent,nUsers,model_serving_bin,customerAgeQuarters,pct_gpu,pct_photon,pct_streaming,DLTPercent,ServerlessSqlPercent):
    headers = {
       "Authorization": f'Bearer {os.getenv("token")}',
       "Content-Type": "application/json",
    }
    input_data = {
      "dataframe_split": {
        "columns": [
          "cloudType",
          "marketSegment",
          "industryVertical",
          "customerStatus",
          "pct_ml",
          "pct_de",
          "pct_bi",
          "pct_automation",
          "DailyGbProcessCatOrd",
          "DeltaPercent",
          "nUsers",
          "model_serving_bin",
          "customerAgeQuarters",
          "pct_gpu",
          "pct_photon",
          "pct_streaming",
          "DLTPercent",
          "ServerlessSqlPercent"
        ],
        "data": [
          [
            cloudType,
            marketSegment,
            industryVertical,
            customerStatus,
            pct_ml,
            pct_de,
            pct_bi,
            pct_automation,
            DailyGbProcessCatOrd,
            DeltaPercent,
            nUsers,
            model_serving_bin,
            customerAgeQuarters,
            pct_gpu,
            pct_photon,
            pct_streaming,
            DLTPercent,
            ServerlessSqlPercent
          ]
        ]
      }
    }
    model_uri=f'https://{os.getenv("db_host")}/serving-endpoints/Guestimate/invocations'
    response = requests.post(headers=headers, url=model_uri, json=input_data, timeout=200)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    
    dollars = response.json()["predictions"][0]

    model_uri=f'https://{os.getenv("db_host")}/serving-endpoints/GuestimateMinimum/invocations'
    response = requests.post(headers=headers, url=model_uri, json=input_data, timeout=200)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    
    dollars_min = response.json()["predictions"][0]

    model_uri=f'https://{os.getenv("db_host")}/serving-endpoints/GuestimateMaximum/invocations'
    response = requests.post(headers=headers, url=model_uri, json=input_data, timeout=200)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    
    dollars_max = response.json()["predictions"][0]

    return (int(dollars),int(dollars_min),int(dollars_max))


def create_salesforce_table(dollars,cloudType,pct_bi,ServerlessSqlPercent,pct_ml,model_serving_bin,pct_automation, pct_de, DLTPercent):
    price = {
        "Jobs Compute": 0.13,
        "Delta Live Table": 0.32,
        "SQL Compute": 0.18,
        "All Purpose Compute": 0.42,
        "Serverless SQL": 0.52,
        "Model Serving": 0.07
    }

    bi_dollars = dollars * pct_bi
    bi_dbus = bi_dollars / (price["SQL Compute"]*(1-ServerlessSqlPercent)+price["Serverless SQL"]*(ServerlessSqlPercent))
    bi_dbus_serverless = bi_dbus * ServerlessSqlPercent
    bi_dbus_classic = bi_dbus * (1-ServerlessSqlPercent)
    if model_serving_bin == 1:
        pct_model_serving = 0.05
    else:
        pct_model_serving = 0
    ml_dollars = dollars * pct_ml
    model_serving_dollars = ml_dollars * pct_model_serving
    model_serving_dbus = model_serving_dollars / price["Model Serving"]
    ml_dbus = ml_dollars * (1-pct_model_serving) / (price["All Purpose Compute"]*(1-pct_automation)+price["Jobs Compute"]*(pct_automation))
    jobs_dbus = ml_dbus * pct_automation
    interactive_dbus = ml_dbus * (1-pct_automation)
    de_dollars = dollars * pct_de
    dlt_dollars = de_dollars * DLTPercent
    dlt_dbus = dlt_dollars / price["Delta Live Table"]
    de_dbus = de_dollars * (1-DLTPercent) / (price["All Purpose Compute"]*(1-pct_automation)+price["Jobs Compute"]*(pct_automation))
    jobs_dbus += de_dbus * pct_automation
    interactive_dbus += de_dbus * (1-pct_automation)
    
    table_header = [
      html.Thead(html.Tr([html.Th("SKU"), html.Th("DBUs")]))
    ]
    row1 = html.Tr([html.Td("Jobs Compute"), html.Td(int((jobs_dbus+50)/100)*100)])
    row2 = html.Tr([html.Td("Delta Live Table"), html.Td(int((dlt_dbus+50)/100)*100)])
    row3 = html.Tr([html.Td("SQL Compute"), html.Td(int((bi_dbus_classic+50)/100)*100)])
    row4 = html.Tr([html.Td("All Purpose Compute"), html.Td(int((interactive_dbus+50)/100)*100)])
    row5 = html.Tr([html.Td("Serverless SQL"), html.Td(int((bi_dbus_serverless+50)/100)*100)])
    row6 = html.Tr([html.Td("Model Serving"), html.Td(int((model_serving_dbus+50)/100)*100)])

    table_body = [html.Tbody([row1, row2, row3, row4, row5, row6])]

    table = dbc.Table(
        # using the same table as in the above example
        table_header + table_body,
        color="light",
        ),
    return table
