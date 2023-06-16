from dotenv import load_dotenv
import os
from databricks.connect import DatabricksSession
from databricks.sdk.core import Config
from dash import Dash, dash_table, html, dcc, Output, Input
import pandas as pd

load_dotenv()
config = Config(
  host = os.getenv("db_host"),
  token = os.getenv("token"),
  cluster_id = os.getenv("cluster_id")
)

spark = DatabricksSession.builder.sdkConfig(config).getOrCreate()

df = spark.read.table("users.alexandre_farin.customers_last_month").toPandas()

app = Dash()

app.layout = html.Div([
    html.H1(children="DBUs guesstimate",className="hello",
            style={'color':'#00361c','text-align':'center'}),

    dcc.Slider(0, 100, 10, value=50),

    dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    
    ])

if __name__=='__main__':
    app.run_server()