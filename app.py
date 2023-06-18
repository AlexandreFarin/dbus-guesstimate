import os
import requests
import json
import pandas as pd
from databricks import sql
from dash import Dash, dash_table, html, dcc, Output, Input
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

load_dotenv()

data = {
  "dataframe_split": {
    "columns": [
      "pct_eda",
      "pct_etl",
      "pct_edw",
      "pct_jobs",
      "DailyGbProcess",
      "DeltaPercent",
      "activeUsers",
      "ServerlessPercent",
      "Industry",
      "customerAgeMonths"
    ],
    "data": [
      [
        0,
        0,
        0.99,
        0,
        7,
        0.86,
        22,
        0,
        "Professional, Scientific, and Technical Services",
        45
      ]
    ]
  }
}
def get_dataset(table_name):
    with sql.connect(server_hostname=os.getenv("db_host"),
                    http_path=os.getenv("http_path"),
                    access_token=os.getenv("token")) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM users.alexandre_farin.{table_name}")
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(result, columns=columns)
    return df

def score_model(data):
    headers = {
       "Authorization": f'Bearer {os.getenv("token")}',
       "Content-Type": "application/json",
    }
    model_uri=f'https://{os.getenv("db_host")}/serving-endpoints/dus-guesstimate/invocations'
    response = requests.post(headers=headers, url=model_uri, json=data)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()["predictions"][0]

df = get_dataset("customers_last_month")

app = Dash(
   external_stylesheets=[dbc.themes.LUX]
)

app.layout = html.Div(
  [
    dbc.Row(
        [
          dbc.Col(
            html.Img(src='assets/databricks-logo.jpeg',className="img-fluid",
                     style={'height':'30%', 'width':'30%'}),
            width="2", align="center"
          ),
          dbc.Col(
            html.H1(children="DBUs Guesstimate", style={'text-align':'center'}),
            width=8, align="center"
          )
        ]
    ),
    dbc.Row(
        dbc.Col(html.Figure([
            html.Blockquote(
                html.P('The idea that the future is unpredictable is undermined everyday by the ease with which the past is explained.',
                       className="mb-0"),
                className="blockquote"
            ),
            html.Figcaption('Daniel Kahneman',
                            className="blockquote-footer")
        ]
      ),
      width={"size": 4, "offset": 4})
    ),
    dbc.Row(
        [
          dbc.Col(
              html.Div([
                  html.H3("Customer", className="card-header"),
                  html.Div(
                     html.H6("cfrfrtftgt", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/company.svg'),
                           style={'textAlign': 'center'}),
                  html.Div(
                        [
                          dbc.Label("Dropdown", html_for="dropdown"),
                          dcc.Dropdown(df['Industry'].unique(), id="dropdown"
                          ),
                        ],
                        className="mb-3")
              ],className="card mb-3")
          ),

          dbc.Col(
              html.Div([
                  html.H3("Data Preparation", className="card-header"),
                  html.Div(
                     html.H6("cfrfrtftgt", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div([
                    dbc.Label("Email", html_for="example-email"),
                    dbc.Input(type="email", id="example-email", placeholder="Enter email"),
                    dbc.FormText(
                        "Are you on email? You simply have to be these days",
                        color="secondary",
                    ),
                  ], className="mb-3")
              ],className="card mb-3")
          ),

          dbc.Col(html.Div("One of four columns")),

          dbc.Col(html.Div("One of four columns")),
        ]
    ),
        dbc.Row(
        dbc.Col(html.H1(children=score_model(data),
                       style={'text-align':'center'}))
    ),
  ]
)

if __name__=='__main__':
    app.run_server(debug=True)
