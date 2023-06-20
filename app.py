import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Output, Input, State
from utils import get_dataset, score_model

df = get_dataset("customers_last_month")

DBC_CSS = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(
   external_stylesheets=[dbc.themes.LUX, DBC_CSS]
)

app.layout = html.Div(
  [
    dbc.Row(
        [
          html.H4("Warning!", className="alert-heading"),
          html.P("Work in progress, use it with a grain of salt.", className="mb-0") 
        ],
        className="alert alert-dismissible alert-warning"
    ),
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
      width={"size": 6, "offset": 3})
    ),
    dbc.Row(
      dbc.Button("Predict", className="btn btn-danger", id="prediction-button", n_clicks=0, disabled=False),
      className="d-grid gap-2 col-6 mx-auto"
    ),
    html.Br(),  
    dbc.Row(
       html.H3(style={'text-align':'center'},id="prediction-output"),
       className="d-grid gap-2 col-4 mx-auto"
    ),
    dbc.Row(
        [
          dbc.Col(
              html.Div([
                  html.H4("Customer", className="card-header"),
                  html.Div(
                     html.H6("Tell me a bit more about the customer", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/company.svg'),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                      html.Div(
                        [
                          html.P("Uses Databricks since . months"),
                          dbc.Input(type="number", min=0, max=200, step=1, value=df['customerAgeMonths'].median(), id="age", className="form-control"),
                        ],
                        className="mb-3"),
                      html.Div(
                        [
                          html.P("Number of Databricks users (all personas)"),
                          dbc.Input(type="number", min=0, max=1000, step=1, value=df['activeUsers'].median(), id="active_users", className="form-control"),
                        ],
                        className="mb-3"),
                      html.Div(
                        [
                          html.P("Industry"),
                          dbc.Select(
                          id="industry", value=df["Industry"].mode()[0],
                          options=[{"value":v, "label":v} for v in df['Industry'].unique()]
                        ),
                        ],
                        className="mb-3"
                      )
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),

          dbc.Col(
              html.Div([
                  html.H4("Data Engineering", className="card-header"),
                  html.Div(
                     html.H6("Bronze, Silver and Gold my friend ", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/data-eng.svg'),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                     html.Div(
                        [
                          html.P("ETL usage in %"),
                          dbc.Input(type="number", min=0, max=100, step=1, value=70, id="etl_pct", className="form-control", invalid=False),
                        ],
                        className="mb-3"),
                      html.Div(
                        [
                          html.P("Data volume processed daily"),
                          dbc.Select(
                          id="data_volume",
                          value=2,
                          options=[
                              {"value": 1, "label": "Less than 10 Giga"},
                              {"value": 2, "label": "10 - 100 Giga"},
                              {"value": 3, "label": "100 - 500 Giga"},
                              {"value": 4, "label": "500 Giga - 1 Tera"},
                              {"value": 5, "label": "1 - 10 Tera"},
                              {"value": 6, "label": "More than 10 Tera"},
                          ]
                        ),
                        ],
                        className="mb-3"
                      ),
                      html.Div(
                        [
                          html.P("Type of workloads (interactive - automated)"),
                          dcc.Slider(id="jobs", min=0, max=100, step=10, value=round(df["pct_jobs"].median(),1) * 100, 
                                     className="dbc",marks={
                                          0: {'label': 'Inter.'},
                                          100: {'label': 'Auto.'}
                                          },
                                    ), 
                        ],
                      className="mb-3"),
                      html.Div(
                        [
                          html.P("Delta usage"),
                          dcc.Slider(id="delta", min=0, max=100, step=10, value=round(df["DeltaPercent"].median(),1) * 100, 
                                     className="dbc", marks={
                                          0: {'label': '0%'},
                                          100: {'label': '100%'}
                                          }
                                     ),
                        ],
                      className="mb-3")
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),

          dbc.Col(
              html.Div([
                  html.H4("Data Science / ML", className="card-header"),
                  html.Div(
                     html.H6("pip install sklearn", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/data-science.svg'),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                     html.Div(
                        [
                          html.P("ML usage in %"),
                          dbc.Input(type="number", min=0, max=100, step=1, value=10, id="ml_pct", className="form-control", invalid=False),
                        ],
                        className="mb-3"),
                    html.Div(
                        [
                          html.P("Uses Model Serving"),
                          dbc.RadioItems(
                              options=[
                                  {"label": "Yes", "value": 1},
                                  {"label": "No", "value": 0},
                              ],
                              value=0, id="model_serving"
                          ),
                        ],
                      className="mb-3")
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),

          dbc.Col(
              html.Div([
                  html.H4("Data Analysis", className="card-header"),
                  html.Div(
                     html.H6("SELECT * FROM MARKET", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/data-analysis.svg'),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                     html.Div(
                        [
                          html.P("SQL usage in %"),
                          dbc.Input(type="number", min=0, max=100, step=1, value=20, id="sql_pct", className="form-control", invalid=False),
                        ],
                        className="mb-3"),
                    html.Div(
                        [
                          html.P("Serverless usage"),
                          dcc.Slider(id="serverless", min=0, max=100, step=10 , value=round(df["ServerlessSqlPercent"].mean(),1) * 100, 
                                     className="dbc", marks={
                                          0: {'label': '0%'},
                                          100: {'label': '100%'}
                                          }
                                    ),
                        ],
                      className="mb-3")
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),
        ],
        style = {'margin-left':'10px', 'margin-top':'7px', 'margin-right':'10px'}
    ),
  ]
)

@app.callback(
      Output('prediction-output', 'children'),
      [
        Input('prediction-button', 'n_clicks'),
        State('age', 'value'),
        State('active_users', 'value'),
        State('industry', 'value'),
        State('etl_pct', 'value'),
        State('data_volume', 'value'),
        State('jobs', 'value'),
        State('delta', 'value'),
        State('ml_pct', 'value'),
        State('model_serving', 'value'),
        State('sql_pct', 'value'),
        State('serverless', 'value')
      ]
    )
def predict(n_clicks, age, active_users, industry, etl_pct, data_volume, jobs, delta, ml_pct, model_serving, sql_pct, serverless):
    if n_clicks == 0:
        return ""
    else:
        dbus = score_model(age, active_users, industry, etl_pct/100, data_volume, jobs/100, delta/100, ml_pct/100, model_serving, sql_pct/100, serverless/100)
        return f"{max(200, round(dbus/10)*10)} DBUs / month"

@app.callback(
      Output('etl_pct', 'invalid'),
      Output('ml_pct', 'invalid'),
      Output('sql_pct', 'invalid'),
      Output('prediction-button', 'disabled'),
      [
        Input('etl_pct', 'value'),
        Input('ml_pct', 'value'),
        Input('sql_pct', 'value')
      ]
    )
def sum_is_100(etl_pct, ml_pct, sql_pct):
    if etl_pct + ml_pct + sql_pct == 100:
        return False, False, False, False
    else:
        return True, True, True, True

if __name__=='__main__':
    app.run_server(debug=True)
