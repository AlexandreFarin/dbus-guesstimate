import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Output, Input, State
from utils import get_dataset, score_model

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
      width={"size": 6, "offset": 3})
    ),
    dbc.Row(
      dbc.Button("Predict", className="btn btn-danger", id="prediction-button", n_clicks=0, disabled=False),
      className="d-grid gap-2 col-4 mx-auto"
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
                  html.H3("Customer", className="card-header"),
                  html.Div(
                     html.H6("cfrfrtftgt", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/company.svg'),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                      html.Div(
                        [
                          html.P("Customer seniority (in months)"),
                          dbc.Input(type="number", min=0, max=200, step=1, value=df['customerAgeMonths'].median(), id="age", className="form-control"),
                        ],
                        className="mb-3"),
                      html.Div(
                        [
                          html.P("Number of active users"),
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
                  html.H3("Data Engineering", className="card-header"),
                  html.Div(
                     html.H6("cfrfrtftgt", className="card-subtitle text-muted"),
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
                          value=df["DailyGbProcessCat"].mode()[0],
                          options=[
                              {"value": "0-10 G", "label": "Less than 10 Giga"},
                              {"value": "10-100 G", "label": "10 - 100 Giga"},
                              {"value": "100-500 G", "label": "100 - 500 Giga"},
                              {"value": "500-1000 G", "label": "500 Giga - 1 Tera"},
                              {"value": "1-10 T", "label": "1 - 10 Tera"},
                              {"value": "+10 T", "label": "More than 10 Tera"},
                          ]
                        ),
                        ],
                        className="mb-3"
                      ),
                      html.Div(
                        [
                          html.P("Percentage jobs"),
                          dcc.Slider(id="jobs", min=0, max=100, step=10, value=round(df["pct_jobs"].median(),1) * 100),
                        ],
                      className="mb-3"),
                      html.Div(
                        [
                          html.P("Delta usage"),
                          dcc.Slider(id="delta", min=0, max=100, step=10, value=round(df["DeltaPercent"].median(),1) * 100),
                        ],
                      className="mb-3")
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),

          dbc.Col(
              html.Div([
                  html.H3("Data Science / ML", className="card-header"),
                  html.Div(
                     html.H6("cfrfrtftgt", className="card-subtitle text-muted"),
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
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),

          dbc.Col(
              html.Div([
                  html.H3("Data Analysis", className="card-header"),
                  html.Div(
                     html.H6("cfrfrtftgt", className="card-subtitle text-muted"),
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
                          dcc.Slider(id="serverless", min=0, max=100, step=10 , value=round(df["ServerlessPercent"].median(),1) * 100),
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
        State('sql_pct', 'value'),
        State('serverless', 'value')
      ]
    )
def predict(n_clicks, age, active_users, industry, etl_pct, data_volume, jobs, delta, ml_pct, sql_pct, serverless):
    if n_clicks == 0:
        return ""
    else:
        dbus = score_model(age, active_users, industry, etl_pct/100, data_volume, jobs/100, delta/100, ml_pct/100, sql_pct/100, serverless/100)
        return f"{round(dbus/10)*10} DBUs / month"

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
