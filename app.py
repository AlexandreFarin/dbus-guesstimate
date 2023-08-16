import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Output, Input, State, dash_table
from utils import get_dataset, call_models

df = get_dataset("customers_last_month")
df_dollars = get_dataset("workspaces_last_month")

DBC_CSS = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(
   external_stylesheets=[dbc.themes.LUX, DBC_CSS]
)
app.config.suppress_callback_exceptions=True
app.layout = html.Div(
  [
    dbc.Row(
        [
          html.H4("Warning!", className="alert-heading"),
          html.P("Work in progress, use it with a grain of salt. Feedbacks are welcome (alexandre.farin@databricks.com)."
                 , className="mb-0") 
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
            html.H1(children="Use Case Guesstimate", style={'text-align':'center'}),
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
    dbc.Row(id="predict-section", className="d-grid gap-2 col-4 mx-auto"),
    dbc.Row(
        [
          dbc.Col(
              html.Div([
                  html.H4("Customer", className="card-header"),
                  html.Div(
                     html.H6("The one we are obessed about", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/company.svg',style={'height':'40%', 'width':'40%'}),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                      html.Div(
                        [
                          html.P("Industry"),
                          dbc.Select(
                          id="industryVertical", value=df_dollars["industryVertical"].mode()[0],
                          options=[{"value":v, "label":v} for v in df_dollars['industryVertical'].unique()]
                        ),
                        ],
                        className="mb-3"
                      ),
                      html.Div(
                        [
                          html.P("Status"),
                          dbc.Select(
                          id="customerStatus", value=df_dollars["customerStatus"].mode()[0],
                          options=[{"value":v, "label":v} for v in df_dollars['customerStatus'].unique()]
                        ),
                        ],
                        className="mb-3"
                      ),
                      html.Div(
                        [
                          html.P("Market Segment"),
                          dbc.Select(
                          id="marketSegment", value=df_dollars["marketSegment"].mode()[0],
                          options=[{"value":v, "label":v} for v in df_dollars['marketSegment'].unique()]
                        ),
                        ],
                        className="mb-3"
                      ),
                      html.Div(
                        [
                          html.P("Uses Databricks since . quarters"),
                          dbc.Input(type="number", min=0, max=df_dollars["customerAgeQuarters"].max(), step=1, value=df_dollars["customerAgeQuarters"].median(), id="customerAgeQuarters", className="form-control"),
                        ],
                        className="mb-3")
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),
          dbc.Col(
              html.Div([
                  html.H4("Use Case", className="card-header"),
                  html.Div(
                     html.H6("The thing on Salesforce", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/use-case.svg',style={'height':'40%', 'width':'40%'}),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                      html.Div(
                        [
                          html.P("Number of users", id="user_title"),
                          dbc.Tooltip(
                              "This include all users who write code: data engineers + data scientists + data analysts",
                              target="user_title",placement="top"
                          ),
                          dbc.Input(type="number", min=0, max=df_dollars["nUsers"].max(), step=1, value=df_dollars["nUsers"].median(), id="nUsers", className="form-control"),
                        ],
                        className="mb-3"),
                      html.Div(
                        [
                          html.P("Maturity", id="maturity_title"),
                          dbc.Tooltip(
                              "If the use case is in an early stage, a lot of development on an interactive cluster, nothing run on a regular basis then move the knob towards POC.",
                              target="maturity_title",placement="top"
                          ),
                          dcc.Slider(id="pct_automation", min=0, max=100, step=10, value=round(df_dollars["pct_automation"].median(),1) * 100, 
                                     className="dbc",marks={
                                          0: {'label': 'POC'},
                                          100: {'label': 'PROD'}
                                          },
                                    ), 
                        ],
                      className="mb-3"),
                      html.Div(
                        [
                          html.P("Data volume processed daily"),
                          dbc.Select(
                          id="DailyGbProcessCatOrd",
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
                          html.P("Delta usage"),
                          dcc.Slider(id="DeltaPercent", min=0, max=100, step=10, value=round(df_dollars["DeltaPercent"].median(),1) * 100, 
                                     className="dbc", marks={
                                          0: {'label': '0%'},
                                          100: {'label': '100%'}
                                          }
                                     ),
                        ],
                      className="mb-3"),
                      html.Div(
                        [
                          html.P("Cloud Provider"),
                          dbc.Select(
                          id="cloudType", value=df_dollars["cloudType"].mode()[0],
                          options=[{"value":v, "label":v} for v in df_dollars['cloudType'].unique()]
                        ),
                        ],
                        className="mb-3"
                      ),
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),
          dbc.Col(
              html.Div([
                  html.H4("Data Engineering", className="card-header"),
                  html.Div(
                     html.H6("Bronze, Silver and Gold", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/data-eng.svg',style={'height':'40%', 'width':'40%'}),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                      html.Div(
                        [
                          html.P("ETL usage in %"),
                          dbc.Input(type="number", min=0, max=100, step=1, value=70, id="pct_de", className="form-control", invalid=False),
                        ],
                        className="mb-3"),
                      html.Div(
                        [
                          html.P("Photon usage in %"),
                          dcc.Slider(id="pct_photon", min=0, max=100, step=10 , value=round(df_dollars["pct_photon"].mean(),1) * 100, 
                                     className="dbc", marks={
                                          0: {'label': '0%'},
                                          100: {'label': '100%'}
                                          }
                                    ),
                        ],
                      className="mb-3"),
                      html.Div(
                        [
                          html.P("Delta Live Table usage in %"),
                          dcc.Slider(id="DLTPercent", min=0, max=100, step=10 , value=round(df_dollars["DLTPercent"].mean(),1) * 100, 
                                     className="dbc", marks={
                                          0: {'label': '0%'},
                                          100: {'label': '100%'}
                                          }
                                    ),
                        ],
                      className="mb-3"),
                      html.Div(
                        [
                          html.P("Streaming usage in %"),
                          dcc.Slider(id="pct_streaming", min=0, max=100, step=10 , value=round(df_dollars["pct_streaming"].median(),1) * 100, 
                                     className="dbc", marks={
                                          0: {'label': '0%'},
                                          100: {'label': '100%'}
                                          }
                                    ),
                        ],
                      className="mb-3"),
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),
          dbc.Col(
              html.Div([
                  html.H4("Data Science / ML", className="card-header"),
                  html.Div(
                     html.H6("pip install sklearn, dolly, llm", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/data-science.svg',style={'height':'40%', 'width':'40%'}),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                    html.Div(
                        [
                          html.P("ML usage in %"),
                          dbc.Input(type="number", min=0, max=100, step=1, value=10, id="pct_ml", className="form-control", invalid=False),
                        ],
                        className="mb-3"),
                    html.Div(
                        [
                          html.P("Type of compute"),
                          dcc.Slider(id="pct_gpu", min=0, max=100, step=10, value=round(df_dollars["pct_gpu"].mean(),1) * 100, 
                                     className="dbc", marks={
                                          0: {'label': 'CPU'},
                                          100: {'label': 'GPU'}
                                          }
                                     ),
                        ],
                    ),
                    html.Div(
                        [
                          html.P("Uses Model Serving"),
                          dbc.RadioItems(
                              options=[
                                  {"label": "Yes", "value": 1},
                                  {"label": "No", "value": 0},
                              ],
                              value=0, id="model_serving_bin", inline=True
                          ),
                        ],
                      className="mb-3"),
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),

          dbc.Col(
              html.Div([
                  html.H4("Data Analysis / BI", className="card-header"),
                  html.Div(
                     html.H6("SELECT * FROM DATA+AI", className="card-subtitle text-muted"),
                     className="card-body"
                  ),
                  html.Div(html.Img(src='assets/data-analysis.svg',style={'height':'40%', 'width':'40%'}),
                           style={'textAlign': 'center'}),
                  dbc.Form([
                    html.Div(
                        [
                          html.P("SQL usage in %"),
                          dbc.Input(type="number", min=0, max=100, step=1, value=20, id="pct_bi", className="form-control", invalid=False),
                        ],
                      className="mb-3"),
                    html.Div(
                        [
                          html.P("Serverless usage in %"),
                          dcc.Slider(id="ServerlessSqlPercent", min=0, max=100, step=10 , value=round(df_dollars["ServerlessSqlPercent"].mean(),1) * 100, 
                                     className="dbc", marks={
                                          0: {'label': '0%'},
                                          100: {'label': '100%'}
                                          }
                                    ),
                        ],
                      className="mb-3"),
                  ],style = {'margin-left':'5px', 'margin-top':'5px', 'margin-right':'5px'})
              ],className="card mb-3")
          ),
        ],
        style = {'margin-left':'10px', 'margin-top':'7px', 'margin-right':'10px'}
    ),
  ]
)
@app.callback(
      Output('predict-section', 'children'),
      [
        Input('prediction-button', 'n_clicks'),
        State('cloudType', 'value'),
        State('marketSegment', 'value'),
        State('industryVertical', 'value'),
        State('customerStatus', 'value'),
        State('pct_ml', 'value'),
        State('pct_de', 'value'),
        State('pct_bi', 'value'),
        State('pct_automation', 'value'),
        State('DailyGbProcessCatOrd', 'value'),
        State('DeltaPercent', 'value'),
        State('nUsers', 'value'),
        State('model_serving_bin', 'value'),
        State('customerAgeQuarters', 'value'),
        State('pct_gpu', 'value'),
        State('pct_photon', 'value'),
        State('pct_streaming', 'value'),
        State('DLTPercent', 'value'),
        State('ServerlessSqlPercent', 'value')
      ]
    )
def predict(n_clicks,cloudType,marketSegment,industryVertical,customerStatus,pct_ml,pct_de,pct_bi,pct_automation,DailyGbProcessCatOrd,DeltaPercent,nUsers,model_serving_bin,customerAgeQuarters,pct_gpu,pct_photon,pct_streaming,DLTPercent,ServerlessSqlPercent):
    if n_clicks == 0:
        return ""
    else:
        children = []
        dollars = call_models(cloudType,marketSegment,industryVertical,customerStatus,pct_ml/100,pct_de/100,pct_bi/100,pct_automation/100,DailyGbProcessCatOrd,DeltaPercent/100,nUsers,model_serving_bin,customerAgeQuarters,pct_gpu/100,pct_photon/100,pct_streaming/100,DLTPercent/100,ServerlessSqlPercent/100)
        
        children.append(html.H3(f"{max(200, round(dollars[0]/10)*10)} $ DBUs / month",style={'text-align':'center'}))
        
        children.append(html.H6(f"90% confidence interval: [{max(0, round(dollars[1]/10)*10)}$ - {round(dollars[2]/10)*10}$]", style={'text-align':'center'}))

        table_header = [
          html.Thead(html.Tr([html.Th("SKU"), html.Th("DBUs")]))
        ]

        row1 = html.Tr([html.Td("Jobs Compute"), html.Td(100)])
        row2 = html.Tr([html.Td("Delta Live Table"), html.Td(100)])
        row3 = html.Tr([html.Td("SQL Compute"), html.Td(100)])
        row4 = html.Tr([html.Td("All Purpose Compute"), html.Td(100)])
        row5 = html.Tr([html.Td("Serverless SQL"), html.Td(100)])
        row6 = html.Tr([html.Td("Model Serving"), html.Td(100)])

        table_body = [html.Tbody([row1, row2, row3, row4, row5, row6])]

        table = dbc.Table(
            # using the same table as in the above example
            table_header + table_body,
            color="light",
        ),

        children.append(dbc.Button("Nice but what should I report into Salesforce now?", style={'font-style': 'italic'},
                                   className="btn btn-light", n_clicks=0, disabled=False, id="salesforce-btn"))
        children.append(
             dbc.Offcanvas(
              children=table,
              id="off-canvas",
              title="DBUs / SKU",
              is_open=False)
        )   
        return children

@app.callback(
      Output('pct_de', 'invalid'),
      Output('pct_ml', 'invalid'),
      Output('pct_bi', 'invalid'),
      Output('prediction-button', 'disabled'),
      [
        Input('pct_de', 'value'),
        Input('pct_ml', 'value'),
        Input('pct_bi', 'value')
      ]
    )
def sum_is_100(pct_de, pct_ml, pct_bi):
    if pct_de + pct_ml + pct_bi == 100:
        return False, False, False, False
    else:
        return True, True, True, True
@app.callback(
    Output("off-canvas", "is_open"),
    Input("salesforce-btn", "n_clicks"),
    [State("off-canvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

if __name__=='__main__':
    app.run_server(debug=True)
