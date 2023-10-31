# Databricks notebook source
# MAGIC %md
# MAGIC # LightGBM training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **Alexandre Farin's Interactive Cluster Policy Cluster** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/4252588691238631)
# MAGIC - Navigate to the parent notebook [here](#notebook/4252588691238632) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC
# MAGIC Runtime Version: _11.3.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "dollars"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

df = spark.read.table("users.alexandre_farin.workspaces_last_month").toPandas()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["cloudType", "nUsers", "pct_gpu", "customerAgeQuarters", "pct_bi", "DailyGbProcessCatOrd", "pct_streaming", "DeltaPercent", "pct_photon", "marketSegment", "pct_ml", "industryVertical", "pct_de", "model_serving_bin", "pct_automation", "customerStatus", "DLTPercent", "ServerlessSqlPercent"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["DailyGbProcessCatOrd", "DeltaPercent", "customerAgeQuarters", "model_serving_bin", "nUsers", "pct_automation", "pct_bi", "pct_de", "pct_gpu", "pct_ml", "pct_photon", "pct_streaming","DLTPercent", "ServerlessSqlPercent"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["nUsers", "pct_gpu", "customerAgeQuarters", "pct_bi", "pct_streaming", "DeltaPercent", "pct_photon", "DailyGbProcessCatOrd", "pct_ml", "pct_de", "model_serving_bin", "pct_automation", "DLTPercent", "ServerlessSqlPercent"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["cloudType", "customerStatus", "industryVertical", "marketSegment", "model_serving_bin"])]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers + categorical_one_hot_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC
# MAGIC

# COMMAND ----------

df.drop(["workspaceId","monthStart","sfdcAccountId","sfdcAccountName","dbus","DailyGbProcessCat","DailyGbProcess"], axis=1, inplace=True)
df_y = df[target_col]
df_x = df.drop(target_col, axis=1)

# COMMAND ----------

from sklearn.model_selection import train_test_split

train_ratio = 0.60
validation_ratio = 0.20
test_ratio = 0.20

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=1 - train_ratio)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

# COMMAND ----------

len(y_train), len(y_val), len(y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train regression model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/4252588691238631)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

import lightgbm
from lightgbm import LGBMRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

import mlflow
import sklearn
import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials


# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])

mlflow.sklearn.autolog(disable=True)
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

# COMMAND ----------

def objective(params):
  with mlflow.start_run(experiment_id="2543932886726660", run_name="lightgbm") as mlflow_run:
    lgbmr_regressor = LGBMRegressor(**params,objective="quantile", alpha=0.5)

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("regressor", lgbmr_regressor),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True,
    )

    model.fit(X_train, y_train, regressor__callbacks=[lightgbm.early_stopping(5), lightgbm.log_evaluation(0)], regressor__eval_set=[(X_val_processed,y_val)])

    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    lgbmr_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    lgbmr_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    loss = lgbmr_val_metrics["val_mae"]

    # Truncate metric key names so they can be displayed together
    lgbmr_val_metrics = {k.replace("val_", ""): v for k, v in lgbmr_val_metrics.items()}
    lgbmr_test_metrics = {k.replace("test_", ""): v for k, v in lgbmr_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": lgbmr_val_metrics,
      "test_metrics": lgbmr_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

def objective_min(params):
  with mlflow.start_run(experiment_id="2543932886726683", run_name="lightgbm") as mlflow_run:
    lgbmr_regressor = LGBMRegressor(**params,objective="quantile", alpha=0.1)

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("regressor", lgbmr_regressor),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True,
    )

    model.fit(X_train, y_train, regressor__callbacks=[lightgbm.early_stopping(5), lightgbm.log_evaluation(0)], regressor__eval_set=[(X_val_processed,y_val)])

    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    lgbmr_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    lgbmr_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    loss = lgbmr_val_metrics["val_mae"]

    # Truncate metric key names so they can be displayed together
    lgbmr_val_metrics = {k.replace("val_", ""): v for k, v in lgbmr_val_metrics.items()}
    lgbmr_test_metrics = {k.replace("test_", ""): v for k, v in lgbmr_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": lgbmr_val_metrics,
      "test_metrics": lgbmr_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

def objective_max(params):
  with mlflow.start_run(experiment_id="2543932886726721", run_name="lightgbm") as mlflow_run:
    lgbmr_regressor = LGBMRegressor(**params,objective="quantile", alpha=0.9)

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("regressor", lgbmr_regressor),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True,
    )

    model.fit(X_train, y_train, regressor__callbacks=[lightgbm.early_stopping(5), lightgbm.log_evaluation(0)], regressor__eval_set=[(X_val_processed,y_val)])

    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    lgbmr_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    lgbmr_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    loss = lgbmr_val_metrics["val_mae"]

    # Truncate metric key names so they can be displayed together
    lgbmr_val_metrics = {k.replace("val_", ""): v for k, v in lgbmr_val_metrics.items()}
    lgbmr_test_metrics = {k.replace("test_", ""): v for k, v in lgbmr_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": lgbmr_val_metrics,
      "test_metrics": lgbmr_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions but can be
# MAGIC modified to widen the search space. For example, when training a decision tree regressor, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html
# MAGIC
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

space = {
  "feature_fraction": hp.uniform('feature_fraction', 0.2, 0.8),
  "lambda_l1": hp.uniform('lambda_l1', 0.0001, 1),
  "lambda_l2": hp.uniform('lambda_l2', 0.0001, 1),
  "learning_rate": hp.uniform('learning_rate', 0.001, 0.1),
  "max_bin": hp.choice('max_bin', range(20,200)),
  "max_depth": hp.choice('max_depth', range(2,20)),
  "n_estimators": hp.choice('num_iterations', range(10,300)),
  "num_leaves": hp.choice('num_leaves', range(50,100)),
  "subsample": hp.uniform('subsample', 0.5, 0.9),
  "random_state": 310852885,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC

# COMMAND ----------

from hyperopt import SparkTrials
trials = SparkTrials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=100,
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model

# COMMAND ----------

mlflow.autolog(disable=True)
mlflow.sklearn.autolog(disable=True)
from shap import KernelExplainer, summary_plot
# SHAP cannot explain models using data with nulls.
# To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).
mode = X_train.mode().iloc[0]

# Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=310852885).fillna(mode)

# Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
example = X_val.sample(n=min(100, X_val.shape[0]), random_state=310852885).fillna(mode)

# Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
explainer = KernelExplainer(predict, train_sample, link="identity")
shap_values = explainer.shap_values(example, l1_reg=False, nsamples=500)
summary_plot(shap_values, example)

# COMMAND ----------

model_name = "Guestimate"
model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
registered_model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=registered_model_version.version,
    stage="Production"
)
client.transition_model_version_stage(
    name=model_name,
    version=f"{int(registered_model_version.version)-1}",
    stage="Archived"
)

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
  }
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
instance = tags["browserHostName"]

# COMMAND ----------

import requests
def update_endpoint(my_json):
  model_serving_endpoint_name=my_json['name']
  #get endpoint status
  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  new_model_version = (my_json['config'])['served_models'][0]['model_version']
  print("We are updating it to a new config with new model version: ", new_model_version)
  # update config
  url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
  re = requests.put(url, headers=headers, json=my_json['config']) 
  # wait till new config file in place
  import time,json
  #get endpoint status
  url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
  retry = True
  total_wait = 0
  while retry:
    r = requests.get(url, headers=headers)
    assert r.status_code == 200, f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
    endpoint = json.loads(r.text)
    if "pending_config" in endpoint.keys():
      seconds = 10
      print("New config still pending")
      if total_wait < 6000:
        #if less the 10 mins waiting, keep waiting
        print(f"Wait for {seconds} seconds")
        print(f"Total waiting time so far: {total_wait} seconds")
        time.sleep(10)
        total_wait += seconds
      else:
        print(f"Stopping,  waited for {total_wait} seconds")
        retry = False  
    else:
      print("New config in place now!")
      retry = False
  print(re.text)
  assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"

# COMMAND ----------

my_json = {
  "name": "Guestimate",
  "config": {
   "served_models": [{
     "model_name": model_name,
     "model_version": registered_model_version.version,
     "workload_size": "Small",
     "scale_to_zero_enabled": True
   }]
 }
}
update_endpoint(my_json)

# COMMAND ----------

trials = SparkTrials()
fmin(objective_min,
     space=space,
     algo=tpe.suggest,
     max_evals=100,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model_min = best_result["model"]
mlflow_run = best_result["run"]

# COMMAND ----------

model_name = "GuestimateMinimum"
model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
registered_model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=registered_model_version.version,
    stage="Production"
)
client.transition_model_version_stage(
    name=model_name,
    version=f"{int(registered_model_version.version)-1}",
    stage="Archived"
)

# COMMAND ----------

my_json = {
  "name": model_name,
  "config": {
   "served_models": [{
     "model_name": model_name,
     "model_version": registered_model_version.version,
     "workload_size": "Small",
     "scale_to_zero_enabled": True
   }]
 }
}
update_endpoint(my_json)

# COMMAND ----------

trials = SparkTrials()
fmin(objective_max,
     space=space,
     algo=tpe.suggest,
     max_evals=100,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model_min = best_result["model"]
mlflow_run = best_result["run"]

# COMMAND ----------

model_name = "GuestimateMaximum"
model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
registered_model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=registered_model_version.version,
    stage="Production"
)
client.transition_model_version_stage(
    name=model_name,
    version=f"{int(registered_model_version.version)-1}",
    stage="Archived"
)

# COMMAND ----------

my_json = {
  "name": model_name,
  "config": {
   "served_models": [{
     "model_name": model_name,
     "model_version": registered_model_version.version,
     "workload_size": "Small",
     "scale_to_zero_enabled": True
   }]
 }
}
update_endpoint(my_json)

# COMMAND ----------


