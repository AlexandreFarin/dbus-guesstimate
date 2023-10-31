-- Databricks notebook source
SET ansi_mode = false

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Base query: dbus, dollars, and customers info

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW base AS
SELECT
  monthStart,
  workspaceId,
  sfdcAccountId,
  sfdcAccountName,
  cloudType,
  CASE 
    WHEN customerStatus = "SCP" THEN "commit"
    ELSE customerStatus
  END customerStatus,
  CASE
    WHEN industryVertical = "Retail & CPG" THEN "Retail & CPG, Food"
    ELSE industryVertical
  END industryVertical,
  CASE
    WHEN marketSegment = "Unknown Market Segment" THEN NULL
    ELSE marketSegment
  END marketSegment,
  SUM(dollars) AS dollars,
  SUM(dbus) AS dbus
FROM
  prod.workloads_sku_agg
WHERE
  monthStart = date_add(last_day(add_months(current_date(), -2)), 1)
  AND cloudType IN ("gcp", "azure", "aws")
  AND customerType = "Customer"
  AND sfdcAccountId IS NOT NULL
  AND lower(sfdcAccountName) != "databricks"
GROUP BY
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8

-- COMMAND ----------

SELECT COUNT(*) FROM base

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Users

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW users AS
SELECT
  workspaceId,
  nUsers
FROM
  redash_ds.workspaceinsights
WHERE
  weekDate = (
    SELECT
      max(weekDate)
    FROM
      redash_ds.workspaceinsights
  )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Compute type

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW workloads AS
SELECT
  workspaceId,
  (ml_eda + ml_etl) / (ml_eda + ml_etl + de_eda + de_etl + bi) AS pct_ml,
  (de_eda + de_etl) / (ml_eda + ml_etl + de_eda + de_etl + bi) AS pct_de,
  bi / (ml_eda + ml_etl + de_eda + de_etl + bi) AS pct_bi,
  structuredStreamingDbus / ETLDbus AS pct_streaming,
  ETLDbus / (ETLDbus + EDADbus) AS pct_automation
FROM
  (
    SELECT
      workspaceId,
      nvl(sum(intCmdsFromMlNotebooksDbus), 0) AS ml_eda,
      greatest(nvl(sum(EDADbus - intCmdsFromMlNotebooksDbus), 0),0) AS de_eda,
      nvl(sum(mlJobsDbus), 0) AS ml_etl,
      greatest(nvl(sum(ETLDbus - mlJobsDbus), 0),0) AS de_etl,
      nvl(sum(EDWDbus), 0) AS bi,
      nvl(sum(structuredStreamingDbus), 0) AS structuredStreamingDbus,
      nvl(sum(EDADbus), 0) AS EDADbus,
      nvl(sum(ETLDbus), 0) AS ETLDbus
    FROM
      prod_insights.feature_kpi_workspace_daily
    WHERE
      monthStart = date_add(last_day(add_months(current_date(), -2)), 1)
    GROUP BY
      1
  )


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Delta / Data

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW data AS WITH data_volume AS (
  SELECT
    workspaceID,
    month,
    scanType,
    round(
      sum(scannedBytes) * pow(10, -9) / day(last_day(month)),
      0
    ) as DailyGbProcess
  FROM
    prod_insights.table_insights
  WHERE
    month = date_add(last_day(add_months(current_date(), -2)), 1)
  GROUP BY
    1,
    2,
    3
)
SELECT
  t.*,
  coalesce(
    round(dv.DailyGbProcess / t.DailyGbProcess, 2),
    0
  ) AS DeltaPercent,
  CASE
    WHEN t.DailyGbProcess < 10 THEN "0-10 G"
    WHEN t.DailyGbProcess BETWEEN 10
    AND 100 THEN "10-100 G"
    WHEN t.DailyGbProcess BETWEEN 100
    AND 500 THEN "100-500 G"
    WHEN t.DailyGbProcess BETWEEN 500
    AND 1000 THEN "500-1000 G"
    WHEN t.DailyGbProcess BETWEEN 1000
    AND 10000 THEN "1-10 T"
    ELSE "+10 T "
  END DailyGbProcessCat,
  CASE
    WHEN t.DailyGbProcess < 10 THEN 1
    WHEN t.DailyGbProcess BETWEEN 10
    AND 100 THEN 2
    WHEN t.DailyGbProcess BETWEEN 100
    AND 500 THEN 3
    WHEN t.DailyGbProcess BETWEEN 500
    AND 1000 THEN 4
    WHEN t.DailyGbProcess BETWEEN 1000
    AND 10000 THEN 5
    ELSE 6
  END DailyGbProcessCatOrd
FROM
  (
    SELECT
      workspaceID,
      sum(DailyGbProcess) AS DailyGbProcess
    FROM
      data_volume
    GROUP BY
      1
  ) AS t
  LEFT OUTER JOIN data_volume dv ON t.workspaceID = dv.workspaceID
  AND dv.scanType = "Delta"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Machine Learning

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW model_serving AS
SELECT
  workspaceId,
  CASE
    WHEN sum(modelServingDbus) > 0 THEN 1
    ELSE 0
  END model_serving_bin
FROM
  prod_insights.feature_kpi_workspace_daily
WHERE
  monthStart = date_add(last_day(add_months(current_date(), -2)), 1)
GROUP BY
  1

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW cpu_gpu AS WITH ml_workloads AS (
  SELECT
    workspaceId,
    clusterFeatureFlags["GPU"] AS has_gpu,
    sum(attributedInfo["attributedDbus"]) AS dbus
  FROM
    prod.workload_insights
  WHERE
    date BETWEEN date_add(last_day(add_months(current_date(), -2)), 1)
    AND date_add(last_day(add_months(current_date(), -1)), 1)
    AND clusterFeatureFlags ["MLR"] = "true"
  GROUP BY
    1,
    2
)
SELECT
  workspaceId,
  `true` AS has_gpu,
  `false` AS has_cpu,
  nvl(`true`, 0) / (nvl(`true`, 0) + nvl(`false`, 0)) AS pct_gpu
FROM
  (ml_workloads) PIVOT (
    sum(dbus) for has_gpu in ("true", "false")
  )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Photon

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW photon AS WITH de_workloads AS (
  SELECT
    workspaceId,
    clusterFeatureFlags["photon"] AS is_photon,
    sum(attributedInfo["attributedDbus"]) AS dbus
  FROM
    prod.workload_insights
  WHERE
    date BETWEEN date_add(last_day(add_months(current_date(), -2)), 1)
    AND date_add(last_day(add_months(current_date(), -1)), 1)
    AND clusterFeatureFlags ["MLR"] = "false"
  GROUP BY
    1,
    2
)
SELECT
  workspaceId,
  `true` AS is_photon,
  `false` AS is_not_photon,
  nvl(`true`, 0) / (nvl(`true`, 0) + nvl(`false`, 0)) AS pct_photon
FROM
  (de_workloads) PIVOT (
    sum(dbus) for is_photon in ("true", "false")
  )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Delta Live Table

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW dlt AS
SELECT
workspaceId,
round(sum(CASE WHEN sku LIKE "%DLT%" THEN dbus ELSE 0 END) / sum(dbus), 2) AS DLTPercent
FROM prod.workloads_sku_agg
WHERE sku NOT LIKE "%SQL%"
AND monthStart = date_add(last_day(add_months(current_date(), -2)), 1)
GROUP BY 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## SQL Serverless

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW serverless_sql AS
SELECT
workspaceId,
round(sum(CASE WHEN sku LIKE "%SERVERLESS%" THEN dbus ELSE 0 END) / sum(dbus), 2) AS ServerlessSqlPercent
FROM prod.workloads_sku_agg
WHERE sku LIKE "%SQL%"
AND monthStart = date_add(last_day(add_months(current_date(), -2)), 1)
GROUP BY 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Final dataset

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW workspaces AS
SELECT
  base.*,
  datediff(QUARTER, ac.CreatedDate, date_add(last_day(add_months(current_date(), -2)), 1)) AS customerAgeQuarters,
  users.nUsers,
  workloads.pct_ml,
  workloads.pct_de,
  workloads.pct_bi,
  workloads.pct_automation,
  workloads.pct_streaming,
  data.DeltaPercent,
  data.DailyGbProcess,
  data.DailyGbProcessCat,
  data.DailyGbProcessCatOrd,
  model_serving.model_serving_bin,
  cpu_gpu.pct_gpu,
  photon.pct_photon,
  dlt.DLTPercent,
  serverless_sql.ServerlessSqlPercent
FROM
  base
  LEFT OUTER JOIN users ON base.workspaceId = users.workspaceId
  LEFT OUTER JOIN workloads ON base.workspaceId = workloads.workspaceId
  LEFT OUTER JOIN data ON base.workspaceId = data.workspaceId
  LEFT OUTER JOIN model_serving ON base.workspaceId = model_serving.workspaceId
  LEFT OUTER JOIN sfdc.accounts AS ac ON base.sfdcAccountId = ac.Id
  LEFT OUTER JOIN cpu_gpu ON base.workspaceId = cpu_gpu.workspaceId
  LEFT OUTER JOIN photon ON base.workspaceId = photon.workspaceId
  LEFT OUTER JOIN serverless_sql ON base.workspaceId = serverless_sql.workspaceId
  LEFT OUTER JOIN dlt ON base.workspaceId = dlt.workspaceId
WHERE base.dollars BETWEEN 200 AND 100000
AND workloads.pct_ml IS NOT NULL


-- COMMAND ----------

SELECT COUNT(*) FROM workspaces

-- COMMAND ----------

SELECT * FROM workspaces WHERE sfdcAccountId = "0016100001AUNhXAAX"

-- COMMAND ----------

SELECT * FROM workspaces WHERE sfdcAccountId = "0016100001F0UhOAAV"

-- COMMAND ----------

SELECT * FROM workspaces WHERE sfdcAccountId = "0016100001F0UhOAAV"

-- COMMAND ----------

SELECT * FROM workspaces WHERE sfdcAccountId = "0016100001F0UhOAAV"

-- COMMAND ----------

CREATE
OR REPLACE TABLE users.alexandre_farin.workspaces_last_month AS
SELECT
  *
FROM
  workspaces

-- COMMAND ----------

SELECT * FROM users.alexandre_farin.workspaces_last_month

-- COMMAND ----------


