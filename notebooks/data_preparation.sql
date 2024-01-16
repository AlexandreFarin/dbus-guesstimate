-- Databricks notebook source
SET ansi_mode = false

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW base AS
SELECT
  monthStart,
  workspaceId,
  sfdcAccountId,
  sfdcAccountName,
  accountRegion,
  cloudType,
  CASE
    WHEN customerStatus = "SCP" THEN "commit"
    ELSE customerStatus
  END AS customerStatus,
  CASE
    WHEN industryVertical = "Retail & CPG" THEN "Retail & CPG, Food"
    ELSE industryVertical
  END AS industryVertical,
  CASE
    WHEN marketSegment = "Unknown Market Segment" THEN NULL
    ELSE marketSegment
  END AS marketSegment,
  SUM(dollars) AS dollars,
  SUM(dbus) AS dbus
FROM
  main.data_df_metering.workloads_sku_agg
WHERE
  monthStart IN (
    date_add(last_day(add_months(current_date(), -2)), 1),
    date_add(last_day(add_months(current_date(), -8)), 1),
    date_add(last_day(add_months(current_date(), -14)), 1),
    date_add(last_day(add_months(current_date(), -20)), 1)
  )
  AND cloudType IN ("gcp", "azure", "aws")
  AND customerType = "Customer"
  AND sfdcAccountId IS NOT NULL
  AND lower(sfdcAccountName) != "databricks"
GROUP BY
  ALL

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW metrics AS
SELECT
  CAST(calendar_month AS DATE) AS calendar_month,
  workspace_id,
  datediff(
    QUARTER,
    creation_ts,
    CAST(calendar_month AS DATE)
  ) AS age_quarter,
  max(total_num_users) as max_num_users,
  sum(etl_dbus) / (
    sum(etl_dbus) + sum(ds_dbus) + sum(ml_dbus) + sum(edw_dbus)
  ) as pct_etl,
  (sum(ds_dbus) + sum(ml_dbus)) / (
    sum(etl_dbus) + sum(ds_dbus) + sum(ml_dbus) + sum(edw_dbus)
  ) as pct_ml,
  sum(edw_dbus) / (
    sum(etl_dbus) + sum(ds_dbus) + sum(ml_dbus) + sum(edw_dbus)
  ) as pct_bi,
  sum(automated_job_dbus) / (sum(all_purpose_dbus) + sum(automated_job_dbus)) as pct_automation,
  sum(streaming_dbus) / sum(total_dbus) as pct_streaming,
  sum(delta_dbus) / sum(total_dbus) as pct_delta,
  sum(photon_job_dbus) / sum(job_dbus) as pct_photon,
  sum(dlt_dbus) / sum(etl_dbus) as pct_dlt,
  sum(ml_dbus) / (sum(ds_dbus) + sum(ml_dbus)) as pct_adv_ml,
  sum(serverless_dbus) / sum(total_dbus) as pct_serverless,
  nvl(
    sum(unity_catalog_dbus) / (sum(unity_catalog_dbus) + sum(metastore_dbus)),
    0
  ) as pct_uc,
  /* binary value for whether there is any ml serving traffic */
  CASE
    WHEN sum(ml_serving_dbus) > 0 THEN 1
    ELSE 0
  END as ml_serving_bin,
  /* average gigabytes written and read */
  (avg(total_bytes_written) + avg(total_bytes_read)) * pow(10, -9) as avg_gb
FROM
  main.data_product_features.workspace_metrics
WHERE
  /* date range condition */
  CAST(calendar_month AS DATE) IN (
    date_add(last_day(add_months(current_date(), -2)), 1),
    date_add(last_day(add_months(current_date(), -8)), 1),
    date_add(last_day(add_months(current_date(), -14)), 1),
    date_add(last_day(add_months(current_date(), -20)), 1)
  )
GROUP BY
  ALL

-- COMMAND ----------

CREATE
OR REPLACE TABLE users.alexandre_farin.guesstimate_data AS
SELECT
  *
EXCEPT
  (calendar_month, workspace_id),
  CASE
    WHEN avg_gb < 10 THEN "0-10 G"
    WHEN avg_gb BETWEEN 10
    AND 100 THEN "10-100 G"
    WHEN avg_gb BETWEEN 100
    AND 500 THEN "100-500 G"
    WHEN avg_gb BETWEEN 500
    AND 1000 THEN "500-1000 G"
    WHEN avg_gb BETWEEN 1000
    AND 10000 THEN "1-10 T"
    ELSE "+10 T "
  END avg_gb_cat,
  CASE
    WHEN avg_gb < 10 THEN 1
    WHEN avg_gb BETWEEN 10
    AND 100 THEN 2
    WHEN avg_gb BETWEEN 100
    AND 500 THEN 3
    WHEN avg_gb BETWEEN 500
    AND 1000 THEN 4
    WHEN avg_gb BETWEEN 1000
    AND 10000 THEN 5
    ELSE 6
  END avg_gb_cat_ord
FROM
  base
  LEFT OUTER JOIN metrics ON base.monthStart = metrics.calendar_month
  AND base.workspaceId = metrics.workspace_id
WHERE
  dollars > 200

-- COMMAND ----------


