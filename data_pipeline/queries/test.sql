-- 2022-07-08 bdewhirst
-- example BigQuery query which accesses a public dataset


select * from `bigquery-public-data.census_bureau_acs.censustract_2017_5yr`
WHERE geo_id LIKE '25017%' --MIddlesex County, MA (FIPS ID)
limit 10;