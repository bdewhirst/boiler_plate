from google.cloud import bigquery
import pandas as pd


# bigquery_queries: list = ["test.sql",]
#
# def run_query(query: str,) -> pd.DataFrame():
#     """execute the indicated query and return the result as a dataframe"""
#     query_path = "".join(["/queries/", query])
#     print(query_path)
#
# client = bigquery.Client()
#
# for query in bigquery_queries:
#     run_query(query=query)

client = bigquery.Client(project="able-air-354719", credentials="google_credential.json")
# query="test.sql"

query = """
    select * from `bigquery-public-data.census_bureau_acs.censustract_2017_5yr`
    WHERE geo_id LIKE '25017%' --MIddlesex County, MA (FIPS ID)
    limit 10
"""
bigquery_queries: list = ["test.sql",]
query_path = "".join(["/queries/", query])

output = client.query(query_path)