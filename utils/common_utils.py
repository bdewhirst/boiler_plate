import sqlite3
import os

import pandas as pd


def csv_loader(
    source_file: str,
) -> pd.DataFrame:
    """
    Read specified file from /data folder
    :param source_file: filename as string, with extension (e.g. 'data/titanic.csv')
    :return: pandas dataframe from indicated CSV
    """
    path_to_load: str = "".join([source_file])
    data = pd.read_csv(path_to_load)
    return data


def sqlite_connect():
    """
    Create a connection to the sqlite database specified by the included constants
    :return: connection to database
    """
    YOUR_PREFERRED_DIRECTORY = r"C:\Users\Brian\sqlite\db"
    DB_NAME = "pythonsqlite.db"

    db_file = os.path.join(YOUR_PREFERRED_DIRECTORY, DB_NAME)
    try:
        conn = sqlite3.connect(db_file)
        (
            "connecting to sqlite-- remember to exit cleanly"
        )  # e.g., see finally... if conn... commented lines below
    except sqlite3.Error as e:
        print(e)
    # finally:
    #     if conn:
    #         conn.close()
    # print("done")

    return conn


def run_sqlite_query(
    conn,
    table_name,
) -> pd.DataFrame:
    """
    Retrieve specified table via specified connection
    :param conn: database connection
    :param table_name: name of table/view to retrieve
    :return: pandas dataframe
    """
    # print("note-- currently `run_sqlite_query(...)` is rather inflexible")

    query = f"select * from {table_name}"
    data = pd.read_sql_query(sql=query, con=conn)

    return data
