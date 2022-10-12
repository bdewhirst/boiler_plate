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


def load_and_clean(
    csv: str, do_sample: bool = False, do_seed: bool = True
) -> pd.DataFrame:
    """
    Load csv-format data and clean it using static code (for now)
    :param csv: string specifying location of csv file (e.g., 'data/sundae.csv')
    :param do_sample: True/False value which controls whether to sample or use all raw data
    :param do_seed: True/False value which controls whether to use a fixed seed for random selection
    :return: returns a pandas dataframe containing the cleaned result

    - could be refactored to separate concerns of loading and cleaning
    - a more complex project would benefit from a dedicated data validation setup
    """
    data = pd.read_csv(
        csv, delimiter=";"
    )  # may need to be refactored if the data isn't in csv format. Alternatively, convert it.

    if do_sample:  # for dev. n.b.: "row 0" here != "row 0" in original.
        if do_seed:
            data = data.sample(
                n=c.SAMPLE, replace=True, random_state=c.SEED
            ).reset_index(drop=True)
        else:
            data = data.sample(
                n=c.SAMPLE,
                replace=True,
            ).reset_index(drop=True)

    # data cleaning/prep:  (iterate w/ EDA)
    data.columns = data.columns.str.lower()  # all column names to lower case
    # this next section will be hard-coded for each project for now-- later, this might be lists/dicts in constants.py
    data = pd.get_dummies(data, columns=["y"])
    data = data.drop(labels=["y_no"], axis=1)
    # reference data prep steps/methods
    # cols to dummy-- i.e., onehot encode, etc.
    # cols_to_dummy: list = []  # !!!
    # data = pd.get_dummies(data, columns=cols_to_dummy)
    # cols_to_drop: list = []  # !!!
    # data = data.drop(labels=cols_to_drop, axis=1)
    # cols_w_nan: list = []  # !!!
    # for nancol in cols_w_nan:
    #     data[nancol].fillna(value=data[nancol].mean(), inplace=True)
    # ...
    # ...

    # crude... but may catch something otherwise missed
    data = data.dropna()
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
