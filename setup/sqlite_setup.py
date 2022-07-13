import sqlite3
import os

from utils.common_utils import csv_loader


def write_df_to_db(name, conn, df) -> None:
    """
    Write the specified dataframe to the database whose connection is specified by conn

    :param:
    :param conn: database connection
    :param df: pandas dataframe
    :return: nothing
    """
    df.to_sql(name=name, con=conn, index=False)


def populate_database(conn) -> None:
    """
    Populate database with sample data (in this case, from kaggle titanic data)
    :return:
    """
    # load local data
    manifest: list = ["titanic", "train", "test"]
    for name in manifest:
        print(name)
        path_and_name: str = "".join(["data/", name, ".csv"])
        df = csv_loader(source_file=path_and_name)
        write_df_to_db(name=name, conn=conn, df=df)


def create_sqlite_db(db_file):
    """ create a database connection to a SQLite database"""
    try:
        conn = sqlite3.connect(db_file)
        print("sqlite3 version ", sqlite3.version)
        print("reading some csv files in and writing them to the database")
        populate_database(conn=conn)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
    print("done")


if __name__ == '__main__':
    # n.b.: trying to connect to a SQLite database that doesn't exist results in the creation of said database
    # (this assumes the directory already exists-- it'll only create the *.db file)

    YOUR_PREFERRED_DIRECTORY = r"C:\Users\Brian\sqlite\db"
    DB_NAME = "pythonsqlite.db"

    db_location = os.path.join(YOUR_PREFERRED_DIRECTORY, DB_NAME)
    create_sqlite_db(db_location)


# based on the tutorials at:
#   https://www.sqlitetutorial.net/sqlite-python/creating-database/
#   https://towardsdatascience.com/python-sqlite-tutorial-the-ultimate-guide-fdcb8d7a4f30
# "train.csv" and "test.csv" are from https://www.kaggle.com/c/titanic/data. (titanic.csv is from elsewhere)
