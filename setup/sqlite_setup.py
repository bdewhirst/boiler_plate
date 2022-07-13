import sqlite3
import os

from utils.common_utils import csv_loader


def create_sqlite_db(db_file):
    """ create a database connection to a SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


def tmp():
        """
        Poke at the famous titanic dataset.

        n.b.: 'train' and 'titanic' aren't the same (length or columns)
        """
        titanic: pd.DataFrame
        train: pd.DataFrame
        test: pd.DataFrame

        agenda: list = ["titanic", "train", "test"]
        for name in agenda:
            print(name)
            df = csv_loader(source_file=''.join([name, ".csv"]))
            sniff_frame(df=df)







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
