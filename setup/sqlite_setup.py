import sqlite3
import os


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    # n.b.: trying to connect to a SQLite database that doesn't exist results in the creation of said database
    # (this assumes the directory already exists-- it'll only create the *.db file)

    YOUR_PREFERRED_DIRECTORY = r"C:\Users\Brian\sqlite\db"
    DB_NAME = "pythonsqlite.db"

    db_location = os.path.join(YOUR_PREFERRED_DIRECTORY, DB_NAME)
    create_connection(db_location)


# based on the tutorials at:
#   https://www.sqlitetutorial.net/sqlite-python/creating-database/
#   https://towardsdatascience.com/python-sqlite-tutorial-the-ultimate-guide-fdcb8d7a4f30
