import pandas as pd
import numpy as np
# import sk-learn as skl...

import utils.common_utils as u
import model.model as model


def demo() -> None:
    """
    do something (say 'hello world', in this case)
    """
    model.hello()


def main() -> None:
    """
    Main execution path of function
    :return: nothing
    """
    try:
        conn = u.sqlite_connect()
        train = u.run_sqlite_query(conn=conn, table_name="train")
        # test = u.get_table(conn=conn, table_name="test")
    finally:
        if conn:
            conn.close()
    print("done with database retrieval")

    print(train.head())



if __name__ == '__main__':
    demo()
    main()