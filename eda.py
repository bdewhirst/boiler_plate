import pandas as pd

import utils.common_utils as u


def get_data() -> pd.DataFrame:
    """
    For now, hard coded function to pull example data.

    exactly how/where data should be pulled
    :return: pandas dataframe with the intended raw data
    """
    try:
        conn = u.sqlite_connect()
        data = u.run_sqlite_query(conn=conn, table_name="train")
    finally:
        if conn:
            conn.close()
    return data


def eda(data: pd.DataFrame) -> None:
    """
    main execution loop-- "do exploratory data analysis"
    :param data: pandas dataframe of data we're exploring, analyzing
    :return: nothing (it writes to standard out, or logs if logs are later implemented)
    """
    # still not showing full width... (pycharm + pandas issue...)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_columns", 10)
    message = "".join(["data shape is ", str(data.shape)])
    print(message)  # <-- just use a logger?
    message = "".join(["missing data by column: \r\n", str(data.isnull().sum())])
    print(message)
    message = "".join(["data head is: \r\n", str(data.head())])
    print(message)
    message = "".join(["pandas describe is: \r\n", str(data.describe())])
    print(message)
    message = "".join(["data types are: \r\n", str(data.dtypes)])
    print(message)

    # correlations-- table of correlation of x0, x1... xN v. y matrix (seaborne (sp.) for heatmap?)
    # correlation matrix
    """
    # using seaborn and matplotlib to do correlation matrix w/ heatmap (here, of X's)

    Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C',
                         'Embarked_S', 'Sex_male', 'IsMinor']
    X = final_train[Selected_features]
    
    plt.subplots(figsize=(8, 5))
    sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
    plt.show()
    
    # other EDA helper stuff
    prints(some_dataframe.isnull().sum())
    """


if __name__ == "__main__":
    data: pd.DataFrame = get_data()
    eda(data)
