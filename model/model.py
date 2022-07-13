import sklearn
from sklearn import linear_model
import pandas as pd
import numpy as np


def hello() -> None:
    print("hello world")


def prep_lin_reg(df: pd.DataFrame, cols_to_drop: list, cols_to_dummy: list, cols_w_nan: list,) -> pd.DataFrame:
    """
    prepare the training/test data for use with linear regresssion-- i.e., everything has to be a number or dummy
    :param df:
    :param cols_to_drop: list of column names to drop
    :param cols_to_dummy: list of columns to convert to dummy variables
    :return:
    """
    # drop columns not thought to meaningfully encode data-- that said, there are titles and stuff
    df = df.drop(labels=cols_to_drop, axis=1)
    # dummy vars
    df = pd.get_dummies(df, columns=cols_to_dummy)

    # fill NaNs, etc. using mean
    for c in cols_w_nan:
        df[c].fillna(value=df[c].mean(), inplace=True)

    # convert cols to lower case
    df.columns = df.columns.str.lower()
    return df


def fit_lin_reg(df: pd.DataFrame, x_cols: list, y_col: str="survived",):  # -> model:
    """
    Fit a linear regresssion on the specified columns
    :param df: pandas dataframe to fit linear regression to

    :return:

    n.b.: this isn't expected to perform great, but it is often best to start small and work up.
    """
    # consider refactoring out this prep-- it'll need to be applied to `test` as well
    xs = df[x_cols].to_numpy()  # x_cols is already a list, whereas y_col isn't, hence the syntax difference
    y = df[[y_col]].to_numpy()

    regr = linear_model.LinearRegression()
    
    regr.fit(X=xs, y=y)

    return regr


# tmp --- tmp --- tmp --- tmp --- tmp --- tmp --- tmp --- tmp --- tmp --- tmp
from utils.common_utils import sqlite_connect, run_sqlite_query

cn = sqlite_connect()
train = run_sqlite_query(conn=cn, table_name="train")
to_drop = ["PassengerId", "Name", "Ticket", "Cabin",]
to_dummy = ["Sex", "Embarked",]
fix_nan = ["Age", ]
train_cleaned = prep_lin_reg(df=train, cols_to_drop=to_drop, cols_to_dummy=to_dummy, cols_w_nan=fix_nan)
train_on = ["pclass", "age", "sibsp", "parch", "fare", "sex_male", "embarked_c", "embarked_q"]  # everything, without extra (male 1 means female 0, etc.)
train_linear_model = fit_lin_reg(df=train_cleaned, x_cols=train_on)

# ...
# /tmp