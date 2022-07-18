import sklearn
from sklearn import linear_model
import pandas as pd
# import numpy as np


def fit_unknown_data(test: pd.DataFrame, model_fit) -> pd.DataFrame:
    """
    Fit test data to the trained model
    :param test: dataframe of test data to fit
    :param model_fit: fitted sklearn model on training data
    :return: result of applying test data
    """
    # ISSUE!!! "test.csv" doesn't have survived/not, so it is not suitable for evaluating the model fit-- I could probably find it, but I should, instead, do my own holdout first.

    # https://www.datacourses.com/evaluation-of-regression-models-in-scikit-learn-846/
    # X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)

    to_drop = ["PassengerId", "Name", "Ticket", "Cabin", ]
    to_dummy = ["Sex", "Embarked", ]
    fix_nan = ["Age", "Fare"]
    clean_test_data = prep_data(df=test, cols_to_drop=to_drop, cols_to_dummy=to_dummy, cols_w_nan=fix_nan)
    on = ["pclass", "age", "sibsp", "parch", "fare", "sex_male", "embarked_c", "embarked_q"]  # dupe code, etc...
    ready_test_data = clean_test_data[on].to_numpy()
    y_predicted = model_fit.predict(ready_test_data)
    y_df = pd.DataFrame(y_predicted, columns=["survived"],)
    return y_df


def prep_data(df: pd.DataFrame, cols_to_drop: list, cols_to_dummy: list, cols_w_nan: list, ) -> pd.DataFrame:
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


def split_xs_and_ys(df: pd.DataFrame, x_cols: list, y_col: str="survived",) -> tuple:
    """

    :param df: dataframe to split into pre
    :param x_cols: predictors of y
    :param y_col: column to be solved for
    :return: tuple of two numpy objects based on xs and y respectively
    """
    xs = df[x_cols].to_numpy()  # x_cols is already a list, whereas y_col isn't, hence the syntax difference
    y = df[[y_col]].to_numpy()
    return (xs, y)


def fit_lin_reg(xs, y):  # -> model:
    """
    Fit a linear regresssion on the specified columns
    :param df: pandas dataframe to fit linear regression to
    :return: returns sklearn model fit on the provided data

    n.b.: this isn't expected to perform great, but it is often best to start small and work up.
    """
    regr = linear_model.LinearRegression()
    regr.fit(X=xs, y=y)
    return regr


def score_fit(model, x_test, y_test) -> None:
    # need more than this... probably need other modules to do so (scipy, statsmodels, plots, etc.)
    score = model.score(x_test, y_test)
    params = model.get_params(deep=True)
    print("score was: ", score, "\r\nparams were: ", params)
    print(model.coef_)
    print("this is just a placeholder so PyCharm has a valid line to latch onto")


# tmp --- tmp --- tmp --- tmp --- tmp --- tmp --- tmp --- tmp --- tmp --- tmp
from utils.common_utils import sqlite_connect, run_sqlite_query

cn = sqlite_connect()
all_training = run_sqlite_query(conn=cn, table_name="train")
to_drop = ["PassengerId", "Name", "Ticket", "Cabin",]
to_dummy = ["Sex", "Embarked",]
fix_nan = ["Age", ]
all_train_cleaned = prep_data(df=all_training, cols_to_drop=to_drop, cols_to_dummy=to_dummy, cols_w_nan=fix_nan)
train_on = ["pclass", "age", "sibsp", "parch", "fare", "sex_male", "embarked_c", "embarked_q"]  # everything, without extra (male 1 means female 0, etc.)
xs, y = split_xs_and_ys(df=all_train_cleaned, x_cols=train_on,)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xs, y, test_size =0.25)
del all_train_cleaned, xs, y

trained_linear_model = fit_lin_reg(xs= x_train, y=y_train)

score = score_fit(model=trained_linear_model,x_test=x_test, y_test=y_test,)

# scoring stuff

test = run_sqlite_query(conn=cn, table_name="test")
fit_unknown_data(test=test, model_fit=trained_linear_model)

# ...
# /tmp