import statsmodels.api as sm
import pandas as pd
import sklearn
from sklearn import linear_model


def split_xs_and_ys(
    df: pd.DataFrame,
    x_cols: list,
    y_col: str,
) -> tuple:
    """
    :param df: dataframe to split into pre
    :param x_cols: predictors of y
    :param y_col: column to be solved for
    :return: tuple of two numpy objects based on xs and y respectively
    """
    xs = df[x_cols]
    y = df[[y_col]]
    return (xs, y)


def do_test_train_split(
    df: pd.DataFrame, indep_vars: list, dep_var: str, test_size: float = 0.25
) -> tuple:
    """
    given a pandas dataframe and specified independent variable, perform a test train split and return tuple of all four
    :param df: pandas dataframe
    :param indep_vars: list of independent variable columns to train on
    :param dep_var: dependent variable's column name
    :param test_size: fractional value to determine size of holdout (test data)
    :return: tuple of the matrices of independent variables to train on and corresponding train/test dependent variables
    """
    xs, y = split_xs_and_ys(
        df=df,
        x_cols=indep_vars,
        y_col=dep_var,
    )
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        xs, y, test_size=test_size
    )
    return x_train, x_test, y_train, y_test


def do_global_naive(y: pd.Series) -> dict:
    """
    Take the global average of all dependent training data
    :param y: array of values of dependent variable
    :return: dictionary of model type and fitted model as key/value pair
    """
    # take a global average-- work on this further
    y_mean = y.mean()
    return {
        "global_naive": y_mean,
    }


def do_groupby_naive(xs: pd.DataFrame, y: pd.Series) -> dict:
    """
    ...
    :param xs:
    :param y:
    :return:
    """
    raise ValueError("This model type isn't actually implemented yet")


def do_statsmodels_lm(xs, y: pd.Series) -> dict:
    """
    Fit a linear regression on the specified columns using statsmodels
    :param xs: (i.e. X) pandas dataframe to fit linear regression to
    :param y: pandas dataframe to fit linear regression to
    :return: dictionary of model type and fitted model as key/value pair
    """
    xs2 = sm.add_constant(xs)
    est = sm.OLS(y, xs2)
    est2 = est.fit()
    print(est2.summary())
    return {"sm_linear": est2}


def fit_lin_reg(xs, y: pd.Series) -> dict:
    """
    Fit a linear regression on the specified columns using sklearn
    :param xs: (i.e. X) pandas dataframe to fit linear regression to
    :param y: pandas dataframe to fit linear regression to
    :return: dictionary of model type and fitted model as key/value pair
    """
    regr = linear_model.LinearRegression()
    fit_model = regr.fit(X=xs, y=y)
    return {"sk_linear": fit_model}
