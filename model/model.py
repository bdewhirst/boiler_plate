import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn import model_selection as skl_model_selection
from sklearn import linear_model as skl_linear_model
from sklearn.metrics import r2_score as skl_r2


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
    x_train, x_test, y_train, y_test = skl_model_selection.train_test_split(
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


def do_statsmodels_lm(xs: pd.DataFrame, y: pd.Series) -> dict:
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


def do_lin_reg(xs: pd.DataFrame, y: pd.Series) -> dict:
    """
    Fit a linear regression on the specified columns using sklearn
    :param xs: (i.e. X) pandas dataframe to fit linear regression to
    :param y: pandas dataframe to fit linear regression to
    :return: dictionary of model type and fitted model as key/value pair
    """
    regr = skl_linear_model.LinearRegression()
    xs = xs
    y = y
    fit_model = regr.fit(X=xs, y=y)
    return {"sk_linear": fit_model}


def score_global_naive(y_test: pd.Series, fit_model) -> None:
    """
    shoehorn the static value into an array of length
    :param y_test:
    :param fit_model:
    :return: returns nothing, but prints to STDOUT
    """
    print("." * 10)
    print("Accuracy metrics for global naive:")
    y_test_label = y_test.columns
    static_prediction = fit_model.iloc[0]
    test_len = len(y_test)
    same_len_array = np.arange(0, test_len, dtype=int)
    y_pred = pd.DataFrame(
        data=static_prediction, index=same_len_array, columns=[y_test_label]
    )
    r2 = skl_r2(y_true=y_test, y_pred=y_pred)  # strongly consider refactoring
    print("global naive coefficient of determination (R^2) is: ", str(r2))
    print("." * 10)


def score_sm_linear_fit(x_test: pd.DataFrame, y_test: pd.Series, fit_model) -> None:
    """
    ...
    :param x_test:
    :param y_test:
    :param fit_model:
    :return:
    """
    print("." * 10)
    print("Accuracy metrics for statsmodels linear regression:")
    y_pred = fit_model.predict(x_test)
    r2 = skl_r2(y_true=y_test, y_pred=y_pred)
    print("global naive coefficient of determination (R^2) is: ", str(r2))
    print("." * 10)


def score_sk_linear_fit(x_test: pd.DataFrame, y_test: pd.Series, fit_model) -> None:
    print("." * 10)
    print("accuracy metrics for Scikit-learn:")
    score = fit_model.score(x_test, y_test)  # i.e. coefficient of determination, R^2
    params = fit_model.get_params(deep=True)
    coefs = fit_model.coef_
    print(
        "R^2 score was: ",
        score,
        "\r\nparams were: ",
        params,
        "\r\ncoefficients were: ",
        coefs,
    )
    print("." * 10)
