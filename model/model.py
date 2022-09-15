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
    df: pd.DataFrame,
    indep_vars: list,
    dep_var: str,
    test_size: float = 0.25,
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
        xs,
        y,
        test_size=test_size,
    )
    return x_train, x_test, y_train, y_test


def do_global_naive(y: pd.DataFrame) -> dict:
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


def do_groupby_naive(
    xs: pd.DataFrame,
    y: pd.DataFrame,
    group_cols: list,
) -> dict:
    """
    Use pandas' groupby method to take the averages of each unique combination of columns grouped-by
    :param xs:  (i.e. X) independent variables
    :param y: 1-D dependent data to fit on by taking
    :param group_cols: list of column names to aggregate on
    :return: dictionary of a label and 'trained model' (averages by
    """
    raise ValueError("This model type isn't actually implemented yet")


def do_statsmodels_lm(
    xs: pd.DataFrame,
    y: pd.DataFrame,
) -> dict:
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


def do_lin_reg(
    xs: pd.DataFrame,
    y: pd.DataFrame,
) -> dict:
    """
    Fit a linear regression on the specified columns using sklearn
    :param xs: (i.e. X) pandas dataframe to fit linear regression to
    :param y: pandas dataframe to fit linear regression to
    :return: dictionary of model type and fitted model as key/value pair
    """
    regr = skl_linear_model.LinearRegression()
    y = y.values.ravel()
    fit_model = regr.fit(X=xs, y=y)
    return {"sk_linear": fit_model}


def score_global_naive(
    y_test: pd.DataFrame,
    fit_model: dict,
) -> None:
    """
    shoehorn the static value into an array of same length as the test data and evaluate coefficient of determination
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
    y_test = y_test.values.ravel()
    y_pred = pd.DataFrame(
        data=static_prediction, index=same_len_array, columns=[y_test_label]
    ).values.ravel()
    r2 = skl_r2(y_true=y_test, y_pred=y_pred)  # strongly consider refactoring
    print("global naive coefficient of determination (R^2) is: ", str(r2))
    print("." * 10)


def score_sm_linear_fit(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    fit_model,
) -> None:
    """
    Evaluate the coefficient of determination (R^2) of statsmodels linear regression
    :param x_test: (aka X) holdout independent variable values
    :param y_test: holdout dependent variable values
    :param fit_model: model trained on training data to be evaluated
    :return: returns nothing; prints to STDOUT
    """
    x_test = sm.add_constant(x_test)
    y_test = y_test.values.ravel()
    print("." * 10)
    print("Accuracy metrics for statsmodels linear regression:")
    y_pred = fit_model.predict(x_test).values.ravel()
    r2 = skl_r2(y_true=y_test, y_pred=y_pred)
    print("statsmodels coefficient of determination (R^2) is: ", str(r2))
    print("." * 10)


def score_sk_linear_fit(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    fit_model,
) -> None:
    """
    Evaluate the coefficient of determination (R^2) of sklearn linear regression
    :param x_test: (aka X) holdout independent variable values
    :param y_test: holdout dependent variable values
    :param fit_model: model trained on training data to be evaluated
    :return: returns nothing; prints to STDOUT
    """
    print("." * 10)
    print("accuracy metrics for Scikit-learn:")
    y_test = y_test.values.ravel()
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


def do_skl_logit(
    xs: pd.DataFrame,
    y: pd.DataFrame,
) -> dict:
    """
    scikit-learn logistic regression (regularized out of the box)
    :param x_test: (aka X) holdout independent variable values
    :param y_test: holdout dependent variable values
    :param fit_model: model trained on training data to be evaluated
    :return: returns nothing; prints to STDOUT
    """
    y = y.values.ravel()
    skl_logit = skl_linear_model.LogisticRegression()
    skl_logit.fit(xs, y)
    return {"logistic": skl_logit}


def score_sk_logistic(
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    fit_model,
) -> None:
    """
    Evaluate the accuracy of the logistic regression on holdout data. Note that "Vanilla R^2" isn't helpful here.
    :param x_test: matrix of independent holdout data (aka X)
    :param y_test: array of dependent holdout data
    :param fit_model: fitted sklearn logistic regression to be evaluated
    :return: returns nothing; prints to STDOUT

    future work: add additional accuracy metrics (a pseudo R^2, possibly plots as well)
    """
    y_test = y_test.values.ravel()
    print("." * 10)
    print("accuracy metrics for Scikit-learn: ")
    score = fit_model.score(x_test, y_test)
    params = fit_model.get_params(deep=True)
    coefs = fit_model.coef_
    print(
        "n.b.: this is not R^2 \r\n\ Mean accuracy on the given test data. score was: ",
        score,
        "\r\nparams were: ",
        params,
        "\r\ncoefficients were: ",
        coefs,
    )
