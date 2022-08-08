import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import linear_model


def fit_unknown_data(test: pd.DataFrame, model_fit) -> pd.DataFrame:
    """
    Fit test data to the trained model
    :param test: dataframe of test data to fit
    :param model_fit: fitted sklearn model on training data
    :return: result of applying test data
    """
    to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin",
    ]
    to_dummy = [
        "Sex",
        "Embarked",
    ]
    fix_nan = ["Age", "Fare"]
    clean_test_data = prep_data(
        df=test, cols_to_drop=to_drop, cols_to_dummy=to_dummy, cols_w_nan=fix_nan
    )
    on = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "sex_male",
        "embarked_c",
        "embarked_q",
    ]  # dupe code, etc...
    ready_test_data = clean_test_data[on].to_numpy()
    y_predicted = model_fit.predict(ready_test_data)
    y_df = pd.DataFrame(
        y_predicted,
        columns=["survived"],
    )
    # needs work...
    return y_df


def prep_data(
    df: pd.DataFrame,
    cols_to_drop: list,
    cols_to_dummy: list,
    cols_w_nan: list,
) -> pd.DataFrame:
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


def split_xs_and_ys(
    df: pd.DataFrame,
    x_cols: list,
    y_col: str = "survived",
) -> tuple:
    """
    :param df: dataframe to split into pre
    :param x_cols: predictors of y
    :param y_col: column to be solved for
    :return: tuple of two numpy objects based on xs and y respectively
    """
    xs = df[
        x_cols
    ].to_numpy()  # x_cols is already a list, whereas y_col isn't, hence the syntax difference
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


def do_statsmodels_lm(
    xs,
    y,
):
    xs2 = sm.add_constant(xs)
    est = sm.OLS(y, xs2)
    est2 = est.fit()
    print(est2.summary())


def slap_into_xgb_format(
    xs,
    y,
    train_on: list,
) -> xgb.DMatrix:
    """
    given a np array of x(s) and y, assemble

        n.b.:... better upstream plumbing is desirable, when I get around to it

    :param xs: 2d numpy array of features
    :param y: 1d numpy array of results
    :return: a xgboost DMatrix object
    """
    x_df = pd.DataFrame(xs, columns=train_on)
    if y is not None:
        y_df = pd.DataFrame(
            y,
            columns=[
                "Target",
            ],
        )
    else:
        y_df = None

    dtrain = xgb.DMatrix(data=x_df, label=y_df)
    return dtrain


def do_xgb(
    xs,
    y,
    train_on: list,
    param: dict = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"},
):
    num_round: int = 2  # default is 10; presumably, this at 2 is to speed demo code?
    dtrain = slap_into_xgb_format(xs=xs, y=y, train_on=train_on)
    booster = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=num_round,
    )
    return booster  # a trained booster model


def pred_xgb(
    fit_model,
    dtest,
    x_cols: list,
):
    # dtest is expected to be a numpy 2d array, which we'll need to convert into xgb.DMatrix
    xs = slap_into_xgb_format(
        xs=dtest,
        y=None,
        train_on=x_cols,
    )
    preds = fit_model.predict(xs)
    return preds


def score_fit(model, x_test, y_test) -> None:
    # see also statsmodels linear model, where I didn't break out scoring stuff.
    # still could use graphs, etc.
    score = model.score(x_test, y_test)
    params = model.get_params(deep=True)
    coefs = model.coef_
    print(
        "score was: ",
        score,
        "\r\nparams were: ",
        params,
        "\r\ncoefficients were: ",
        coefs,
    )


def strawman_plot(
    xs,
    y,
    cols: list,
) -> None:
    # quickly try to plot what we've got, via pandas's methods
    both = np.append(arr=xs, values=y, axis=1)
    df = pd.DataFrame(data=both, columns=cols)
    # print(df.head())
    y_ = df["survived"]
    for col in cols:  # scroll through each predictor
        x = df[col]
        plt.scatter(
            x, y_
        )  # not useful as-is for discrete values; perhaps a pair of histograms or something else?
        plt.show()
