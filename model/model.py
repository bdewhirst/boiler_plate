import statsmodels.api as sm

# import sklearn
from sklearn import linear_model


def do_test_train_split(
    df: pd.DataFrame, train_on: list, test_size: float = 0.25
) -> tuple:
    """
    given a pandas dataframe and specified independent variable, perform a test train split and return tuple of all four
    :param df: pandas dataframe
    :param train_on: list of independent variable columns to train on
    :param test_size: fractional value to determine size of holdout (test data)
    :return: tuple of the matrices of independent variables to train on and corresponding train/test dependent variables
    """
    xs, y = model.split_xs_and_ys(
        df=df,
        x_cols=train_on,
    )
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        xs, y, test_size=test_size
    )
    return x_train, x_test, y_train, y_test


def do_global_naive(y):
    # take a global average-- work on this further
    y_mean = y.mean()
    pass



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
