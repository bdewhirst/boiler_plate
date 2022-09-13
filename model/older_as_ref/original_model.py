import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn
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


# scikit-learn logistic regression (regularized out of the box)
def do_skl_logit(xs, y, train_on):
    skl_logit = linear_model.LogisticRegression()  # sklearn
    skl_logit.fit(xs, y)
    return skl_logit  # a trained booster model


def pred_skl_logit(
    fit_model,
    dtest,
    x_cols,
):
    preds = fit_model.predict(dtest)
    pred_probs = fit_model.predict_proba(dtest)[:, 1]
    return preds, pred_probs


def prelim_logit_eval(y_test, y_calc, y_probability_predictions, fit_model) -> None:
    """
    Run adhoc code to output selected model performance metrics
    :param y_test: holdout data as 1d numpy array
    :param y_calc: calculated values corresponding to the actuals of y_test; also a 1d numpy array
    :param y_probability_predictions: probabilities used to generate y_calc (read the docs further JIC); 1d np. array
    :param fit_model: fitted logistic model generating the calculated values above (suggests future refactoring?)
    :return: returns nothing (outputs to stdout and/or csvs written to the working directory

    handy refresher reference: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    ref: https://www.kaggle.com/code/mnassrib/titanic-logistic-regression-with-python/notebook
    """
    [fpr, tpr, thr] = sklearn.metrics.roc_curve(y_test, y_probability_predictions)
    print("Train/Test split results:")
    print(
        fit_model.__class__.__name__
        + " accuracy is %2.3f" % sklearn.metrics.accuracy_score(y_test, y_calc)
    )
    print(
        fit_model.__class__.__name__
        + " log_loss is %2.3f"
        % sklearn.metrics.log_loss(y_test, y_probability_predictions)
    )
    print(
        fit_model.__class__.__name__ + " auc is %2.3f" % sklearn.metrics.auc(fpr, tpr)
    )
    idx = np.min(np.where(tpr > 0.95))  # first threshold w/ sensibility > 0.95
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="coral",
        label="ROC curve (area = %0.3f)" % sklearn.metrics.auc(fpr, tpr),
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot([0, fpr[idx]], [tpr[idx], tpr[idx]], "k--", color="blue")
    plt.plot([fpr[idx], fpr[idx]], [0, tpr[idx]], "k--", color="blue")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - specificity)", fontsize=14)
    plt.ylabel("True Positive Rate (recall)", fontsize=14)
    plt.title("Receiver operating characteristic (ROC) curve")
    plt.legend(loc="lower right")
    plt.show()
    print(
        "Using a threshold of %.3f " % thr[idx]
        + "guarantees a sensitivity of %.3f " % tpr[idx]
        + "and a specificity of %.3f" % (1 - fpr[idx])
        + ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx]) * 100)
    )
    ## manual accuracy from these csvs TLDR: (calc by hand indicates 80% correct (true pos or true neg), 20% incorrect (false pos or false neg)
    # pd.DataFrame(y_calc).to_csv("y_calc.csv", index=False)
    # pd.DataFrame(y_test).to_csv("y_test.csv", index=False)
    return None


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
