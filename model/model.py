from abc import ABC, abstractmethod

import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn import model_selection as skl_model_selection
from sklearn import linear_model as skl_linear_model
from sklearn.metrics import r2_score as skl_r2


class ModelPrep(ABC):
    """
    Abstract base class defining required functionality to prepare data for modeling
    """

    @abstractmethod
    def __init__(self, data, dep_var, indep_vars, test_size):
        raise NotImplementedError(
            "Prep must implement initialization of data, dependent and independent variables"
        )

    @abstractmethod
    def _x_y_split(self):
        raise NotImplementedError(
            "Prep must implement a private method to split data into x and y"
        )

    @abstractmethod
    def _filter_indep_cols(self):
        raise NotImplementedError(
            "Prep must implement a private method filter raw inputs to columns of interest"
        )

    @abstractmethod
    def test_train_split(self):
        raise NotImplementedError(
            "Prep must implement a method to split the data into test and train"
        )


class SimpleModelPrep(ModelPrep):
    def __init__(
        self,
        data: pd.DataFrame,
        dep_var: str,
        indep_vars: list,
        test_size: float = 0.25,
    ):
        self.data = data
        self.dep_var = dep_var
        self.indep_vars = indep_vars
        self.test_size = test_size

    def _filter_indep_cols(self) -> pd.DataFrame:
        return self.data[self.indep_vars]

    def _x_y_split(self) -> tuple:
        xs = self._filter_indep_cols()
        y = self.data[[self.dep_var]]
        return (xs, y)

    def test_train_split(self) -> tuple:
        xs, y = self._x_y_split()
        # x_train, x_test, y_train, y_test
        return skl_model_selection.train_test_split(
            xs,
            y,
            test_size=self.test_size,
        )


class Model(ABC):
    """
    Abstract base class defining required functionality of a model
    """

    # @abstractmethod
    # def __init__(self, data):
    #     raise NotImplementedError(
    #         "Prep must implement initialization of data, dependent and independent variables"
    #     )

    @abstractmethod
    def fit(self, indep_data, dep_data):
        raise NotImplementedError(
            "A model must implement a method to fit data to the model"
        )

    @abstractmethod
    def apply(self, dep_data):
        raise NotImplementedError(
            "A model must implement a method to apply data to a fit model"
        )

    @abstractmethod
    def score(self, true_data):
        raise NotImplementedError(
            "A model must implement a method to fit data to the model"
        )


# statsmodels linear model
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


# skl linear model
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


# scikit-learn logistic regression
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
    ref on different pseudo-R^2 methods: https://datascience.oneoffcoder.com/psuedo-r-squared-logistic-regression.html
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


class NaiveModel(Model):  # not fully implemented /tested
    def _do_global_naive(self, indep_data: pd.DataFrame) -> dict:
        """
        Take the global average of all dependent training data
        :param y: array of values of dependent variable
        :return: dictionary of model type and fitted model as key/value pair
        """
        # take a global average-- work on this further
        return {"global_naive": indep_data.mean()}

    # def _do_groupby_naive(
    #     xs: pd.DataFrame,
    #     y: pd.DataFrame,
    #     group_cols: list,
    # ) -> dict:
    #     """
    #     Use pandas' groupby method to take the averages of each unique combination of columns grouped-by
    #     :param xs:  (i.e. X) independent variables
    #     :param y: 1-D dependent data to fit on by taking
    #     :param group_cols: list of column names to aggregate on
    #     :return: dictionary of a label and 'trained model' (averages by
    #     """
    #     raise ValueError("This model type isn't actually implemented yet")

    def fit(self, indep_data, dep_data):
        self.global_naive_fit = self._do_global_naive(indep_data=indep_data)

    def apply(self, dep_data):
        self.applied_result = self.global_naive_fit.get("global_naive")[
            0
        ]  # i.e., the only value in the series
        # note: this model.apply _purposefully_ does nothing with dep_data-- a global naive model has a constant result

    def score(self, true_data) -> None:
        """
        shoehorn the static value into an array of same length as the test data and evaluate coefficient of determination
        :return: returns nothing, but prints to STDOUT
        """
        print("." * 10)
        print("Accuracy metrics for global naive:")
        y_test_label = true_data.columns
        test_len = len(true_data)
        same_len_array = np.arange(0, test_len, dtype=int)
        y_test = true_data.values.ravel()
        y_pred = pd.DataFrame(
            data=self.applied_result,
            index=same_len_array,
            columns=[y_test_label],
        ).values.ravel()
        r2 = skl_r2(y_true=y_test, y_pred=y_pred)  # strongly consider refactoring
        print("global naive coefficient of determination (R^2) is: ", str(r2))
        print("." * 10)
