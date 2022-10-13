from abc import ABC, abstractmethod
import warnings

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
        # output is x_train, x_test, y_train, y_test
        return skl_model_selection.train_test_split(xs, y, test_size=self.test_size)


class ABCModel(ABC):
    """
    Abstract base class defining required functionality of a model
    """

    @abstractmethod
    def fit(self, indep_data, dep_data):
        raise NotImplementedError(
            "A model must implement a method to fit data to the model"
        )

    @abstractmethod
    def apply(self, indep_data):
        raise NotImplementedError(
            "A model must implement a method to apply data to a fit model"
        )

    @abstractmethod
    def score(self, true_data):
        raise NotImplementedError(
            "A model must implement a method to fit data to the model"
        )


class Model(ABCModel):
    """
    Model class with "common-denominator" versions of apply and score methods
    """
    def fit(self, indep_data, dep_data):
        warnings.warn("Warning: This is a default method")
        self.fit_model = None

    def apply(self, indep_data):
        self.applied_results = self.fit_model.predict(indep_data)

    def score(self, true_data):
        y_test = true_data.values.ravel()
        y_pred = self.applied_results
        r2 = skl_r2(y_true=y_test, y_pred=y_pred)
        print("model coefficient of determination (R^2) is: ", str(r2))


class StatsModelsLinear(Model):
    def fit(self, indep_data, dep_data):
        """
        Fit a linear regression on the specified columns using statsmodels
        :param dep_data: (i.e. X) pandas dataframe to fit linear regression to
        :param indep_data: pandas dataframe to fit linear regression to
        :return: dictionary of model type and fitted model as key/value pair
        """
        xs = sm.add_constant(indep_data)
        est = sm.OLS(dep_data, xs)
        self.fit_model = est.fit()
        print("In-sample fit is:", self.fit_model.summary())

    def apply(self, indep_data):
        x_test = sm.add_constant(indep_data)
        self.applied_results = self.fit_model.predict(x_test).values.ravel()

    def score(self, true_data):
        r2 = skl_r2(y_true=true_data, y_pred=self.applied_results)
        print("coefficient of determination (R^2) is: ", str(r2))


class SciKitLearnLinear(Model):
    def fit(self, indep_data, dep_data):
        """
        Fit a linear regression on the specified columns using sklearn
        :param dep_data: (i.e. X) pandas dataframe to fit linear regression to
        :param indep_data: pandas dataframe to fit linear regression to
        :return: dictionary of model type and fitted model as key/value pair
        """
        regr = skl_linear_model.LinearRegression()
        y = dep_data.values.ravel()
        self.fit_model = regr.fit(X=indep_data, y=y)
        params = self.fit_model.get_params(deep=True)
        coefs = self.fit_model.coef_
        cols = indep_data.columns
        print(
            "\r\nparams were: ",
            params,
            "\r\ncolumns were: ",
            cols,
            "\r\ncoefficients were: ",
            coefs,
        )


class SciKitLearnLogistic(Model):
    def fit(self, indep_data, dep_data):
        y = dep_data.values.ravel()
        skl_logit = skl_linear_model.LogisticRegression()
        self.fit_model = skl_logit.fit(indep_data, y)
        params = self.fit_model.get_params(deep=True)
        coefs = self.fit_model.coef_
        cols = indep_data.columns
        print(
            "\r\nparams were: ",
            params,
            "\r\ncolumns were: ",
            cols,
            "\r\ncoefficients were: ",
            coefs,
        )


    def score(self, true_data):
        """
        Evaluate the accuracy of the logistic regression on holdout data. Note that "Vanilla R^2" isn't helpful here.
        :param x_test: matrix of independent holdout data (aka X)
        :param y_test: array of dependent holdout data
        :param fit_model: fitted sklearn logistic regression to be evaluated
        :return: returns nothing; prints to STDOUT

        future work: ADD ADDITIONAL ERROR METRICS (a pseudo R^2, true pos/ false pos, etc., possibly plots as well)
        ref on different pseudo-R^2 methods: https://datascience.oneoffcoder.com/psuedo-r-squared-logistic-regression.html
        """
        print("." * 10)
        print("TEMPORARY accuracy metrics for Scikit-learn LOGISTIC regression:")
        y_test = true_data
        y_pred = self.applied_results
        r2 = skl_r2(y_true=y_test, y_pred=y_pred)
        warnings.warn("Warning: this isn't a good/sufficient accuracy metric")
        print("coefficient of determination (R^2) is: ", str(r2))

    # ensembling?
    # evaluate ensemble results?
    # treat as a separate 'class' of model

    # random forest, etc., from earlier? (classifier, regressor...)


class NaiveModel(Model):
    def _do_global_naive(self, dep_data: pd.DataFrame) -> dict:
        """
        Take the global average of all dependent training data
        :param y: array of values of dependent variable
        :return: dictionary of model type and fitted model as key/value pair
        """
        # take a global average-- work on this further
        return {"global_naive": dep_data.mean()}

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
        self.global_naive_fit = self._do_global_naive(dep_data=dep_data)

    def apply(self, indep_data):
        self.applied_result = self.global_naive_fit.get("global_naive")[
            0
        ]
        # note: this model.apply _purposefully_ does nothing with indep_data: a global naive model has a constant result

    def score(self, true_data) -> None:
        """
        shoehorn the static value into an array of same length as the test data and evaluate coefficient of determination
        :return: returns nothing, but prints to STDOUT
        """
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
        print("coefficient of determination (R^2) is: ", str(r2))
