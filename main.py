import pandas as pd
import numpy as np

import utils.constants as c
from utils import eda_tools
from model import model


def main(do_sample: bool = False, do_eda: bool = False, do_seed: bool = True) -> None:
    """
    Main function which is called when the model framework is run
    :param do_sample: True/False value which controls whether raw inputs are sampled or complete
    :param do_eda: True/False value which controls whether to run semi-automated exploratory data analysis
    :param do_seed: True/False value which controls whether to set a seed for
    :return: returns nothing; outputs to STDOUT
    """
    # for reproducible iteration, set seed
    if do_seed:
        np.random.seed(c.SEED)
    data = load_and_clean(csv=c.RAWCSV, do_sample=do_sample, do_seed=do_seed)

    if do_eda:
        eda(data=data)  # n.b.: some approaches here _need_ scaled-down or sampled data
    dep_var = "y_yes"
    indep_vars = ["age", "cons.price.idx"]
    x_train, x_test, y_train, y_test = model.do_test_train_split(
        df=data, indep_vars=indep_vars, dep_var=dep_var, test_size=0.30
    )
    # model_types = ["global_naive", "sm_linear", "sk_linear"]
    model_types = ["logistic"]
    trained_models = fit_several_models(
        x_train=x_train, y_train=y_train, model_types=model_types
    )
    score_several_models(x_test=x_test, y_test=y_test, models_to_test=trained_models)

    # ensembling?
    # evaluate ensemble results?

    print(
        "Modeling complete. Check that full dataset was used if intended. Consider further improvements."
    )


def load_and_clean(
    csv: str, do_sample: bool = False, do_seed: bool = True
) -> pd.DataFrame:
    """
    Load csv-format data and clean it using static code (for now)
    :param csv: string specifying location of csv file (e.g., 'data/sundae.csv')
    :param do_sample: True/False value which controls whether to sample or use all raw data
    :param do_seed: True/False value which controls whether to use a fixed seed for random selection
    :return: returns a pandas dataframe containing the cleaned result

    - could be refactored to separate concerns of loading and cleaning
    - a more complex project would benefit from a dedicated data validation setup
    """
    data = pd.read_csv(
        csv, delimiter=";"
    )  # may need to be refactored if the data isn't in csv format. Alternatively, convert it.

    if do_sample:  # for dev. n.b.: "row 0" here != "row 0" in original.
        if do_seed:
            data = data.sample(
                n=c.SAMPLE, replace=True, random_state=c.SEED
            ).reset_index(drop=True)
        else:
            data = data.sample(
                n=c.SAMPLE,
                replace=True,
            ).reset_index(drop=True)

    # data cleaning/prep:  (iterate w/ EDA)
    data.columns = data.columns.str.lower()  # all column names to lower case
    # ...
    # cols to dummy-- i.e., onehot encode, etc.
    # cols_to_dummy: list = []  # !!!
    # data = pd.get_dummies(data, columns=cols_to_dummy)
    # cols_to_drop: list = []  # !!!
    # data = data.drop(labels=cols_to_drop, axis=1)
    # cols_w_nan: list = []  # !!!
    # for nancol in cols_w_nan:
    #     data[nancol].fillna(value=data[nancol].mean(), inplace=True)
    data = pd.get_dummies(data, columns=["y"])
    # columns = data.columns
    data = data[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'y_yes']]
    # data.to_csv('data/sundae-cleaned.csv')
    # ...
    data = data.dropna()  # crude... but may catch something otherwise missed
    return data


def eda(data: pd.DataFrame) -> None:
    """
    main execution loop-- "do exploratory data analysis"
    :param data: pandas dataframe of data we're exploring, analyzing
    :return: nothing (it writes to standard out, or logs if logs are later implemented)
    """
    eda_tools.verbose_sniff(data=data)
    eda_tools.col_deepdive(
        data=data
    )  # consider passing a slice of data (certain columns of interest)
    eda_tools.do_correl_heatmap(data=data)
    eda_tools.do_small_multiples(data=data, y_col=c.DEP_VAR_COL_NAME)
    print("finished with this round of EDA")


def fit_model(x_train: pd.DataFrame, y_train: pd.Series, model_type: str) -> dict:
    """
    Using the provided training data
    :param x_train:
    :param y_train:
    :param model_type:
    :return: returns a fit model as a dictionary like '{model_type: fit_model,}'

    currently supported options for model_type:
    - global_naive
    - sm_linear
    - sk_linear

    - n.b.: python now has a switch statement for v >= 3.10; for now, use the more traditional if..elif..else
    """
    result: dict
    if model_type == "global_naive":
        result = model.do_global_naive(
            y=y_train
        )  # as it is global naive, it doesn't take xs as an input
    elif model_type == "sm_linear":
        result = model.do_statsmodels_lm(xs=x_train, y=y_train)
    elif model_type == "sk_linear":
        result = model.do_lin_reg(xs=x_train, y=y_train)
    elif model_type == "logistic":
        result = model.do_skl_logit(xs=x_train, y=y_train, train_on=["age", "cons.price.idx"])
    else:
        raise ValueError(
            "Unexpected value for model_type. Value given: ", str(model_type)
        )
    return result


def fit_several_models(
    x_train: pd.DataFrame, y_train: pd.Series, model_types: list
) -> dict:
    """
    apply fit_model over a list of supported model types, collect and return the results as a dictionary
    :param x_train: matrix of independent variables as a pandas dataframe
    :param y_train: corresponding dependent variables as a pandas series
    :param model_types: list of model types to fit
    :return: collection of fitted models as a dictionary like '{model_type1: fit_model1,..., model_typeN: fit_modelN,}'
    """
    results: dict = {}
    for model_type in model_types:
        result = fit_model(x_train=x_train, y_train=y_train, model_type=model_type)
        results.update(result)
    return results


def score_model(x_test: pd.DataFrame, y_test: pd.Series, model_to_test: dict) -> None:
    """
    Using the provided test data, evaluate the performance of the provided model
    :param x_test: matrix of independent variables as a pandas dataframe
    :param y_test: array of dependent variable as a pandas series
    :param model_to_test: dictionary of {model_type: fitted_model,} pair
    :return: nothing-- function called prints to STDOUT

    possible future work:  return a dict for use in ensembling the models (e.g., weighted average approach)
    """
    for model_type, fit_model in model_to_test.items():
        if model_type == "global_naive":
            model.score_global_naive(y_test=y_test, fit_model=fit_model)
        elif model_type == "sm_linear":
            model.score_sm_linear_fit(x_test=x_test, y_test=y_test, fit_model=fit_model)
        elif model_type == "sk_linear":
            model.score_sk_linear_fit(x_test=x_test, y_test=y_test, fit_model=fit_model)
        elif model_type == "logistic":
            model.score_sk_linear_fit(x_test=x_test, y_test=y_test, fit_model=fit_model)
        else:
            raise ValueError(str(model_type), "is not yet implemented")


def score_several_models(
    x_test: pd.DataFrame, y_test: pd.Series, models_to_test: dict
) -> None:
    """
    Using the provided test data, evaluate the performance of several models and print the results
    :param x_test: matrix of independent variables as a pandas dataframe
    :param y_test: array of dependent variable as a pandas series
    :param models_to_test: dictionary of {model_type: fitted_model,} pairs
    :return: nothing-- function it calls prints to STDOUT

    possible future work:  return a dict for use in ensembling the models (e.g., weighted average approach)
    """
    for k, v in models_to_test.items():
        model_to_test: dict = {
            k: v,
        }
        score_model(x_test=x_test, y_test=y_test, model_to_test=model_to_test)


if __name__ == "__main__":
    main(do_sample=True, do_eda=False)
