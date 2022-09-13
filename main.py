import pandas as pd
import numpy as np

import utils.constants as c
from utils import eda_tools
from model import model


def main(do_sample: bool = False, do_eda: bool = False) -> None:
    """
    Main function which is called when the model framework is run
    :return: returns nothing; output to STDOUT
    """
    # for reproducible iteration, set seed
    np.random.seed(c.SEED)
    data = load_and_clean(csv=c.RAWCSV, do_sample=do_sample)

    if do_eda:
        eda(data=data)  # n.b.: some approaches here _need_ scaled-down or sampled data
    dep_var = "a"
    indep_vars = ["b", "c"]
    x_train, x_test, y_train, y_test = model.do_test_train_split(
        df=data, indep_vars=indep_vars, dep_var=dep_var, test_size=0.30
    )

    model_types = ["global_naive", "sm_linear", "sk_linear"]
    rs = fit_several_models(x_train=x_train, y_train=y_train, model_types=model_types)

    # score_model(...)
    # score_several_models(...)
    # display_scores(...)

    # print(
    #     "modeling complete-- check that full dataset was used if intended-- consider further feature selection"
    # )


def load_and_clean(csv: str, do_sample: bool = False) -> pd.DataFrame:
    """
    Load csv-format data and clean it using static code (for now)
    :param csv: string specifying location of csv file (e.g., 'data/sundae.csv')
    :return: returns a pandas dataframe containing the cleaned result

    - could be refactored to separate concerns of loading and cleaning
    - a more complex project would benefit from a dedicated data validation setup
    """
    data = pd.read_csv(
        csv
    )  # may need to be refactored if the data isn't in csv format. Alternatively, convert it.

    if do_sample:  # for dev. n.b.: "row 0" here != "row 0" in original.
        data = data.sample(n=c.SAMPLE, replace=True, random_state=c.SEED).reset_index(
            drop=True
        )

    # data cleaning/prep:  (iterate w/ EDA)
    data.columns = data.columns.str.lower()  # all column names to lower case
    # ...
    # cols to dummy-- i.e., onehot encode, etc.
    cols_to_dummy: list = []  # !!!
    data = pd.get_dummies(data, columns=cols_to_dummy)
    cols_to_drop: list = []  # !!!
    data = data.drop(labels=cols_to_drop, axis=1)
    cols_w_nan: list = []  # !!!
    for nancol in cols_w_nan:
        data[nancol].fillna(value=data[nancol].mean(), inplace=True)
    # ...
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
        result = model.fit_lin_reg(xs=x_train, y=y_train)
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


def score_model():
    pass


if __name__ == "__main__":
    main(do_sample=True, do_eda=False)
