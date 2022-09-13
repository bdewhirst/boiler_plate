import pandas as pd
import numpy as np

import utils.constants as c
from utils import eda_tools
from model import model


def main(do_sample: bool = False) -> None:
    """
    This function
    :return: returns nothing; output to STDOUT for now, and/or direct inspection via breakpoints, etc.
    """
    # for reproducible iteration, set seed
    np.random.seed(c.SEED)
    # when using pandas's sample method, df.sample(..., random_state= SEED)
    data = load_and_clean(csv=c.RAWCSV, do_sample=do_sample)
    eda(data=data)
    # ...
    pass


def load_and_clean(csv: str, do_sample: bool = False) -> pd.DataFrame:
    """
    Load csv-format data and clean it using static code (for now)
    :param csv: string specifying location of csv file (e.g., 'data/sundae.csv')
    :return: returns a pandas dataframe containing the cleaned result
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


def do_modeling(data: pd.DataFrame) -> None:
    pass


if __name__ == "__main__":
    main(do_sample=True)  # remember to disable later; arg. defaults to false
