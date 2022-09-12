import pandas as pd
import numpy as np

import utils.constants as c
import utils.eda_tools as eda_tools


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
    data = pd.read_csv(csv)

    if do_sample:  # for dev. n.b.: "row 0" here != "row 0" in original.
        data = data.sample(n=c.SAMPLE, replace=True, random_state=c.SEED).reset_index(
            drop=True
        )
    # data cleaning/prep:  (iterate w/ EDA)
    # ...
    # ...
    return data


def eda(data: pd.DataFrame) -> None:
    """
    main execution loop-- "do exploratory data analysis"
    :param data: pandas dataframe of data we're exploring, analyzing
    :return: nothing (it writes to standard out, or logs if logs are later implemented)
    """
    eda_tools.verbose_sniff(data=data)
    eda_tools.do_correl_heatmap(data=data)
    eda_tools.do_small_multiples(data=data, y_col=c.DEP_VAR_COL_NAME)
    # more?


def do_modeling(data: pd.DataFrame) -> None:
    pass


if __name__ == "__main__":

    main(do_sample=True)  # remember to disable later; arg. defaults to false

    # tmp reminders
    # scrap note
    # how to introduce dummies
    # create classes at some point
