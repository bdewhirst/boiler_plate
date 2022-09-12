import pandas as pd
import numpy as np


# CONSTANTS
RAWCSV: str = "data/sundae-raw.csv"
SAMPCSV: str = "data/sundae-sample.csv"
SEED: int = 202209131701


def load_and_clean(csv: str) -> pd.DataFrame:
    """
    Load csv-format data and clean it using static code (for now)
    :param csv: string specifying location of csv file (e.g., 'data/sundae.csv')
    :return: returns a pandas dataframe containing the cleaned result
    """
    data = pd.read_csv(csv)  # various parameters such as specifying index column, etc.

    # data cleaning:
    # ...
    # ...
    return data


def main() -> None:
    """
    This function
    :return: returns nothing; output to STDOUT for now, and/or direct inspection via breakpoints, etc.
    """
    # for reproducible iteration, set seed
    np.random.seed(SEED)
    data = load_and_clean(csv=RAWCSV)


if __name__ == "__main__":

    main()
