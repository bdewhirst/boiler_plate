import pandas as pd


# CONSTANTS
CSV = "data/sundae.csv"


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
    data = load_and_clean(csv=CSV)


if __name__ == "__main__":
    main()
