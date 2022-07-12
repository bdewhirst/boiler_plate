import pandas as pd


def csv_loader(source_file: str, source_folder: str="data/") -> pd.DataFrame:
    """
    Read specified file from /data folder
    :param source_file: filename as string, with extension (e.g. '.csv')
    :param source_folder: source subfolder as string, defaulting to /data
    :return: pandas dataframe from indicated CSV
    """
    path_to_load: str= "".join([source_folder, source_file])
    data = pd.read_csv(path_to_load)
    return data


def sniff_frame(df: pd.DataFrame) -> None:
    """
    run some standard sniff tests on the indicated dataframe
    :param df: pandas dataframe
    :return: nothing; function outputs to terminal
    """
    print(df.columns)
    print(df.shape)
    print(df.describe())  # we'd need to tweak the width


def titanic_main(source_file:str ="data/titanic.csv") -> None:
    """
    Poke at the famous titanic dataset.

    n.b.: 'train' and 'titanic' aren't the same (length or columns)
    """
    titanic: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame

    agenda: list = ["titanic", "train", "test"]
    for name in agenda:
        print(name)
        df = csv_loader(source_file=''.join([name, ".csv"]))
        sniff_frame(df=df)


    # future work:
    #  play with date and memory...
    #  test/train split of titanic.csv
    #  throw it all into a pipeline class or something
    #  retrieve from database instead of csv; cache too.


if __name__ == "__main__":
    """
    "test" and "train" are from https://www.kaggle.com/c/titanic/data
    """
    titanic_main()
