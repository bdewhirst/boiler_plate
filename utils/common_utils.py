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