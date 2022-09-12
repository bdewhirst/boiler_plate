import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def say_tp_message(desc: str, strdata: str) -> None:
    """
    Print a simple (two part) message consisting of two strings
    :param desc: string describing what data will be output next:
    :param strdata: meaningful data represented as a string
    :return: returns nothing, but does print to STDOUT
    """
    message = "".join([desc, strdata])
    print(message)


def verbose_sniff(data: pd.DataFrame) -> None:
    """
    Provide descriptive information about a given dataframe
    :param data: pandas dataframe to be examined-- often from a call of a native pandas DataFrame method
    :return: nothing-- currently prints to STDOUT
    """
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_columns", 10)

    say_tp_message(desc="data info is \r\n", strdata=str(data.info()))
    say_tp_message(desc="data shape is ", strdata=str(data.shape))
    say_tp_message(
        desc="missing data by column: \r\n", strdata=str(data.isnull().sum())
    )
    say_tp_message(desc="data head is: \r\n", strdata=str(data.head()))
    say_tp_message(desc="pandas describe is: \r\n", strdata=str(data.describe()))
    say_tp_message(desc="data types are: \r\n", strdata=str(data.dtypes))
    say_tp_message(desc="data types are: \r\n", strdata=str(data.dtypes))


def do_correl_matrix(data: pd.DataFrame) -> None:
    """
    print correlation matrix of supplied dataframe
    :param data: pandas dataframe containing data of interest
    :return: nothing, but should print to STDOUT
    """
    message = "".join(["correlation matrix is: \r\n", str(data.corr().round(3))])
    print(message)


def do_correl_heatmap(data: pd.DataFrame) -> None:
    """
    Display seaborn heatmap of correlation matrix of provided data
    :param data: pandas dataframe containing the data of interest
    :return: returns nothing, but plots should open during runtime

    ref: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    """
    do_correl_matrix(data=data)
    rounded_corr_data = data.corr().round(3)
    to_plot = sb.heatmap(rounded_corr_data, annot=True)
    plt.show()


def do_small_multiples(data: pd.DataFrame, y_col: str) -> None:
    """
    Generate small multiples of pairwise comparisons of columns in the provided data
    :param data: pandas dataframe containing the data of interest
    :param y_col: string specifying column of dependent variable
    :return: returns nothing, but a plot of small multiples should open during runtime

    future work: this could be cleaner
    """
    cells = len(data.columns)
    dep_min = data[y_col].min()
    dep_max = data[y_col].max()
    num: int = 0
    for column in data:
        num += 1
        nrows = 3
        ncols = int(cells / 3) + 1
        plt.subplot(nrows, ncols, num)  # nrows, ncols, index
        # plt.plot(data[y_col], data[column], label=column)
        plt.scatter(y=data[y_col], x=data[column], marker=",", label=column, alpha=0.7)
        indep_min = data[column].min()
        indep_max = data[column].max()
        plt.legend(loc=2)
        plt.xlim(indep_min, indep_max)
        plt.ylim(dep_min, dep_max)
    plt.show()
