import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


from abc import ABC, abstractmethod
from utils.common_utils import say_tp_message


class AbstractEDASniff(ABC):
    @abstractmethod
    def __init__(self, data):
        raise NotImplementedError("EDA must implement data initialization")

    @abstractmethod
    def sniff(self):
        raise NotImplementedError("EDA must implement sniff tests")


class AbstractEDAPlot(ABC):
    @abstractmethod
    def __init__(self, data):
        raise NotImplementedError("EDA must implement data initialization")

    @abstractmethod
    def plot(self):
        raise NotImplementedError("EDA must implement plots as part of data analysis")


class SimpleEDASniff(AbstractEDASniff):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def _col_deepdive(self) -> None:
        """
        iterate over each column in the dataframe and output the number and values of unique entries
        :param data:  pandas dataframe to be examined
        :return: nothing-- currently prints to STDOUT
        """
        print("scrutinizing unique values for each column")
        for col in self.data.columns:
            print(col)
            print(self.data[[col]].nunique())
            print(sorted(self.data[col].unique()))

    def _verbose_sniff(self) -> None:
        """
        Provide descriptive information about a given dataframe
        :param data: pandas dataframe to be examined-- often from a call of a native pandas DataFrame method
        :return: nothing-- currently prints to STDOUT
        """
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.width", 2000)
        pd.set_option("display.max_columns", 10)
        data = self.data

        say_tp_message(desc="data info is \r\n", strdata="")
        data.info()  # prints itself
        say_tp_message(desc="data shape is ", strdata=str(data.shape))
        say_tp_message(
            desc="missing data by column: \r\n", strdata=str(data.isnull().sum())
        )
        say_tp_message(desc="data head is: \r\n", strdata=str(data.head()))
        say_tp_message(desc="pandas describe is: \r\n", strdata=str(data.describe()))
        say_tp_message(desc="data types are: \r\n", strdata=str(data.dtypes))
        print("for these next two, remember we're sampling with replacement to start")
        say_tp_message(
            desc="unique vals by column are: \r\n", strdata=str(data.nunique())
        )
        say_tp_message(
            desc="check for duplicate rows: \r\n", strdata=str(data.duplicated().sum())
        )

    def sniff(self):
        self._col_deepdive()
        self._verbose_sniff()


class SimpleEDAPlot(AbstractEDAPlot):
    def __init__(self, data: pd.DataFrame, dependent_var):
        self.data = data
        self.dependent_var = dependent_var

    def _do_correl_matrix(self) -> None:
        """
        print correlation matrix of supplied dataframe
        :param data: pandas dataframe containing data of interest
        :return: nothing, but should print to STDOUT
        """
        message = "".join(
            ["correlation matrix is: \r\n", str(self.data.corr().round(3))]
        )
        print(message)

    def _do_correl_heatmap(self) -> None:
        """
        Display seaborn heatmap of correlation matrix of provided data
        :param data: pandas dataframe containing the data of interest
        :return: returns nothing, but plots should open during runtime

        ref: https://seaborn.pydata.org/generated/seaborn.heatmap.html
        """
        self._do_correl_matrix()
        rounded_corr_data = self.data.corr().round(3)
        to_plot = sb.heatmap(rounded_corr_data, annot=True)
        plt.show()

    def _do_small_multiples(self) -> None:
        """
        Generate small multiples of pairwise comparisons of columns in the provided data
        :param data: pandas dataframe containing the data of interest
        :param dependent_var: string specifying column of dependent variable
        :return: returns nothing, but a plot of small multiples should open during runtime

        future work: this could be cleaner
        """
        cells = len(self.data.columns)
        dep_min = self.data[self.dependent_var].min()
        dep_max = self.data[self.dependent_var].max()
        num: int = 0
        for column in self.data:
            num += 1
            nrows = 3
            ncols = int(cells / 3) + 1
            plt.subplot(nrows, ncols, num)  # nrows, ncols, index
            # plt.plot(data[y_col], data[column], label=column)
            plt.scatter(
                y=self.data[self.dependent_var],
                x=self.data[column],
                marker=",",
                label=column,
                alpha=0.7,
            )
            indep_min = self.data[column].min()
            indep_max = self.data[column].max()
            plt.legend(loc=2)
            plt.xlim(indep_min, indep_max)
            plt.ylim(dep_min, dep_max)
        plt.show()

    def plot(self):
        self._do_correl_heatmap()
        self._do_small_multiples()
