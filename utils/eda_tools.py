import pandas as pd
import numpy as np


def verbose_sniff(data: pd.DataFrame) -> None:
    """
    Provide descriptive information about a given dataframe
    :param data: pandas dataframe to be examined
    :return: nothing-- currently prints to STDOUT
    """
    # still not showing full width... (pycharm + pandas issue, however pycharm does support direct inspection)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.max_columns", 10)
    print("REMINDER TO SELF: FOR FULL-WIDTH, USE THE TERMINAL!")
    message = "".join(["data shape is ", str(data.shape)])
    print(message)
    message = "".join(["missing data by column: \r\n", str(data.isnull().sum())])
    print(message)
    message = "".join(["data head is: \r\n", str(data.head())])
    print(message)
    message = "".join(["pandas describe is: \r\n", str(data.describe())])
    print(message)
    message = "".join(["data types are: \r\n", str(data.dtypes)])
    print(message)
    pass


def do_correl_matrix(data: pd.DataFrame) -> None:
    """

    :param data: pandas dataframe of
    :return: returns nothing (prints to stdout)
    """
    pass  # TODO week of 8/29
    # correlations-- table of correlation of x0, x1... xN v. y matrix (seaborne (sp.) for heatmap?)
    # correlation matrix
    """
    # using seaborn and matplotlib to do correlation matrix w/ heatmap (here, of X's)

    Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C',
                         'Embarked_S', 'Sex_male', 'IsMinor']
    X = final_train[Selected_features]

    plt.subplots(figsize=(8, 5))
    sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
    plt.show()

    # other EDA helper stuff
    prints(some_dataframe.isnull().sum())
    """