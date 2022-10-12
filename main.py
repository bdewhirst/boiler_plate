import pandas as pd
import numpy as np

import utils.constants as c
from utils.data_utils import load_and_clean
from eda.eda_tools import SimpleEDASniff, SimpleEDAPlot
from model import model


def main(do_sample: bool = False, do_eda: bool = False, do_seed: bool = True) -> None:
    """
    Main function which is called when the model framework is run
    :param do_sample: True/False value which controls whether raw inputs are sampled or complete
    :param do_eda: True/False value which controls whether to run semi-automated exploratory data analysis
    :param do_seed: True/False value which controls whether to set a seed for
    :return: returns nothing; outputs to STDOUT
    """
    # for reproducible iteration, set seed
    dep_var = c.DEP_VAR_COL_NAME
    indep_vars = c.INDEP_VAR_COL_NAMES

    if do_seed:
        np.random.seed(c.SEED)
    data = load_and_clean(csv=c.RAWCSV, do_sample=do_sample, do_seed=do_seed)

    if do_eda:
        SimpleEDASniff(data=data).sniff()
        SimpleEDAPlot(data=data, dependent_var=dep_var).plot()
        print("finished with this round of EDA")

    trim_data = model.SimpleModelPrep(
        data=data, dep_var=dep_var, indep_vars=indep_vars, test_size=0.30
    )
    x_train, x_test, y_train, y_test = trim_data.test_train_split()

    model_classes = c.SUPPORTED_MODEL_CLASSES  # i.e. {"linear": model.LinearModel,...}

    for model_type, model_class in model_classes.items():
        #
        this_model = model_class()
        print(f"fitting {model_type} on training data")
        this_model.fit(indep_data=x_train, dep_data=y_train)
        print(f"applying {model_type} on test data")
        this_model.apply(indep_data=x_test)
        print(f"scoring {model_type}")
        this_model.score(true_data=y_test)


if __name__ == "__main__":
    main(do_sample=c.DO_SAMPLE, do_eda=c.DO_EDA)
    print(
        """
        Modeling complete. Check that full dataset was used if intended. Consider further improvements.
        """
    )
