import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency  # calc chi2 stats and p-value
from scipy.stats import chi2  # find critical value given acceptance criteria


def data_bootstrap() -> tuple:
    """
    For development purposes, load and prep data used in Z. Luna's example
    :return: return tuple of arrays of values of the two mailers ('a' and 'b')

    note: I have chosen to compare only Mailer1 and Mailer2-- I'll be discarding the control values for now.
    """
    campaign_data = pd.read_excel("data/grocery_database.xlsx", sheet_name="campaign_data")
    # n.b.: values of "mailer_type":
    print("unique values of mailer type are", str(campaign_data.mailer_type.unique()))
    relevant_data = campaign_data[["mailer_type", "signup_flag"]].copy()
    del campaign_data
    relevant_data = relevant_data[(relevant_data.mailer_type.isin(["Mailer1", "Mailer2"]))].copy()
    relevant_pivot = relevant_data.pivot(columns="mailer_type", values="signup_flag")

    # now that it is pivoted, values where Mailer1 is present are null for Mailer2 column and vice versa
    # (in a more general situation, check for missing data closer to reading it in)
    a = relevant_pivot.Mailer1.dropna()
    b = relevant_pivot.Mailer2.dropna()
    return (a, b)


def do_ab_test(
    a,
    b,
    acceptance_criteria: float= 0.05,
) -> tuple:
    """
    Implement Student's t-test-- What is the likelihood that the null hypothesis (`mean of a` == `mean of b`) is True.

    :param a: first of two arrays of values to compare
    :param b: second of two arrays of values to compare
    :param acceptance_criteria: threshold to determine whether null hypothesis has been falsified (typically 0.05)
    :return: two-value tuple of probability as a float and whether null hypothesis has been falsified as a boolian

    other considerations for manual consideration and/or future enhancements:
    - non-normal distribution(s) of sample(s), etc.
    - 'retesting' on the same data-- try not to discover false positives (avoid accidental p-hacking/ assump. violation)

    ref: https://github.com/ZL63388/grocery-signup-rate (I'll wind up heavily changing it, but I started with this)
    """
    # observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values
    print(a.head())  # tmp
    print(b.head())  # tmp
    pass


if __name__ == "__main__":
    a, b = data_bootstrap()
    do_ab_test(a=a, b=b, acceptance_criteria='TBD')
