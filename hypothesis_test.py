import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency  # calc chi2 stats and p-value
from scipy.stats import chi2  # find critical value given acceptance criteria
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil


def simple_agg_helper(vector: pd.Series) -> tuple:
    """
    Given a one-column dataframe, return a tuple of count, sum, and mean of that column

    :param vector: pandas series (1d array)
    :return: a tuple of count, sum, and mean of that column
    """
    vector = vector.dropna().copy()  # just in case

    if type(vector) != pd.Series:
        raise ValueError
    return (vector.count(), vector.sum(), vector.sum() / vector.count())


def data_bootstrap() -> tuple:
    """
    For development purposes, load and prep data used in Z. Luna's example
    :return: return tuple of arrays of values of the two mailers ('a' and 'b')

    note: I have chosen to compare only Mailer1 and Mailer2-- I'll be discarding the control values for now.
    """
    campaign_data = pd.read_excel(
        "data/grocery_database.xlsx", sheet_name="campaign_data"
    )
    observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values
    # n.b.: values of "mailer_type":
    # print("unique values of mailer type are", str(campaign_data.mailer_type.unique()))
    relevant_data = campaign_data[["mailer_type", "signup_flag"]].copy()
    del campaign_data
    relevant_data = relevant_data[
        (relevant_data.mailer_type.isin(["Mailer1", "Mailer2"]))
    ].copy()
    relevant_pivot = relevant_data.pivot(columns="mailer_type", values="signup_flag")

    # now that it is pivoted, values where Mailer1 is present are null for Mailer2 column and vice versa
    # (in a more general situation, check for missing data closer to reading it in)
    a = relevant_pivot.Mailer1.dropna()
    b = relevant_pivot.Mailer2.dropna()
    return (a, b, observed_values)


def do_ab_test(
    a,
    b,
    observed_values,
    acceptance_criteria: float = 0.05,
) -> None:
    """
    Implement Student's t-test-- What is the likelihood that the null hypothesis (`mean of a` == `mean of b`) is True.

    :param a: first of two arrays of values to compare
    :param b: second of two arrays of values to compare
    :param observed_values: crosstab of mailer type and signup flag based on raw data
    :param acceptance_criteria: threshold to determine whether null hypothesis has been falsified (typically 0.05)
    :return: nothing

    other considerations for manual consideration and/or future enhancements:
    - non-normal distribution(s) of sample(s), etc.
    - 'retesting' on the same data-- try not to discover false positives (avoid accidental p-hacking/ assump. violation)

    design of experiments is necessary, but out of scope. (future work: necessary sample size calculator)

    ref: https://github.com/ZL63388/grocery-signup-rate (I'll wind up heavily changing it, but I started with this)

    (turns out this is too sparse on details to be particularly useful as-is)
    """
    a_count, a_sum, a_mean = simple_agg_helper(vector=a)
    b_count, b_sum, b_mean = simple_agg_helper(vector=b)

    # a_rate = a_mean
    # b_rate = b_mean

    # calculate expected frequencies & chi square statistic
    chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction=False)
    print(chi2_statistic, p_value)

    # find the critical value for our test
    critical_value = chi2.ppf(1 - acceptance_criteria, dof)
    print(critical_value)

    print(1)
    pass


# def format_stuff():
#     plt.style.use('seaborn-whitegrid')
#     font = {'family': 'Helvetica',
#             'weight': 'bold',
#             'size': 14}
#
#     mpl.rc('font', **font)


def tds_bootstrap(est_req_sample: int, random_state: int = 22) -> tuple:
    """
    ref: https://towardsdatascience.com/ab-testing-with-python-e5964dd66143

    :param est_req_sample: estimated necessary sample size to see estimated effect
    :param random_state: any integer-- to set seed
    :return: tuple of dataframes of control and sample

    note: currently, this expects specific (hard-coded) data as an input in the /data directory
    """
    df = pd.read_csv("data/ab_data.csv")
    # print(df.head())
    # print(df.info())
    # print(pd.crosstab(df['group'], df['landing_page']).head(10))
    session_counts = df["user_id"].value_counts(ascending=False)
    multi_users = session_counts[session_counts > 1].count()

    print(f"There are {multi_users} users that appear multiple times in the dataset")
    users_to_drop = session_counts[session_counts > 1].index

    df = df[~df["user_id"].isin(users_to_drop)]
    print(f"The updated dataset now has {df.shape[0]} entries")
    control_sample = df[df["group"] == "control"].sample(
        n=est_req_sample, random_state=random_state
    )
    treatment_sample = df[df["group"] == "treatment"].sample(
        n=est_req_sample, random_state=random_state
    )
    return control_sample, treatment_sample


def samp_calcer(
    base_rate: float, lifted_rate: float, acceptance_criteria: float = 0.05
) -> int:
    """
    :param base_rate: baseline rate of effect (e.g., "13% conversion"--> 0.13)
    :param lifted_rate: expected improved result w/ treatment (e.g., "15% conversion (13% +2%)" --> 0.15)
    :param acceptance_criteria:
    :return: returns (rounded up) int of estimated sample size
    """
    effect_size = sms.proportion_effectsize(base_rate, lifted_rate)
    required_n = sms.NormalIndPower().solve_power(
        effect_size, power=0.8, alpha=acceptance_criteria, ratio=1
    )
    required_n: int = ceil(required_n)
    return required_n


def example(est_required_sample: int = 0) -> None:
    """
    clean up and generalize, and here's a sample size calculator

    ref: https://towardsdatascience.com/ab-testing-with-python-e5964dd66143
    """
    # format_stuff()
    if est_required_sample <= 0:
        # i.e., if it has the default value (or a bad value less than or equal to 0 is passed)
        est_required_sample = samp_calcer(
            base_rate=0.13, lifted_rate=0.15
        )  # per example from reference

    control_sample, treatment_sample = tds_bootstrap(
        est_req_sample=est_required_sample, random_state=22
    )

    ab_test = pd.concat([control_sample, treatment_sample], axis=0)
    ab_test.reset_index(drop=True, inplace=True)
    # print(ab_test.head(10))
    # print(ab_test.info())
    # print(ab_test['group'].value_counts())

    conversion_rates = ab_test.groupby("group")["converted"]

    std_p = lambda x: np.std(x, ddof=0)  # Std. deviation of the proportion
    se_p = lambda x: stats.sem(
        x, ddof=0
    )  # Std. error of the proportion (std / sqrt(n))

    conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
    conversion_rates.columns = ["conversion_rate", "std_deviation", "std_error"]

    # conversion_rates.style.format('{:.3f}')  # requires another optional package (Jinja2)
    print("conversion rates are:", str(conversion_rates.round(3).head()))

    # plt.figure(figsize=(8, 6))
    # sns.barplot(x=ab_test['group'], y=ab_test['converted'], ci=False)
    # plt.ylim(0, 0.17)
    # plt.title('Conversion rate by group', pad=20)
    # plt.xlabel('Group', labelpad=15)
    # plt.ylabel('Converted (proportion)', labelpad=15);
    # plt.show()

    control_results = ab_test[ab_test["group"] == "control"]["converted"]
    treatment_results = ab_test[ab_test["group"] == "treatment"]["converted"]
    n_con = control_results.count()
    n_treat = treatment_results.count()
    successes = [control_results.sum(), treatment_results.sum()]
    nobs = [n_con, n_treat]

    z_stat, pval = proportions_ztest(successes, nobs=nobs)
    (lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(
        successes, nobs=nobs, alpha=0.05
    )

    print(f"z statistic: {z_stat:.2f}")
    print(f"p-value: {pval:.3f}")
    print(
        f"ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]"
    )  # confidence interval is useful
    print(
        f"ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]"
    )  # " " "
    print("done")


if __name__ == "__main__":
    # general reminder: the details and assumptions matter-- violate the assumptions, and the test doesn't mean much.
    # example()
    a, b, observed_values = data_bootstrap()
    do_ab_test(
        a=a,
        b=b,
        observed_values=observed_values,
    )
    # future thought notes:
    # TODO !!!  all I really need is the confidence interval, plus some logic to see whether the ranges overlap