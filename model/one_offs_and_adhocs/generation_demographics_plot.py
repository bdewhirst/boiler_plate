# import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# various expedient choices were made (no type hints, very simple structure, etc.)

# read in the (column-truncated) raw data
raw_data: pd.DataFrame = pd.read_csv(
    "data/nc-est2021-alldata-r-file06.csv"
)  # skip typing hints after this

# clean up, add age, generational cohort
lower_case_cols = raw_data.columns.str.lower()
raw_data.columns = lower_case_cols
data: pd.DataFrame = raw_data.copy()
del raw_data, lower_case_cols
# data["birth_year"] = 2022 - data["age"]

data_pivot = data.pivot(index="age", columns="gen_label")

# following seems to be the minimalist solution:
# data_pivot.plot(); plt.show()

# create a cleaner, more aesthetic plot:
data_pivot.columns = data_pivot.columns.to_flat_index()
# print(data_pivot.columns)  # based on this, relabel-- risky if something changes
data_pivot.columns = [
    "Boomer",
    "Gen X",
    "Gen Z",
    "Gen Z*",
    "Greatest",
    "Millenial",
    "Silent",
]

final_col_order = [
    "Gen Z*",
    "Gen Z",
    "Millenial",
    "Gen X",
    "Boomer",
    "Silent",
    "Greatest",
]
data_pivot = data_pivot[final_col_order]

plt.figure()
plt.plot(data_pivot)
plt.xlim([0, 100])
plt.ylim([0, 5e6])
plt.legend(final_col_order)
plt.xlabel("age")
plt.ylabel("population")
plt.title("US Census Population Estimate by age as of Dec. 2022, grouped by generation")

plt.show()
# aesthetics can be improved with iteration, specifying sizes, font sizes, etc.
