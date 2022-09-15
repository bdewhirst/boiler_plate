import os
import pandas as pd


os.chdir("C:/Users/Brian/PycharmProjects/boiler_plate/")

df = pd.read_...('data/sundae-...')
print(df.head())
df.to_csv('data/sundae-raw.csv', index=False)

# this is just a sketch of code which might be used to convert data
# from another supported data type (e.g. JSON) into a csv file"