import os
import pandas as pd

# loading csv with pandas
pd.set_option("display.max_columns", 7)
df = pd.read_csv("auto-mpg.csv")
print(df[0:5])

# display stats from dataframe
stats_df = df.select_dtypes(include=["int", "float"])
headers = list(stats_df.columns.values)
fields = []
for field in headers:
    fields.append({
        "name": field,
        "mean": stats_df[field].mean(),
        "var": stats_df[field].var(),
        "sdev": stats_df[field].std()
    })
for field in fields:
    print(field)