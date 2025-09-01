import pandas as pd

data = pd.read_csv("./test/example.csv")

print(data.iloc[1, 3])