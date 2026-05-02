import pandas as pd

file_path = r"D:\csvcleaned\drowsiness\RGB\drowsinessrgb1.xlsx"
df = pd.read_excel(file_path)

print(df.head(15))
print(df.isnull().sum())