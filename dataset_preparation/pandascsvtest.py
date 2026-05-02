
import pandas as pd
df = pd.read_csv(r"D:\csv\distraction\IR\distractionIR1.csv", sep=";")
print(df[["yaw", "pitch", "roll"]].head(20))
