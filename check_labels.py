import pandas as pd

df = pd.read_csv("App_Review_Labelled.csv")
print(df['sentiment_label'].unique())
