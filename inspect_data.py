import pandas as pd

print("Inspecting App_Review_Labelled.csv")
df_labelled = pd.read_csv("App_Review_Labelled.csv")
print(df_labelled.head())
print(df_labelled.info())

print("\n" + "="*50 + "\n")

print("Inspecting Review_Dataset.csv")
df_review = pd.read_csv("Review_Dataset.csv")
print(df_review.head())
print(df_review.info())
