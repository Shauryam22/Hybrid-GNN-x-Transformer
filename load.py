from sklearn.model_selection import train_test_split
import pandas as pd
device = 'cuda'
df = pd.read_csv("data/cleaned_data_code_ai_human.csv" ,encoding= "utf-8-sig")
#print(df.head())
df = df.sample(frac=1, random_state=23).reset_index(drop=True)

train_df,val_df = train_test_split(df,test_size=.2)
#train_df.shape
