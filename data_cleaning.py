import pandas as pd 
from sklearn.model_selection import train_test_split

# loading csv file into dataframe
df = pd.read_csv('fetal_health.csv')

# only keeping relevant columns 
col_keeping = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'fetal_health']
df = df[col_keeping]
print(df.columns) 

# replacing any 'fetal_heath' of 3 to 2
# we are only have two classification groups to make the models simpler
df['fetal_health'] = df['fetal_health'].replace(3.0, 2.0)

# verify that the replacement worked
print("Fetal_health unique values:", df['fetal_health'].unique())

# train test splitting the data 
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# saving training set to csv
train_df.to_csv("./Data/train_set.csv", index=False)

# saving test set to csv
test_df.to_csv("./Data/test_set.csv", index=False)

print("Training and test sets saved as csv files.") 
