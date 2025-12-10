import pandas as pd 
from sklearn.model_selection import train_test_split

# Loading csv file into dataframe
df = pd.read_csv('./Data/fetal_health.csv')

# Only keeping relevant columns 
col_keeping = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'fetal_health']
df = df[col_keeping]
print(df.columns) 

# Replacing any 'fetal_heath' of 3 to 2
# We are only have two classification groups to make the models simpler
df['fetal_health'] = df['fetal_health'].replace(3.0, 2.0)

# Verify that the replacement worked
print("Fetal_health unique values:", df['fetal_health'].unique())

# Renaming `baseline value` to `baseline_value` for consistency
df.rename(columns={'baseline value': 'baseline_value'}, inplace=True)
# Verifying the column rename
print("Columns after renaming:", df.columns)

# Save cleaned df to csv file 
df.to_csv('./Data/cleaned_fetal_health.csv', index=False)

# Train test splitting the data 
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Saving training set to csv
train_df.to_csv("./Data/train_set.csv", index=False)

# Saving test set to csv
test_df.to_csv("./Data/test_set.csv", index=False)

print("Training and test sets saved as csv files.") 
