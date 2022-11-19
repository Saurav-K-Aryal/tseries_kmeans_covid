# renaming variables to match names in csvs' from both kmeans and data_cleaning_pandas

import pandas as pd

data1 = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/grouped_mean_modes.csv')
data1.rename({'country_name' : 'country', 'iso_3166_1_alpha_3': 'iso_a3'}, axis=1, inplace=True)
print(data1.head())

#merging two csv files
# data1 = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/grouped_mean_modes.csv')
data2 = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/cases_out.csv')

new_df = pd.merge(data1,data2, on='iso_a3',how='inner')


new_df.drop(columns=['Unnamed: 0', 'country_y'], inplace=True)

print(new_df.head())
print(new_df.columns)

new_df.to_csv('combined_dfs.csv')
