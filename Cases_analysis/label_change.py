import pandas as pd

df1 = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/Cases_analysis/cases_out.csv')

df2 = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/Cases_analysis/new_data.csv')

print(df1.shape)
print(df2.shape)

# del df2['labels']
df2.rename(columns={'iso_3166_1_alpha_3' : 'iso_a3'}, inplace=True)
new_df = pd.merge(df1, df2 , on = 'iso_a3')

# df1.merge(df2, how = 'inner', on = 'iso_a3')
# print(new_df[['labels']])
# print(new_df.shape)
# print(new_df.columns)

col_names = ['iso_a3','labels','average_temperature_celsius', 'comorbidity_mortality_rate',
                          'average_new_fully_vaccinated_per_day_per_1000','average_new_vaccinated_per_day_per_1000',
                          'average_new_tested_per_day_per_1000','diabetes_prevalence',
                          'gdp_per_capita_usd','gdp_usd','human_capital_index',
                          'latitude','population_density','smoking_prevalence',
                          'total_cases_divided_by_population', 'total_deaths_divided_by_population','population', 'international_travel_controls', 'stay_at_home_requirements', 'testing_policy',
        'public_information_campaigns', 'vaccination_policy', 'facial_coverings', 'cancel_public_events']

final_df = new_df[col_names]
# print(new_df[['iso_a3', 'labels']].head(25))
print(final_df.shape)
final_df.to_csv('newData_+_newLabels_cases.csv')
# print(final_df.shape)
# print(final_df.head(25))

