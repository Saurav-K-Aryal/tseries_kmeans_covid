import pandas as pd

df1 = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/Deaths_analysis/deaths_out.csv')

df2 = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/Deaths_analysis/nikesh_data_deaths.csv')

print(df1.shape)
print(df2.shape)

del df2['labels']
new_df = pd.merge(df1, df2 , on = 'iso_a3')

# df1.merge(df2, how = 'inner', on = 'iso_a3')
# print(new_df[['labels']])
# print(new_df.shape)
# print(new_df.columns)

col_names = ['iso_a3','labels','average_temperature_celsius','comorbidity_mortality_rate','average_new_fully_vaccinated_per_day_per_1000','average_new_vaccinated_per_day_per_1000','average_new_tested_per_day_per_1000','diabetes_prevalence','gdp_per_capita_usd','gdp_usd','human_capital_index','latitude','population_density','smoking_prevalence', 'total_cases_divided_by_population','total_deaths_divided_by_population','population']


final_df = new_df[col_names]
# print(new_df[['iso_a3', 'labels']].head(25))

final_df.to_csv('nikeshData_+_newLabels.csv')
print(final_df.shape)
# print(final_df.head(25))

