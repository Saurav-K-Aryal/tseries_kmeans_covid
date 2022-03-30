import numpy as np
# from pinguoin import multivariate_normality
import pandas as pd
import pingouin as pg
from pingouin import multivariate_normality
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/combined_dfs.csv')
# print(df.dtypes)
df.drop(columns=['Unnamed: 0'], inplace=True)

# print(df.head())

col_names = ['average_temperature_celsius','comorbidity_mortality_rate','cumulative_persons_fully_vaccinated','cumulative_persons_vaccinated','cumulative_tested','diabetes_prevalence','elevation_m','gdp_per_capita_usd','gdp_usd','human_capital_index','latitude','population_density','smoking_prevalence']

features = df[col_names]

scaler = MinMaxScaler()

df[col_names] = scaler.fit_transform(features.values)


# print(df.head())
print(df.dtypes)

df = df.astype({'labels' : object})

print(df.dtypes)
# print(df.columns)
# print(len(df.columns))

# arr = ['t{0}'.format(x) for x in range(0, 600)]



grouped = df.groupby("labels")
for name, group in grouped:
    print(name, pg.multivariate_normality(group, alpha=.05)) 




# print(p)
