import numpy as np
import matplotlib.pyplot as plt
# from pinguoin import multivariate_normality
import pandas as pd
import pingouin as pg
from pingouin import multivariate_normality
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df = pd.read_csv('combined_dfs.csv')
# print(df.dtypes)
df.drop(columns=['Unnamed: 0'], inplace=True)

# print(df.head())

col_names = ['average_temperature_celsius','comorbidity_mortality_rate','cumulative_persons_fully_vaccinated','cumulative_persons_vaccinated','cumulative_tested','diabetes_prevalence','elevation_m','gdp_per_capita_usd','gdp_usd','human_capital_index','latitude','population_density','smoking_prevalence']

features = df[col_names]

scaler = MinMaxScaler()

df[col_names] = scaler.fit_transform(features.values)
col_names.append('labels')
df = df[col_names]

#
print(df.head())
# print(df.dtypes)

df = df.astype({'labels' : 'category'})

# print(df.dtypes)
# print(df.columns)
# print(len(df.columns))

# arr = ['t{0}'.format(x) for x in range(0, 600)]



grouped = df.groupby("labels")
for name, group in grouped:
    # if name or name == 0:
        # group.fillna(0, inplace=True)
    print('cluster #', name)
    print('Num countries in cluster', group.shape[0])

    # since each of these 20 (except that one 19 which we ignore) or
    # higher we can claim it is normal
    # central limit theorem

    # but let's plot each to see
    group.hist(layout=(5, 3), color='blue', figsize=(32,32), grid=False)
    plt.title('Histogram plot all numeric var in cluster' + str(name))
    plt.subplots_adjust(hspace=1.0)
    plt.show()




# print(p)
