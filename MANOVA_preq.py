import numpy as np
# from pinguoin import multivariate_normality
import pandas as pd
import pingouin as pg
from pingouin import multivariate_normality
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bartlett
from scipy.stats import levene

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
# print(df.dtypes)

df = df.astype({'labels' : object})

# print(df.dtypes)
# print(df.columns)
# print(len(df.columns))

# arr = ['t{0}'.format(x) for x in range(0, 600)]

# grouped = df.groupby("labels")
'''
    # for name, group in grouped:
    #     print(name, pg.multivariate_normality(group, alpha=.05)) 

'''

grouped = df.groupby("labels")

lst = grouped.apply(pd.Series.tolist).tolist()
kargs = []


for outermost in lst:
    for second_lyr in outermost:
        # print(second_lyr)
        second_lyr.pop(0)
        second_lyr.pop(0)
        del second_lyr[-2]
        # del second_lyr[-1]
        # print(second_lyr)
        kargs.append(second_lyr)

args = kargs
# print(args)


stat, p = bartlett(*args)

# print(p)
stat_1, l = levene(*args)

print(stat, p)
print(stat_1, l)
