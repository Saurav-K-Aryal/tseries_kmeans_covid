import numpy as np
import matplotlib.pyplot as plt
# from pinguoin import multivariate_normality
import pandas as pd
import pingouin as pg
from pingouin import multivariate_normality
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bartlett
from scipy.stats import levene
from statsmodels.multivariate.manova import MANOVA
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
# print(df.head())
# print(df.dtypes)

df = df.astype({'labels' : 'category'})

# print(df.dtypes)
# print(df.columns)
# print(len(df.columns))

# arr = ['t{0}'.format(x) for x in range(0, 600)]




grouped = df

cols = ['average_temperature_celsius', 'comorbidity_mortality_rate',
       'cumulative_persons_fully_vaccinated', 'cumulative_persons_vaccinated',
       'cumulative_tested', 'diabetes_prevalence', 'elevation_m',
       'gdp_per_capita_usd', 'gdp_usd', 'human_capital_index', 'latitude',
       'population_density', 'smoking_prevalence']



# print(type(grouped))

# for name, group in grouped:
#     print(group.columns)

for l in cols:
    # subsetting the data:
    trt0 = grouped.query('labels == 0')[l].dropna()
    # print(trt0)
    trt1 = grouped.query('labels == 1')[l].dropna()
    # print(trt1)
    trt2 = grouped.query('labels == 2')[l].dropna()
    # print(trt2)
    trt3 = grouped.query('labels == 3')[l].dropna()
    trt4 = grouped.query('labels == 4')[l].dropna()
    trt5 = grouped.query('labels == 5')[l].dropna()
    trt6 = grouped.query('labels == 6')[l].dropna()

    print(len(trt0), len(trt1), len(trt2), len(trt3), len(trt4), len(trt5), len(trt6))
    # Bartlett's test in Python with SciPy:
    stat, p = bartlett(trt0, trt1, trt2, trt3, trt4, trt5, trt6)

    # Get the results:
    print(l, stat, p)
    print('-----------')


print("\n" + "####################################################################" + "\n")

for l in cols:
    # subsetting the data:
    trt0 = grouped.query('labels == 0')[l].dropna()
    # print(trt0)
    trt1 = grouped.query('labels == 1')[l].dropna()
    # print(trt1)
    trt2 = grouped.query('labels == 2')[l].dropna()
    # print(trt2)
    trt3 = grouped.query('labels == 3')[l].dropna()
    trt4 = grouped.query('labels == 4')[l].dropna()
    trt5 = grouped.query('labels == 5')[l].dropna()
    trt6 = grouped.query('labels == 6')[l].dropna()


    # Bartlett's test in Python with SciPy:
    stat, p = levene(trt0, trt1, trt2, trt3, trt4, trt5, trt6)

    # Get the results:
    print(stat, p)




grouped = df.groupby("labels")
# print(grouped['smoking_prevalence'].dtypes)
lst = grouped.apply(pd.Series.tolist).tolist()


# for name, group in grouped:
#     print(name, pg.multivariate_normality(group, alpha=.05)) 



# kargs = []


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




'''

# initial attempt of bartlett's and levene test


# for outermost in lst:
#     for second_lyr in outermost:
#         # print(second_lyr)
#         second_lyr.pop(0)
#         second_lyr.pop(0)
#         del second_lyr[-2]
#         # del second_lyr[-1]
#         # print(second_lyr)
#         kargs.append(second_lyr)

# args = kargs
# # print(args)


# stat, p = bartlett(*args)

# # print(p)
# stat_1, l = levene(*args)

# print(stat, p)
# print(stat_1, l)

'''