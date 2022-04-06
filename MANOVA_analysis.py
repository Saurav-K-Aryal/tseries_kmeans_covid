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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
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








maov = MANOVA.from_formula('average_temperature_celsius + comorbidity_mortality_rate + cumulative_persons_fully_vaccinated + cumulative_persons_vaccinated + cumulative_tested + diabetes_prevalence + elevation_m + gdp_per_capita_usd + gdp_usd + human_capital_index + latitude + population_density + smoking_prevalence ~ labels', data = df)
print(maov.mv_test())


X = df[['average_temperature_celsius','comorbidity_mortality_rate','cumulative_persons_fully_vaccinated','cumulative_persons_vaccinated','cumulative_tested','diabetes_prevalence','elevation_m','gdp_per_capita_usd','gdp_usd','human_capital_index','latitude','population_density','smoking_prevalence']].fillna(0)
y = df["labels"].fillna(0)
# print(X.shape, y.shape)
# print(X.head)
#print(y.head)
post_hoc = lda().fit(X=X, y=y)

# get Prior probabilities of groups:

print("\n" + "The post hoc priors is: " + "\n")

print(post_hoc.priors_)

print("\n" + "The post hoc means is: " + "\n")

print(post_hoc.means_)

print("\n" + "The post hoc scaling is: " + "\n")

print(post_hoc.scalings_)

print("\n" + "The explained_variance_ratio is: " + "\n")

print(post_hoc.explained_variance_ratio_)