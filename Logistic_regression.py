import numpy as np
import matplotlib.pyplot as plt
# from pinguoin import multivariate_normality
import sklearn
import pandas as pd
import pingouin as pg
from pingouin import multivariate_normality
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bartlett
from scipy.stats import levene
from statsmodels.multivariate.manova import MANOVA
from sklearn.preprocessing import OrdinalEncoder
import plotly.express as px
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

import statsmodels.api as sm

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

X = df[['average_temperature_celsius','comorbidity_mortality_rate','cumulative_persons_fully_vaccinated','cumulative_persons_vaccinated','cumulative_tested','diabetes_prevalence','elevation_m','gdp_per_capita_usd','gdp_usd','human_capital_index','latitude','population_density','smoking_prevalence']].fillna(0)
y = df['labels'].fillna(0)

print(list(X.columns.values)) 

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.20, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model1 = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(X_train, y_train)
preds = model1.predict(X_test)

#print the tunable parameters (They were not tuned in this example, everything kept as default)
params = model1.get_params()
print(params)

#Print model parameters
print('Intercept: \n', model1.intercept_)
print('Coefficients: \n', model1.coef_)

summary = pd.DataFrame(zip(X.columns, np.transpose(model1.coef_.tolist()[0])), columns=['features', 'coef'])

print(summary)

# print(np.exp(model1.coef_))

# logit_model=sm.MNLogit(y_train,sm.add_constant(X_train))
# logit_model
# result=logit_model.fit()
# stats1=result.summary()
# stats2=result.summary2()
# with open("output3.txt", 'a') as f:
#     print(stats1, file = f)
#     print(stats2, file = f)
#     f.close()