import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import pingouin as pg
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import classification_report
from regressors import stats
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
scaler = MinMaxScaler()

# df = pd.read_csv('combined_dfs.csv')

df = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/Deaths_analysis/newData_+_newLabels_deaths.csv')
print(df.head)
# print(df.dtypes)
# df.drop(columns=['Unnamed: 0'], inplace=True)
df = df[df['labels'].notna()]  #dropping countries with no labels
df.replace(to_replace=[], value='')
# print(df.head())

col_names = ['average_temperature_celsius', 'comorbidity_mortality_rate',
                          'average_new_fully_vaccinated_per_day_per_1000','average_new_vaccinated_per_day_per_1000',
                          'average_new_tested_per_day_per_1000','diabetes_prevalence',
                          'gdp_per_capita_usd','gdp_usd','human_capital_index',
                          'latitude','population_density','smoking_prevalence',
                          'total_cases_divided_by_population', 'total_deaths_divided_by_population','population', 'international_travel_controls', 'stay_at_home_requirements', 'testing_policy',
        'public_information_campaigns', 'vaccination_policy', 'facial_coverings', 'cancel_public_events']
features = df[col_names]

scaler = MinMaxScaler()

df[col_names] = scaler.fit_transform(features.values)
col_names.append('labels')
df = df[col_names]
# df.replace('[]', None)
#
print(df.head())
# print(df.dtypes)
X = df[['average_temperature_celsius', 'comorbidity_mortality_rate',
                          'average_new_fully_vaccinated_per_day_per_1000','average_new_vaccinated_per_day_per_1000',
                          'average_new_tested_per_day_per_1000','diabetes_prevalence',
                          'gdp_per_capita_usd','gdp_usd','human_capital_index',
                          'latitude','population_density','smoking_prevalence',
                          'total_cases_divided_by_population', 'total_deaths_divided_by_population','population', 'international_travel_controls', 'stay_at_home_requirements', 'testing_policy',
        'public_information_campaigns', 'vaccination_policy', 'facial_coverings', 'cancel_public_events']].fillna(0)
y = df['labels']

df = df.astype({'labels' : 'category'})

# X = df[['average_temperature_celsius','comorbidity_mortality_rate','cumulative_persons_fully_vaccinated','cumulative_persons_vaccinated','cumulative_tested','diabetes_prevalence','elevation_m','gdp_per_capita_usd','gdp_usd','human_capital_index','latitude','population_density','smoking_prevalence']].fillna(0)
# y = df['labels'].fillna(10)

print(list(X.columns.values)) 

#data divided to train and test 
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.20, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#logistic regression applied
model1 = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(X_train, y_train)
preds = model1.predict(X_test)  #prediction on x_test

#print the tunable parameters 
params = model1.get_params()
print(params)

#Print model parameters
print('Intercept: \n', model1.intercept_)
print('Coefficients: \n', model1.coef_)

summary = pd.DataFrame(zip(X_train.columns, np.transpose(model1.coef_.tolist()[0])), columns=['features', 'coef'])

#output added to summary file for better redability 
with open("deaths_summary2.txt", 'w') as f:
    print(summary, file = f)
    f.close()


#different apporach tried to print the summary working on it.
print("#################################################")


# print the classification report with the y_test and x_test prediction values

class_report = classification_report(y_test, preds)

print(class_report)

#Sklearn approach to print the R-squared value 
print("Sklearn approach to print the R-squared value ", model1.score(X_train, y_train))


#regressor library method
#print the adjusted R-squared

val = stats.adj_r2_score(model1, X_train, y_train)
print("The regressor library adjusted R -squared is : ", val)





#1st approach

logit_model=sm.MNLogit(y_train,sm.add_constant(X_train))
# # logit_model
result=logit_model.fit()
stats1=result.summary()
stats2=result.summary2()
with open("deaths_summary1.txt", 'w') as f:
    print(stats1, file = f)
    print(stats2, file = f)
    f.close()


# 2nd approach
# X_train = sm.add_constant(X_train, prepend=False)

# mnlogit_mod = sm.MNLogit(y_train, X_train)
# mn_logit_fit = mnlogit_mod.fit()
# print(mn_logit_fit.summary())


#stats model to print the summary 

# x_labels = df['labels']

# print(stats.summary(df, X, y))

# vectorizer = TfidfVectorizer(min_df = 10)
# explainer = shap.Explainer(model1, X_train)
# shap_values = explainer(X_test)
# print(shap_values)
# # shap.plots.beeswarm(shap_values)
# shap.summary_plot(shap_values = shap_values)

# import matplotlib.pyplot as plt
# from itertools import cycle
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.metrics import PrecisionRecallDisplay

# setup plot details
# colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

# _, ax = plt.subplots(figsize=(7, 8))

# f_scores = np.linspace(0.2, 0.8, num=4)
# lines, labels = [], []
# for f_score in f_scores:
#     x = np.linspace(0.01, 1)
#     y = f_score * x / (2 * x - f_score)
#     (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
#     plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

# display = PrecisionRecallDisplay(
#     recall="micro",
#     precision="micro",
#     average_precision="micro",
# )
# display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

# plt.show()