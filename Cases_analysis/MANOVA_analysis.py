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
from sklearn.preprocessing import OrdinalEncoder
import plotly.express as px
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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








maov = MANOVA.from_formula('average_temperature_celsius + comorbidity_mortality_rate + cumulative_persons_fully_vaccinated + cumulative_persons_vaccinated + cumulative_tested + diabetes_prevalence + elevation_m + gdp_per_capita_usd + gdp_usd + human_capital_index + latitude + population_density + smoking_prevalence ~ labels', data = df)
print(maov.mv_test())


X = df[['average_temperature_celsius','comorbidity_mortality_rate','cumulative_persons_fully_vaccinated','cumulative_persons_vaccinated','cumulative_tested','diabetes_prevalence','elevation_m','gdp_per_capita_usd','gdp_usd','human_capital_index','latitude','population_density','smoking_prevalence']].fillna(0)
y = df["labels"].fillna(0)

# print(X.shape, y.shape)
# print(X.head)
#print(y.head)


# initial lda approach
post_hoc = lda().fit(X=X, y=y)


# LDA = lda(n_components=3)
# post_hoc = LDA.fit(X, y,z)
# get Prior probabilities of groups:

print("\n" + "The post hoc priors is: " + "\n")

print(post_hoc.priors_)

print("\n" + "The post hoc means is: " + "\n")

print(post_hoc.means_)

print("\n" + "The post hoc scaling is: " + "\n")

print(post_hoc.scalings_)

print("\n" + "The explained_variance_ratio is: " + "\n")

print(post_hoc.explained_variance_ratio_)


#Graph Plot

X_new = pd.DataFrame(lda().fit(X, y).transform(X), columns=["lda0","lda1", "lda2","lda3", "lda4", "lda5"])
X_new["labels"] = df["labels"]
sns.scatterplot(data=X_new, x="lda1", y="lda2", hue=df.labels.tolist())
plt.show()


######################################################################################################

# LDA with 3 features and 3d scatterplot 
print('######################################################################################')
print('LDA with 3 features and 3d scatterplot')
print('######################################################################################')

df['Band'] = pd.qcut(df['labels'], 3, labels=['1.First-range (bottom 33%)', '2.Mid-range (middle 33%)', '3.Bottom-range (top 33%)'])
# Check distribution
price = df['Band'].value_counts().sort_index()
print(price)

enc=OrdinalEncoder() 

# Encode categorical values
df['Band enc']=enc.fit_transform(df[['Band']])

# Check encoding results in a crosstab
crosstab = pd.crosstab(df['Band'], df['Band enc'], margins=False)
print(crosstab)

# plotting the 3d graph 

# Create a 3D scatter plot

'''
fig = px.scatter_3d(df, 
                    x=df['X1 transaction date'], y=df['X2 house age'], z=df['X3 distance to the nearest MRT station'],
                    color=df['Price Band'],
                    color_discrete_sequence=['#636EFA','#EF553B','#00CC96'], 
                    hover_data=['X3 distance to the nearest MRT station', 'Y house price of unit area', 'Price Band enc'],
                    height=900, width=900
                   )

# Update chart looks
fig.update_layout(#title_text="Scatter 3D Plot",
                  showlegend=True,
                  legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                  scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                        center=dict(x=0, y=0, z=-0.2),
                                        eye=dict(x=-1.5, y=1.5, z=0.5)),
                                        margin=dict(l=0, r=0, b=0, t=0),
                  scene = dict(xaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='#f0f0f0',
                                          title_font=dict(size=10),
                                          tickfont=dict(size=10),
                                         ),
                               yaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='#f0f0f0',
                                          title_font=dict(size=10),
                                          tickfont=dict(size=10),
                                          ),
                               zaxis=dict(backgroundcolor='lightgrey',
                                          color='black', 
                                          gridcolor='#f0f0f0',
                                          title_font=dict(size=10),
                                          tickfont=dict(size=10),
                                         )))

# Update marker size
fig.update_traces(marker=dict(size=2))

fig.show()

'''

x_comp = df[['average_temperature_celsius','comorbidity_mortality_rate','cumulative_persons_fully_vaccinated','cumulative_persons_vaccinated','cumulative_tested','diabetes_prevalence','elevation_m','gdp_per_capita_usd','gdp_usd','human_capital_index','latitude','population_density','smoking_prevalence']].fillna(0)
y_comp = df['Band enc'].values
# y_comp = y_comp.fillna(0)

# Get scaler
scaler=StandardScaler()
# Perform standard scaling on model features
X_scaler = scaler.fit_transform(x_comp)


# Select the model and its parameters
LDA = lda(
    solver='eigen', #{‘svd’, ‘lsqr’, ‘eigen’}, default=’svd’
    n_components=3, #int, default=None
    #shrinkage=None, #‘auto’ or float, default=None
    #priors=None, #array-like of shape (n_classes,), default=None, The class prior probabilities. By default, the class proportions are inferred from the training data.
    #store_covariance=False, #bool, default=False, If True, explicitely compute the weighted within-class covariance matrix when solver is ‘svd’. 
    #tol=0.0001, #float, default=1.0e-4, Absolute threshold for a singular value of X to be considered significant, used to estimate the rank of X.
)

# Fit transform the data
X_trans_lda=LDA.fit_transform(X,y)


# Print the results
print('*************** LDA Summary ***************')
print('Classes: ', LDA.classes_)
print('Priors: ', LDA.priors_)
print('Explained variance ratio: ', LDA.explained_variance_ratio_)


## plot 3d scatter plot
sns.set(style = "darkgrid")

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

a = X_trans_lda[:,0]
b = X_trans_lda[:,1]
c = X_trans_lda[:,2]

ax.set_xlabel("lda1")
ax.set_ylabel("lda2")
ax.set_zlabel("lda3")

ax.scatter(a, b, c, c=y)

plt.show()