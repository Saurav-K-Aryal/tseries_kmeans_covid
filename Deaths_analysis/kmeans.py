import math
from data_parser import get_csse_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import geopandas as gpd
import similaritymeasures
from sklearn.cluster import KMeans
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape, TimeSeriesKMeans, KernelKMeans, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import scipy.stats as stats 
# from sklearn.metrics import silhouette_score
# import sys
# sys.setrecursionlimit(2000000)

MAX_CLUSTERS = 15
N_INIT = MAX_CLUSTERS

def plot_world_map(k, deaths_df):
	world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
	world = world[(world.pop_est>0) & (world.name!="Antarctica")]
	world["country"] = world["name"]
	print(world.columns)
	print('LEN 1!!!',len(deaths_df), len(world))
	world = world.merge(deaths_df, on=['iso_a3'], how='outer')
	# print(world.columns, deaths_df.columns)
	world['country'] = world['country_x']
	print(world.columns)
	# world = world.merge(deaths_df, on=['country'])
	# world[['iso_a3_x', 'name']].to_csv('world.csv')
	print('LEN 2!!!',len(deaths_df), len(world))
	# cax = divider.append_axes("right", size="5%", pad=0.1)
	ax = world.plot(column='labels',  legend=True, cmap='rainbow', categorical=True,
			 missing_kwds={
			 "color": "yellow",
			"label": "missing",
		}, edgecolor="black", legend_kwds={'loc': "lower center", "ncol":k+1, "bbox_to_anchor":(0.25, -0.25, 0.5, -.10), 'fontsize':'xx-small'})
	# legends = ax.get_legends()
	# print(legends, dir(legends))
	ax.set_axis_off()
	plt.tight_layout()
	plt.show()
	
	world['labels'] = world['labels'].fillna('missing')
	world['labels'] = [str(num) for num in world['labels']]
	# GPD per cap
	world['gdp_per_cap'] = world['gdp_md_est'] / world['pop_est']
	world['gdp_per_cap']= 10**6 * world['gdp_per_cap']

	print('ANOVA GDP per cap')
	model = ols('gdp_per_cap ~ C(labels)', data=world).fit()
	aov_table = sm.stats.anova_lm(model, typ=2)
	print(aov_table)
	comp = mc.MultiComparison(world['gdp_per_cap'], world['labels'])
	tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")

	print(tbl)

	print('\n\nANOVA GDP')
	model = ols('gdp_md_est ~ C(labels)', data=world).fit()
	aov_table = sm.stats.anova_lm(model, typ=2)
	print(aov_table)
	comp = mc.MultiComparison(world['gdp_md_est'], world['labels'])
	tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")
	print(tbl)

	print('\n\nANOVA pop')
	model = ols('pop_est ~ C(labels)', data=world).fit()
	aov_table = sm.stats.anova_lm(model, typ=2)
	print(aov_table)
	comp = mc.MultiComparison(world['pop_est'], world['labels'])
	tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")
	print(tbl)

	# world
	world.boxplot(column=['gdp_per_cap'], by='labels')
	plt.show()
	world.boxplot(column=['gdp_md_est'], by='labels')
	plt.show()
	world.boxplot(column=['pop_est'], by='labels')
	plt.show()

	grouped_df = world.groupby('labels')

	for key, item in grouped_df:
		group = grouped_df.get_group(key)
		group = group[group['name'].notna()]

		print(group, "\n", key, len(group), "\n\n")



def plot_clusters(k, deaths_df, model):
	for yi in range(k):
		# plt.subplot(k, 1, 1 + yi)
		d_fil = deaths_df.loc[deaths_df['labels'] == yi]
		cols = [col for col in deaths_df.columns if 't' == col[0]]
		d_fil = d_fil[cols].values.tolist()
		for xx in d_fil:
			plt.plot(xx, "k-", alpha=.2)
		plt.plot(model.cluster_centers_[yi].ravel(), "r-")
		# plt.xlim(0, sz)
		# plt.ylim(0, 100)
		plt.title("Cluster %d" % yi)
		plt.show()

def choose_elbow_optimal_k(df, show_plot=True, is_verbose=True):
	cols = [col for col in df.columns if 't' == col[0]]
	vals_df = df[cols]
	# vals_data = to_time_series_dataset(vals_df.values.tolist())
	# N = len(df)
	# num_trails = math.ceil(N * math.log(N) / 2) # log base e to maximize it
	# print('n_init', num_trails)
	model = TimeSeriesKMeans(n_jobs=-1, n_init=N_INIT, metric='euclidean', verbose=is_verbose)
	visualizer = KElbowVisualizer(model, k=(2,MAX_CLUSTERS), metric='distortion', show=show_plot)

	visualizer.fit(vals_df)		# Fit the data to the visualizer
	if show_plot:
		visualizer.show()		# Finalize and render the figure
	# else:
	# 	visualizer.close()

	# print(visualizer)

	# retrain for optimal K
	optimal_k = visualizer.elbow_value_
	model = TimeSeriesKMeans(n_clusters=optimal_k, n_init=N_INIT, n_jobs=-1, metric='euclidean', verbose=True)
	return model, optimal_k

def run_experiment(deaths_df, num_trails=5, num_repeats=10):
	if num_trails==0:
		model = TimeSeriesKMeans(n_clusters=6, n_init=N_INIT, n_jobs=-1, metric='euclidean', verbose=True)
		return model, 6
	count_map = dict()
	prev_common_k = 0

	show_plot = True
	for j in range(num_trails):
		print('TRIAL#', j)
		model, k = choose_elbow_optimal_k(deaths_df, show_plot=show_plot, is_verbose=False)
		if k in count_map:
			count_map[k] += 1
		else:
			count_map[k] = 1
		print(count_map)
		show_plot = False
	# if prev_common_k == 0:		
	# 	prev_common_k = max(count_map.items(), key=lambda x:x[1])
	# 	continue
	# else:
	# prev_common_k= max(count_map.items(), key=lambda x:x[1])
	print(count_map)
	common_k = max(count_map.items(), key=lambda x:x[1])[0]
	# N = len(deaths_df)
	# num_trails = math.ceil(N * math.log(N) / 2) # log base e to maximize it
	print('COMMON', common_k)
	model = TimeSeriesKMeans(n_clusters=common_k, n_init=N_INIT, n_jobs=-1, metric='euclidean', verbose=True)
	return model, common_k

def finalize(deaths_df):
	deaths_list = deaths_df.values.tolist()
	# print(deaths_list)
	import random

	random.shuffle(deaths_list)

	for val in deaths_list[:2]:
		x = val[:-3]
		plt.plot([i for i in range(len(x))], val[:-3], label=val[-3], alpha=0.8, color="blue", linewidth="1")

	plt.show()

	model, k = run_experiment(deaths_df, num_trails=1)
	# model, k = choose_optimal_model(deaths_df)
	vals_df = deaths_df[[col for col in deaths_df.columns if 't' == col[0]]]
	print(vals_df, "HERE!!!")
	model.fit_predict(vals_df)
	# print(deaths_df['labels'])
	deaths_df['labels'] = model.fit_predict(vals_df)
	deaths_df.to_csv('deaths_out.csv')
	print(set(deaths_df['labels']))


	plot_clusters(k, deaths_df, model)
	# plt.figure()

	plot_world_map(k, deaths_df)



# deaths
print('For deaths per capita')
deaths_df = get_csse_data('./COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
# print("LOL", deaths_df.isnull().values.any())
deaths_df = deaths_df[deaths_df['t0'].notna()]
print("check")
# print(deaths_df)
plt.show()
finalize(deaths_df)
