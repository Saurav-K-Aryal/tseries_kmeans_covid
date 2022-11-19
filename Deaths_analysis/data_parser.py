import matplotlib.pyplot as plt
import seaborn as sns

import wget
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

FILTER_SIZE = 575

special_regions = {'Diamond Princess', 'MS Zaandam', 'Taiwan*', 'Ascension and Tristan da Cunha', 'Holy See', 'Sint Eustatius and Saba'}
def get_population_df():
	pop_df = pd.read_csv('API_SP.POP.TOTL_DS2_en_csv_v2_3469297.csv', skiprows=4)
	pop_df = pop_df[['Country Name', '2020', 'Country Code']]
	pop_df.columns = ['country', 'population', 'iso_a3']
	return pop_df

def plot_days_hist(x):
	# histogram and kernel density estimation function of the variable height
	ax = sns.distplot(x, hist=True, hist_kws={"edgecolor": 'w', "linewidth": 3}, kde_kws={"linewidth": 3})
	plt.xticks(fontsize=14)
	plt.yticks([], [])

	# labels and title
	plt.xlabel('# days since first reported', fontsize=14)
	plt.ylabel('# countries', fontsize=14)
	# plt.title('Distribution of Days of Reports', fontsize=20)
	plt.tight_layout()
	plt.show()

def validate_increasing(countries):
	for c in countries:
		ts = countries[c]['ts']
		count = 0
		for i, j in zip(ts, ts[1:]):
			if i-j > 1:
				print('Not increasing for', c, i, j)
				count += 1

		if count > 0.02 * len(ts):
			print('Failed for', c)
			return False
	return True

def parse_daily(countries):
	for c in countries:
		cs = pd.Series(countries[c]['ts'])
		diff = cs.diff()
		if not any(diff < 0):
			diff[0] = cs[0]
			countries[c]['ts'] = diff.tolist()
			continue
		diff = diff.tolist()
		new_ts = [cs[0]]
		# new_ts = [diff[0]]
		tmp = []
		for n in diff[1:]:
			if n < 0:
				tmp.append(n)
			else:
				if len(tmp) != 0:
					num_divs = len(tmp) + 1
					tmp = [int(n / num_divs) for i in tmp]
					tmp.append(n - sum(tmp))
					new_ts.extend(tmp)
					tmp = []
				else:
					new_ts.append(n)
		# make sure no negatives & LENGTH is as expected
		assert any(n < 0 for n in new_ts) == False
		assert len(new_ts) == len(countries[c]['ts'])
		countries[c]['ts'] = new_ts
	return countries

# scale countries by population
def scale_death_by_population(countries):
	pop_df = get_population_df()
	print('lay', pop_df.head())
	for c in countries:
		# print(c)
		population = pop_df[pop_df['country']==c]['population'].tolist()[0]
		countries[c]['ts']= [t/population for t in countries[c]['ts']]
		countries[c]['iso_a3'] = pop_df[pop_df['country']==c]['iso_a3'].tolist()[0].replace('"', '')
	return countries

def dict_to_df(countries, has_iso=True):
	country_list = []
	for c in countries:
		ts = countries[c]['ts']
		my_dict = {'t'+str(i): ts[i] for i in range(len(ts))}
		my_dict['country'] = c
		my_dict['start'] = countries[c]['start']
		if has_iso:
			my_dict['iso_a3'] = countries[c]['iso_a3']
		# print(my_dict)
		country_list.append(my_dict)
	
	df = pd.DataFrame(country_list)
	return df

# scale min max
def scale_min_max(df):
	SCALE_MAX = 1
	# scaler = MinMaxScaler()
	cols = ['t'+str(i) for i in range(FILTER_SIZE)]
	df_cols = df[cols]
	print(df_cols)
	mini = df_cols.min().min()    
	maxi = df_cols.max().max()
	norm = (df_cols - mini) / (maxi - mini)
	# df[cols] = [SCALE_MAX * n for n in norm[cols]
	df[cols] = norm * SCALE_MAX
	return df

# scale mean var
def scale_mean_var(df):
	# SCALE_MAX = 1
	# scaler = MinMaxScaler()
	cols = ['t'+str(i) for i in range(FILTER_SIZE)]
	vals_df = df[cols]
	val_shape = vals_df.values.shape
	vals_df = TimeSeriesScalerMeanVariance().fit_transform(vals_df).reshape(val_shape)
	vals_df = pd.DataFrame(vals_df, columns=cols)
	df[cols] = vals_df[cols]
	return df

# plots distribution of first case
def plot_start_dist(countries):
	countries["date"] = countries["start"].astype("datetime64")
	# xticks = [d[] for d in countries["start"]]
	countries['country'].groupby([countries["date"].dt.year, countries["date"].dt.month]).count().plot(kind="bar")
	# plt.xticks()
	# ticks = [i for i in range(len(countries["start"]))]
	plt.xlabel('Month of First report', fontsize=14)
	plt.ylabel('# Countries', fontsize=14)
	# plt.title('Distribution of First Report in countries', fontsize=20)
	plt.show()

# get cases csv file and parses to return list of dicts by country and cases per week.
def get_csse_data(file_path):
	if 'confirmed' in file_path:
		global FILTER_SIZE
		FILTER_SIZE = 600

	f = open(file_path, 'r')
	columns = f.readline().split(',')
	columns[-1] = columns[-1][:-1]	#remove EOL
	# print('columns', columns)
	countries = {}
	for line in f:
		line = line.split(',')
		country = line[1].strip().replace('"', '')
		line[-1] = line[-1][:-1]	# remove EOL
		if country in countries:
			countries[country] = [int(a)+b for a,b in zip(line[5:], countries[country])]
		else:
			countries[country] = [int(s) for s in line[5:]]

	samples = 0
	for c in countries:
		samples += 1
		buff = 4
		leads = 0
		for k in countries[c]:
			if k == 0:
				leads += 1
			else:
				break
		countries[c] = {'ts':countries[c][leads:], 'start':columns[buff+leads]}	

	# plot to show reporting outliers
	plot_days_hist([len(countries[c]['ts']) for c in countries])

	# plot to show how different first cases were
	plot_start_dist(dict_to_df(countries, has_iso=False))

	# remove outliers
	countries = {c:{'start':countries[c]['start'], 'ts':countries[c]['ts'][:FILTER_SIZE]} for c in countries if len(countries[c]['ts']) >= FILTER_SIZE and c not in special_regions}
	print(countries.keys(), len(countries))

	# validate sequences are all increasing at least 2% of times
	assert validate_increasing(countries), True

	# get daily counts from cumulative data
	countries = parse_daily(countries)

	# scale population for per capita impact
	countries = scale_death_by_population(countries)

	# convert dict to df
	countries = dict_to_df(countries)

	# scale Min-Max
	countries = scale_mean_var(countries)
	# print(countries['iso_a3'])
	return countries

if __name__=="__main__":
	get_csse_data('./COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
	get_csse_data('./COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
