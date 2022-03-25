# data cleaning with pandas
import pandas as pd

# same dtypes to second file
from constants import dtypes, columns


# import python stdlib modules
import traceback
import sys
from pprint import pprint

# read path from command line
# python3 data_cleaning_pandas.py "path_to_file.csv"

# needs only .py file and .csv file
if len(sys.argv) != 2:
	print('CSV filename not supplied')
	sys.exit(1) # terminate execution

print('Running file', sys.argv[0])
print('Using csv at', sys.argv[1])

csv_file_path = sys.argv[1]
# for debug only
N_ROWS = 10000000
inferred_dtypes = {'country_name': 'object', 'iso_3166_1_alpha_3': 'object', 'age_bin_0': 'object', 'age_bin_1': 'object',
       'age_bin_2': 'object',
       'age_bin_3': 'object',
       'age_bin_4': 'object',
       'age_bin_5': 'object',
       'age_bin_6': 'object',
       'age_bin_7': 'object',
       'age_bin_8': 'object',
       'age_bin_9': 'object'}

s = pd.read_csv('/home/anirudd/Documents/tseries_kmeans_covid/aggregated.csv', usecols=['country_name'])
n_countries = s['country_name'].nunique(dropna=False)
# run diagnostics on each column for inferring dtype

for col in columns:
	if col in inferred_dtypes:
		continue
	print('-----------------')
	s = pd.read_csv(csv_file_path, usecols=[col])
	n_unique = s[col].nunique(dropna=False)
	# n_countries = s['country_name'].nunique(dropna=False)
	# if num unqiue values for a column is less than 25% of # countries
	# it is probably of type object
	if n_unique < 0.25 * n_countries:
		print(n_unique, n_countries)
		inferred_dtypes[col] = 'object'
		print(col, 'is of type object')
	else:
		print(n_unique, n_countries)
		inferred_dtypes[col] = s[col].dtype

pprint(inferred_dtypes)
print(inferred_dtypes)


# dtypes inferred just fine...
sys.exit(0)

# /home/anirudd/Documents/tseries_kmeans_covid/aggregated.csv