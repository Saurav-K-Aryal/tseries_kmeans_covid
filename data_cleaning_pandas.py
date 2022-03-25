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
	print('CSV filename not supplied or extra params given.')
	sys.exit(1) # terminate execution

print('Running file', sys.argv[0])
print('Using csv at', sys.argv[1])

csv_file_path = sys.argv[1]
# for debug only
# N_ROWS = 10000000

print('Reading', len(columns), '# of columns')

from time import time
st = time()
# check if read_csv fails..
try:
	df = pd.read_csv(csv_file_path, dtype=dtypes, usecols=columns)
	print(df.columns)
	# print(df[['smoking_prevalence','diabetes_prevalence','stay_at_home_requirements','international_travel_controls','testing_policy']].head())
except:
	traceback.print_exception(*sys.exc_info())	# print error
	print('\n*******\nError reading supplied csv file\n*******\n')
	sys.exit(1)	# terminate execution

print('Grouping the following df')
print(df.head())

# take mean on numeric columns and mode on object columns
# create dict like in example here
# https://stackoverflow.com/questions/48225508/pandas-get-the-average-and-mode-group-by-specified-columns
agg_dict = dict((k,'mean') if dtypes[k] != 'object' else (k, lambda x: x.mode()) for k in dtypes)
del agg_dict['country_name']
del agg_dict['iso_3166_1_alpha_3']
print(agg_dict)

# group_by
grouped_df = df.groupby(['country_name', 'iso_3166_1_alpha_3']).agg(agg_dict)
grouped_df.to_csv('grouped_mean_modes.csv')
print('Script takes', (time() - st) / 60, 'minutes')