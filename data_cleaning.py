# import datatable as dt
# df = dt.fread("s3://.../2018-*-*.csv")

# import pandas as pd
from dask import dataframe as dd
import pandas as pd
df = dd.read_csv(r'C:\Users\nilis\Downloads\Documents\For Research\K means\Dataset\aggregated.csv', dtype={'age_bin_0': 'object',
       'age_bin_1': 'object',
       'age_bin_2': 'object',
       'age_bin_3': 'object',
       'age_bin_4': 'object',
       'age_bin_5': 'object',
       'age_bin_6': 'object',
       'age_bin_7': 'object',
       'age_bin_8': 'object',
       'age_bin_9': 'object',
       'area_sq_km': 'float64',
       'cancel_public_events': 'float64',
       'contact_tracing': 'float64',
       'cumulative_confirmed': 'float64',
       'cumulative_deceased': 'float64',
       'debt_relief': 'float64',
       'emergency_investment_in_healthcare': 'float64',
       'facial_coverings': 'float64',
       'fiscal_measures': 'float64',
       'gdp_per_capita_usd': 'float64',
       'gdp_usd': 'float64',
       'income_support': 'float64',
       'international_support': 'float64',
       'international_travel_controls': 'float64',
       'investment_in_vaccines': 'float64',
       'new_confirmed': 'float64',
       'new_deceased': 'float64',
       'openstreetmap_id': 'float64',
       'population_age_00_09': 'float64',
       'population_age_10_19': 'float64',
       'population_age_20_29': 'float64',
       'population_age_30_39': 'float64',
       'population_age_40_49': 'float64',
       'population_age_50_59': 'float64',
       'population_age_60_69': 'float64',
       'population_age_70_79': 'float64',
       'population_age_80_and_older': 'float64',
       'population_female': 'float64',
       'population_male': 'float64',
       'population_rural': 'float64',
       'population_urban': 'float64',
       'public_information_campaigns': 'float64',
       'public_transport_closing': 'float64',
       'restrictions_on_gatherings': 'float64',
       'restrictions_on_internal_movement': 'float64',
       'school_closing': 'float64',
       'stay_at_home_requirements': 'float64',
       'subregion1_code': 'object',
       'subregion1_name': 'object',
       'subregion2_name': 'object',
       'testing_policy': 'float64',
       'vaccination_policy': 'float64',
       'workplace_closing': 'float64'}, usecols=['country_name','iso_3166_1_alpha_3','elevation_m','average_temperature_celsius','population_density','gdp_usd',
       'smoking_prevalence','international_travel_controls','stay_at_home_requirements','latitude','gdp_per_capita_usd','human_capital_index','cumulative_tested','comorbidity_mortality_rate','diabetes_prevalence','cumulative_persons_vaccinated','cumulative_persons_fully_vaccinated','testing_policy'])


print(df.head(20))          #print the unprocessed head
# print(df.columns)         #print all of the columns
     
     
# counting_NANs = df.groupby(['country_name', 'iso_3166_1_alpha_3']).isnull().sum().compute()   
# counting_NANs = df.isnull().groupby(['country_name', 'iso_3166_1_alpha_3']).sum().compute()
# counting_more_NANS = df.groupby(['country_name', 'iso_3166_1_alpha_3']).size().compute()   #works
#counting_size = df.isnull().groupby(['country_name', 'iso_3166_1_alpha_3']).size().compute()             # doesn't work revisit later 


# print(counting_NANs.head(20))
# print(counting_more_NANS.head(20))
#print(counting_size.head(20))


# df = df.apply(lambda x: pd.Series(x.dropna().values), meta={0: 'object', 1: 'object', 2: 'float64', 3: 'float64', 4: 'float64', 5: 'float64', 6: 'float64', 7: 'float64', 8: 'float64', 9: 'float64', 10: 'float64', 11: 'float64', 12: 'float64', 13: 'float64', 14: 'float64', 15: 'float64', 16: 'float64', 17: 'float64'}, axis=1).compute()

# df = df.apply(lambda x: pd.Series(x.dropna().values))
# print(df.head(20))




cd = df.groupby(['country_name', 'iso_3166_1_alpha_3']).mean(skipna=True)        #collapsing millions line into data to few hundred lines by taking the mean of all the values based on their country identification and the column name
# dataTypeSeries = df.dtypes
# print(cd.head(20))
# print(dataTypeSeries)

''' #counting NANs after calculating the mean
       # def counting_NANs(dframe):
       #        total_NAN = dframe.isnull().sum().sum().compute()                     #calculating the total NANs in the whole csv file

       #        print ('Total count of NaN in DataFrame: ' + str(total_NAN), sep='\n')       #printing total number of NANs for the csv file
       #        print("Nan in each columns" , dframe.isnull().sum().compute(), sep='\n')           #printing NANs for each column
              
       #        # print("***Count NaN in each row of a DataFrame***")
       
       #        # for i in range(len(dframe.index)) :
       #        #        print("Nan in row ", i , " : " ,  dframe.iloc[:, i].isnull().sum().compute())


       # counting_NANs(cd)
'''