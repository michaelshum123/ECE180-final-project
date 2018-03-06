import pandas as pd
import numpy as np

Files = pd.read_csv('DJI_last_10years.csv')
Files['date'] = pd.to_datetime(Files['date'])
Files = Files.set_index(['date'])
month = Files['2017-12']
half_year = Files['2017-06':'2017-12']
year = Files['2017']
month.to_csv('DJI_Last_month_data.csv')
half_year.to_csv('DJI_Half_year_data.csv')
year.to_csv('DJI_Last_year_data.csv')

Files = pd.read_csv('NASDAQ_las_10years.csv')
Files['date'] = pd.to_datetime(Files['date'])
Files = Files.set_index(['date'])
month = Files['2017-12']
half_year = Files['2017-06':'2017-12']
year = Files['2017']
month.to_csv('NASDAQ_Last_month_data.csv')
half_year.to_csv('NASDAQ_Half_year_data.csv')
year.to_csv('NASDAQ_Last_year_data.csv')
