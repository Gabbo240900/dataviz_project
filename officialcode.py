# Let's import the main libraries!
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Let's read the dataset and print an initial overview
df_airplanes = pd.read_csv('International_Report_Departures.csv')
df_airplanes.head()

# Let's check the nature of our values
# Print data types of each column
print(df_airplanes.dtypes)
# Check for missing values
print(df_airplanes.isnull().sum())

#There are 3055obs which lack of 'carrier' obj.. since we have over 900k observations,
#we can just drop them.
df_airplanes = df_airplanes.dropna()

# Check for missing values
print(df_airplanes.isnull().sum())
#All good, 0  missing values!

# Check for duplicates
duplicates = df_airplanes.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")
# No duplicates in our dataframe, great!

# The name of the columns are not explanatory, thus we rename them
df_airplanes = df_airplanes.rename(columns={'data_dte': 'Date',
                                            'usg_apt_id': 'US_airport_ID',
                                            'usg_apt': 'US_airport_code',
                                            'usg_wac': 'US_World_Area_Code',
                                            'fg_apt_id': 'Foreign_airport_ID',
                                            'fg_apt': 'Foreign_airport_code',
                                            'fg_wac': 'Foreign_World_Area_Code',
                                            'airlineid': 'Airline_ID',
                                            'carrier': 'Carrier',
                                            'carriergroup': 'Carrier_of_US',
                                            'Scheduled': 'Scheduled_Flights',
                                            'Charter': 'Charter_Flights',
                                            'Total': 'Total_Flights'})

# It seems that 'type' column has only equal values, let's check it!
if df_airplanes['type'].nunique() == 1:
    print("'type' column contains only equal values.")


#Let's drop 'type' column because it has all equal values
df_airplanes = df_airplanes.drop('type', axis=1)

df_airplanes['Date'] = (pd.to_datetime(df_airplanes['Date'], format='%m/%d/%Y', errors='coerce'))

# Create a boxplot to check for outliers
fig, ax = plt.subplots()
ax.boxplot(df_airplanes['Total_Flights'])

# Add labels and title
ax.set_title('Boxplot ')
ax.set_ylabel('Total_Flights')

plt.show()
# It seems like there are few observations in which there are over than 1000 flights reaching a top number 2000 in May '96
# These should not be removed, but more analyzed to check why there have been so many fligths in those days





# Barplot with airports and number of fligths for each ariport

df_us_fligths = df_airplanes[['US_airport_code','Total_Flights']].groupby(by='US_airport_code').sum()
df_foreign_fligths = df_airplanes[['Foreign_airport_code','Total_Flights']].groupby(by='Foreign_airport_code').sum()

df_fligths_count  = pd.concat([df_foreign_fligths, df_us_fligths])
df_fligths_count.reset_index(inplace=True)

df_fligths_count.sort_values(by='Total_Flights', ascending = False, inplace=True)

top_10_airp = df_fligths_count.head(10)





sns.barplot(data=top_10_airp, x="index", y="Total_Flights")
plt.xlabel('Airport')
plt.title('Fligths per Airport')
plt.ticklabel_format(style = 'plain', axis = 'y')
plt.show()



































