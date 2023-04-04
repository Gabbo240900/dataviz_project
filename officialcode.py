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
# group by US_airport and sum the number of fligths 
df_foreign_fligths = df_airplanes[['Foreign_airport_code','Total_Flights']].groupby(by='Foreign_airport_code').sum() 
# group by foreign and sum the number of fligths 

df_fligths_count  = pd.concat([df_foreign_fligths, df_us_fligths])
# concatenate the two datasets to get a single one with all the fligths
df_fligths_count.reset_index(inplace=True)
# reset the index to set the airport code as a column

df_fligths_count.sort_values(by='Total_Flights', ascending = False, inplace=True)
# Sort based on fligths for creating a barplot of the top 10 airports

top_10_airp = df_fligths_count.head(10)
# New dataset with only the top 10 airports


# Barplot with seaborn 
sns.barplot(data=top_10_airp, x="index", y="Total_Flights") 
plt.xlabel('Airport')
plt.title('Fligths per Airport')
plt.ticklabel_format(style = 'plain', axis = 'y')
plt.show()




# ----- Now we will start creating a heatmap of the US Fligths based on total fligths -----
df_us_fligths.reset_index(inplace=True) #reset index


# we retrived the coordinates by using another dataframe that links the airport identification name to the coordinates
# pip install airportsdata
import airportsdata # new library for retrieving the coordinates

# We discovered that there are various formats for airport codes
# we have both LID format and IATA format. 
# In the next step we will try to catch as most airports as possible


# Dictionary with all the infoprmation we need 
airports = airportsdata.load('LID') 

location = [] #empty list to store all the airports that have an ariport code in 'LID' format
missed = [] #list with all missed airports

# error managemtn for loop, to get all the location with LID format
for index, row in df_us_fligths.iterrows():
    try:
        location.append(airports[row['US_airport_code']])
    except KeyError:
        missed.append([row['US_airport_code']]) # we checked for all the airport ids that didn't match the new dataframe
        continue


df_missed = pd.DataFrame(missed) # all missed airports will now be looped again to check if they have IATA format code
df_missed.columns = ['US_airport_code'] #rename the column

location2=[]#empty list to store all the airports that have an ariport code in 'LID' format
missed=[] # new empty missed list

# Now we try to catch all IATA format airport codes
airports = airportsdata.load('IATA')
for index, row in df_missed.iterrows():
    try:
        location2.append(airports[row['US_airport_code']])
    except KeyError:
        missed.append([row['US_airport_code']])
        continue

df_lid = pd.DataFrame(location)# Dataframe with all LID format codes
df_iata = pd.DataFrame(location2)# Dataframe with all IATA format codes
df_missed = pd.DataFrame(missed) # Dataframe with all missed airport codes




# we created a new df that merges the coordinates with the airport id and the total number of flights
# Repeated both for LID and IATA format code
df_us_fligths.columns = ['lid', 'n_fligths']
df_lid = pd.merge(df_lid, df_us_fligths, on='lid', how='outer')
df_lid.dropna(subset=['icao'], inplace=True)


df_us_fligths.columns = ['iata', 'n_fligths']
df_iata = pd.merge(df_iata, df_us_fligths, on='iata', how='outer')
df_iata.dropna(subset=['icao'], inplace=True)



# # AGGIUNGI A QUELLI DI PRIMA SCRIVENDO A MANO LATITUDINE E LONGITUDINE


# df_us_fligths.columns = ['US_airport_code', 'n_fligths']
# df_missed.columns = ['US_airport_code']

# df_missed = pd.merge(df_missed, df_us_fligths)




# airports['LKE']






df_heat_map = pd.concat([df_lid, df_iata])# final dataset with all catched airports

lat_long_fligths = df_heat_map.iloc[:, [7, 8,11]] # dataset with only the information we need for the heatmap
# new library for the heatmap creation 

# pip install folium
import folium
from folium.plugins import HeatMap

map_obj = folium.Map(location = [38.27312, -98.5821872], zoom_start = 5) #where to make the map start


HeatMap(lat_long_fligths).add_to(map_obj)

map_obj.save(r"us_fligths_map.html") # we saved the figure to visualize the result




# Convert the "Date" column to a datetime object
df_airplanes['Date'] = pd.to_datetime(df_airplanes['Date'], format='%m/%d/%Y')

# Group the data by airport and date and aggregate the total number of flights
flights_per_airport = df_airplanes.groupby(['US_airport_code', 'Date'])['Total_Flights'].sum().reset_index()

# Get the top 5 airports with the most total flights
top_airports = flights_per_airport.groupby('US_airport_code')['Total_Flights'].sum().nlargest(5).index

# Filter the data to only include the top 5 airports
flights_per_airport_top5 = flights_per_airport[flights_per_airport['US_airport_code'].isin(top_airports)]

# Pivot the data to have each airport as a separate column and the dates as the rows
pivoted_flights = flights_per_airport_top5.pivot(index='Date', columns='US_airport_code', values='Total_Flights')

# Create a line plot for each airport showing the total number of flights over time
fig, ax = plt.subplots(figsize=(12,8))
pivoted_flights.plot(ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel("Total Flights")
ax.set_title("Flights per Airport over Time")

# Add a legend to show which color corresponds to each airport
legend_labels = [f"{airport} ({i+1})" for i, airport in enumerate(top_airports)]
ax.legend(legend_labels)

plt.show()













































