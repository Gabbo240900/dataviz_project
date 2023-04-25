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

# Convert 'Date' column to datetime format
df_airplanes['Date'] = (pd.to_datetime(df_airplanes['Date'], format='%m/%d/%Y', errors='coerce'))

# Create a boxplot to check for outliers
fig, ax = plt.subplots()
ax.boxplot(df_airplanes['Total_Flights'])

# Add labels and title
ax.set_title('Boxplot ')
ax.set_ylabel('Total_Flights')

plt.show()
# Observations show more than 1000 flights in some days, with a maximum of 2000 in May '96.
# These should not be removed, but further analyzed to understand why there were so many flights during those days.





# Barplot with airports and number of flights for each airport
df_us_flights = df_airplanes[['US_airport_code', 'Total_Flights']].groupby(by='US_airport_code').sum()
# Group by US_airport and sum the number of flights
df_foreign_flights = df_airplanes[['Foreign_airport_code', 'Total_Flights']].groupby(by='Foreign_airport_code').sum()
# Group by foreign and sum the number of flights

df_flights_count = pd.concat([df_foreign_flights, df_us_flights])
# Concatenate the two datasets to get a single one with all the flights
df_flights_count.reset_index(inplace=True)
# Reset the index to set the airport code as a column

df_flights_count.sort_values(by='Total_Flights', ascending=False, inplace=True)
# Sort based on flights for creating a barplot of the top 10 airports

top_10_airp = df_flights_count.head(10)
# New dataset with only the top 10 airports

# Barplot with seaborn
sns.barplot(data=top_10_airp, x="index", y="Total_Flights")
plt.xlabel('Airport')
plt.title('Flights per Airport')
plt.ticklabel_format(style='plain', axis='y')
plt.show()





# ----- Now we will start creating a heatmap of the US Flights based on total flights -----

df_us_flights.reset_index(inplace=True)  # reset index

# We retrieved the coordinates by using another dataframe that links the airport identification name to the coordinates
# pip install airportsdata
import airportsdata  # new library for retrieving the coordinates

# We discovered that there are various formats for airport codes
# We have both LID format and IATA format.
# In the next step, we will try to catch as many airports as possible

# Dictionary with all the information we need
airports = airportsdata.load('LID')

location = []  # empty list to store all the airports that have an airport code in 'LID' format
missed = []  # list with all missed airports

# Error management for loop, to get all the locations with LID format
for index, row in df_us_flights.iterrows():
    try:
        location.append(airports[row['US_airport_code']])
    except KeyError:
        missed.append([row['US_airport_code']])
        continue

df_missed = pd.DataFrame(missed)  # All missed airports will now be looped again to check if they have IATA format code
df_missed.columns = ['US_airport_code']  # rename the column

location2 = []  # empty list to store all the airports that have an airport code in 'IATA' format
missed = []  # new empty missed list

# Now we try to catch all IATA format airport codes
airports = airportsdata.load('IATA')
for index, row in df_missed.iterrows():
    try:
        location2.append(airports[row['US_airport_code']])
    except KeyError:
        missed.append([row['US_airport_code']])
        continue

df_lid = pd.DataFrame(location)  # Dataframe with all LID format codes
df_iata = pd.DataFrame(location2)  # Dataframe with all IATA format codes
df_missed = pd.DataFrame(missed)  # Dataframe with all missed airport codes

# We created a new df that merges the coordinates with the airport id and the total number of flights
# Repeated both for LID and IATA format code
df_us_flights.columns = ['lid', 'n_flights']
df_lid = pd.merge(df_lid, df_us_flights, on='lid', how='outer')
df_lid.dropna(subset=['icao'], inplace=True)

df_us_flights.columns = ['iata', 'n_flights']
df_iata = pd.merge(df_iata, df_us_flights, on='iata', how='outer')
df_iata.dropna(subset=['icao'], inplace=True)



# df_us_fligths.columns = ['US_airport_code', 'n_fligths']
# df_missed.columns = ['US_airport_code']

# df_missed = pd.merge(df_missed, df_us_fligths)

# airports['LKE']



# Concatenate the 'df_lid' and 'df_iata' dataframes to create the final dataset with all caught airports
df_heat_map = pd.concat([df_lid, df_iata])

# Select the required information for the heatmap (latitude, longitude, and number of flights)
lat_long_flights = df_heat_map.iloc[:, [7, 8, 11]]

# Import the necessary libraries for creating the heatmap
# pip install folium
import folium
from folium.plugins import HeatMap

# Create a folium map object centered on the United States with an initial zoom level
map_obj = folium.Map(location=[38.27312, -98.5821872], zoom_start=5)

# Add the heatmap layer to the map object using the 'lat_long_flights' dataframe
HeatMap(lat_long_flights).add_to(map_obj)

# Save the heatmap to an HTML file for visualization
map_obj.save("us_flights_map.html")





######################### Time series #######################

import plotly.graph_objs as go
import plotly.io as pio
# import dashpip
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash


# Convert the "Date" column to a datetime object
df_airplanes['Date'] = pd.to_datetime(df_airplanes['Date'], format='%m/%d/%Y')

# Group the data by airport and date and aggregate the total number of flights
flights_per_airport = df_airplanes.groupby(['US_airport_code', 'Date'])['Total_Flights'].sum().reset_index()

# Reindex the data to include missing months with 0 flights
flights_per_airport = flights_per_airport.set_index(['US_airport_code', 'Date']).unstack(level=-1, fill_value=0).stack().reset_index()

# Get all unique airport codes
all_airports = flights_per_airport['US_airport_code'].unique().tolist()

# Create airport selection options
airport_options = [{"label": airport, "value": airport} for airport in all_airports]

# Find the top 5 airports with the highest total flights
top_airports = flights_per_airport.groupby('US_airport_code')['Total_Flights'].sum().nlargest(5).index.tolist()

# Get all unique years from the dataset
all_years = sorted(flights_per_airport['Date'].dt.year.unique().tolist())

# Create year selection options
year_options = [{"label": year, "value": year} for year in all_years]
year_options.insert(0, {"label": "All Years", "value": "all"})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = html.Div([
    html.Div([
        html.P("Select Airports:", className="control_label"),
        dcc.Dropdown(
            id="airport_selector",
            options=airport_options,
            multi=True,
            value=top_airports,  # Preselect the top 5 airports with the highest total flights
            className="dcc_control",
        ),
        html.P("Select Year:", className="control_label"),
        dcc.Dropdown(
            id="year_selector",
            options=year_options,
            value="all",  # Preselect the "All Years" option
            className="dcc_control",
        )
    ],
    className="pretty_container two columns",
    id="cross-filter-options",
    ),
    html.Div(
        [dcc.Graph(id="time_series_graph")],
        className="pretty_container ten columns",
        style={"height": "90vh"}  # Set the height of the right container to 90% of the viewport height
    ),
])

@app.callback(
    Output("time_series_graph", "figure"),
    [Input("airport_selector", "value"), Input("year_selector", "value")]
)
def update_time_series(selected_airports, selected_year):
    fig = go.Figure()

    filtered_data = flights_per_airport[flights_per_airport['US_airport_code'].isin(selected_airports)]

    # Filter data by the selected year, if applicable
    if selected_year != "all":
        filtered_data = filtered_data[filtered_data['Date'].dt.year == selected_year]

    for airport in selected_airports:
        fig.add_trace(go.Scatter(
            x=filtered_data[filtered_data['US_airport_code'] == airport]['Date'],
            y=filtered_data[filtered_data['US_airport_code'] == airport]['Total_Flights'],
            name=airport,
            mode='lines'
        ))

    fig.update_layout(
        title='Flights per Airport over Time',
        xaxis_title='Date',
        yaxis_title='Total Flights',
        legend_title_text='Airport',
        plot_bgcolor='rgba(245, 245, 245, 1)',
        hovermode='x unified',
        uirevision='same',  # Preserve the user's zoom level when updating the graph
        yaxis=dict(range=[0, None]),  # Set the minimum value of the y-axis to 0
        autosize=True,  # Enable autosize to make the graph responsive
        margin=dict(l=50, r=50, b=50, t=50, pad=4),
        height=800,  # Set a fixed height for the plot in pixels
    )

    fig.update_traces(
        line=dict(width=2),
        marker=dict(size=6, symbol='circle', line=dict(width=1, color='black'))
    )

    return fig

app.run_server(mode='external')


##### ------ Network analysis ---------- ########
import networkx as nx

top_10_airport = top_10_airp['index'].values.tolist()
df_network_us = df_airplanes.loc[df_airplanes['US_airport_code'].isin(top_10_airport)]
df_network_for = df_airplanes.loc[df_airplanes['Foreign_airport_code'].isin(top_10_airport)]
df_network = pd.concat([df_network_us, df_network_for])


G = nx.from_pandas_edgelist(df_network, source='US_airport_code', target='Foreign_airport_code',
                            edge_attr='Total_Flights')



G.nodes()
G.edges()
print('Nodes: ', G.number_of_nodes(), 'Edges: ', G.number_of_edges())
#print both the number of nodes and edges


import matplotlib.pyplot as plt


pos = nx.kamada_kawai_layout(G)
plt.figure(figsize=(20,20), dpi=300) #set picture resolution

node_sizes = [len(list(G.neighbors(n))) * 100 for n in G.nodes()] #size the nodes based on their degree
nx.draw_networkx_nodes(G, pos, node_size=node_sizes)

# edges
nx.draw_networkx_edges(G, pos,width=1, alpha = 0.5)

nx.draw_networkx_labels(G, pos, font_size=20, alpha = 0.8)


plt.tight_layout()
plt.axis("off")
plt.title('Airports', fontsize=30)

plt.show()

















