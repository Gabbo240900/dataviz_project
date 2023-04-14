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


# # AGGIUNGI A QUELLI DI PRIMA SCRIVENDO A MANO LATITUDINE E LONGITUDINE


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





#####Time series#####

# Import necessary libraries
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go

# Convert the "Date" column to a datetime object
df_airplanes['Date'] = pd.to_datetime(df_airplanes['Date'], format='%m/%d/%Y')

# Group the data by airport and date and aggregate the total number of flights
flights_per_airport = df_airplanes.groupby(['US_airport_code', 'Date'])['Total_Flights'].sum().reset_index()

# Get the top 5 airports with the most total flights
top_airports = flights_per_airport.groupby('US_airport_code')['Total_Flights'].sum().nlargest(5).index

# Filter the data to only include the top 5 airports
flights_per_airport_top5 = flights_per_airport[flights_per_airport['US_airport_code'].isin(top_airports)]


# Create a dynamic line plot for each airport showing the total number of flights over time
fig = go.Figure()

# Add a line trace for each airport
for airport in top_airports:
    fig.add_trace(go.Scatter(x=flights_per_airport_top5[flights_per_airport_top5['US_airport_code'] == airport]['Date'],
                             y=flights_per_airport_top5[flights_per_airport_top5['US_airport_code'] == airport]['Total_Flights'],
                             name=airport,
                             mode='lines'))

# Customize the layout
fig.update_layout(title='Flights per Airport over Time',
                  xaxis_title='Date',
                  yaxis_title='Total Flights',
                  legend_title_text='Airport',
                  plot_bgcolor='rgba(245, 245, 245, 1)',
                  hovermode='x unified')

# Customize the line style and markers
fig.update_traces(line=dict(width=2),
                 marker=dict(size=6, symbol='circle', line=dict(width=1, color='black')))

# Save the figure as an interactive HTML file with zoom functionality enabled
config = {'scrollZoom': True}
pio.write_html(fig, file='dynamic_time_series_enhanced.html', config=config)











##########ci sto a prova





import networkx as nx

from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.palettes import Spectral10

from bokeh.models.graphs import NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models import GlyphRenderer, Circle, MultiLine
from bokeh.models.graphs import GraphRenderer

# Let's assume that df_airplanes is a pandas dataframe with information about flights

# Get the top 10 airports by total flights
top_airports = df_airplanes.groupby('US_airport_code')['Total_Flights'].sum().nlargest(10).index

# Create a new dataframe with only the top 10 airports
df_top_airports = df_airplanes[df_airplanes['US_airport_code'].isin(top_airports)]

# Group the data by date and airport and aggregate the total number of flights
df_top_airports = df_top_airports.groupby(['Date', 'US_airport_code'])['Total_Flights'].sum().reset_index()

# Create a graph object
G = nx.Graph()

# Add nodes for each airport
for airport in top_airports:
    G.add_node(airport)

# Add edges for each combination of airports
for i in range(len(df_top_airports)):
    for j in range(i+1, len(df_top_airports)):
        if df_top_airports.loc[i, 'Date'] == df_top_airports.loc[j, 'Date']:
            airport1 = df_top_airports.loc[i, 'US_airport_code']
            airport2 = df_top_airports.loc[j, 'US_airport_code']
            flights = df_top_airports.loc[i, 'Total_Flights'] + df_top_airports.loc[j, 'Total_Flights']
            if G.has_edge(airport1, airport2):
                G[airport1][airport2]['weight'] += flights
            else:
                G.add_edge(airport1, airport2, weight=flights)

# Convert the graph object to a pandas dataframe
df_edges = pd.DataFrame(G.edges(data=True), columns=['from', 'to', 'weight'])

# Add a column with the edge weight as a string for display purposes
df_edges['weight_str'] = df_edges['weight'].apply(lambda x: str(x['weight']))

# Convert the nodes to a pandas dataframe and add a column with the node degree (number of connections)
df_nodes = pd.DataFrame({'id': list(G.nodes)})
df_nodes['degree'] = df_nodes['id'].apply(lambda x: len(list(G.neighbors(x))))

# Define the data sources for the plot
node_source = ColumnDataSource(df_nodes)
edge_source = ColumnDataSource(df_edges)

# Define the plot figure
plot = figure(title='Top 10 Airports', x_range=(-1.2, 1.2), y_range=(-1.2, 1.2), tools='pan,wheel_zoom,box_zoom,reset')


# Create a new graph renderer
graph_renderer = GraphRenderer()

# Set the layout provider and graph layout
graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=pos)
graph_renderer.node_renderer.data_source.add(df_nodes['id'], 'id')
graph_renderer.node_renderer.data_source.add(df_nodes['degree'], 'degree')
graph_renderer.edge_renderer.data_source.data = edge_source.data

# Create the graph layout
pos = nx.spring_layout(G)
x, y = list(zip(*pos.values()))
df_nodes['x'] = x
df_nodes['y'] = y
df_edges['xs'] = [[pos[start][0], pos[end][0]] for start, end in zip(df_edges['from'], df_edges['to'])]
df_edges['ys'] = [[pos[start][1], pos[end][1]] for start, end in zip(df_edges['from'], df_edges['to'])]

# Set node attributes
circle = Circle(size=15, fill_color=Spectral10[0])
graph_renderer.node_renderer.glyph = circle
graph_renderer.node_renderer.data_source.data = node_source.data
graph_renderer.node_renderer.data_source.data['size'] = df_nodes['degree']*2

# Set edge attributes
edge_attrs = {'xs': 'xs', 'ys': 'ys', 'line_width': 'weight', 'line_alpha': 0.8}
multi_line = MultiLine(line_color=Spectral10[1], **edge_attrs)
graph_renderer.edge_renderer.glyph = multi_line
graph_renderer.edge_renderer.data_source.data = edge_source.data
graph_renderer.edge_renderer.data_source.data['weight_str'] = df_edges['weight_str']

# Add hover tooltips for the nodes and edges
hover_node = HoverTool(tooltips=[('Airport', '@id'), ('Degree', '@degree')])
hover_edge = HoverTool(tooltips=[('From', '@start'), ('To', '@end'), ('Weight', '@weight_str')])
plot.add_tools(hover_node, hover_edge)

# Add the graph renderer to the plot
plot.renderers.append(graph_renderer)

# Show the plot in a notebook
output_notebook()
show(plot)
































