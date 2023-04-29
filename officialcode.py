# Let's import the required libraries!
import pandas as pd
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import seaborn as sns
import matplotlib.pyplot as plt

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

# Prepare data for barplot
df_us_flights = df_airplanes[['US_airport_code', 'Total_Flights']].groupby(by='US_airport_code').sum()
df_foreign_flights = df_airplanes[['Foreign_airport_code', 'Total_Flights']].groupby(by='Foreign_airport_code').sum()
df_flights_count = pd.concat([df_foreign_flights, df_us_flights])
df_flights_count.reset_index(inplace=True)
df_flights_count.sort_values(by='Total_Flights', ascending=False, inplace=True)


# Get the top 10 airport codes with the highest number of flights
top_10 = df_flights_count.head(10)

# Set the size of the figure for the bar plot
plt.figure(figsize=(12, 6))

# Create a bar plot of the top 10 airport codes with the highest number of flights
sns.barplot(data=top_10, x='index', y='Total_Flights')
plt.ticklabel_format(style='plain', axis='y')
plt.ylabel('Flights')
plt.xlabel('Top 10 Airports')
plt.title('Top 10 Airports in Flights')

# Display the bar plot
plt.show()

# Set the size of the figure for the box plot
plt.figure(figsize=(12, 6))

# Create a box plot of the 'Total_Flights' column
sns.boxplot(x='Total_Flights', data=df_airplanes)

# Set the title and labels for the box plot
plt.title('Box Plot of Total Flights')
plt.xlabel('Total Flights')

# Display the box plot
plt.show()




# ----- Now we will start creating a heatmap of the US Flights based on total flights -----
df_us_flights.reset_index(inplace=True)  # reset index
df_us_flights.columns = ['ariport_code', 'count']
df_foreign_flights.reset_index(inplace=True)
df_foreign_flights.columns = ['ariport_code', 'count']
df_flights_all = pd.concat([df_us_flights, df_foreign_flights])

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
for index, row in df_flights_all.iterrows():
    try:
        location.append(airports[row['ariport_code']])
    except KeyError:
        missed.append([row['ariport_code']])
        continue

df_missed = pd.DataFrame(missed)  # All missed airports will now be looped again to check if they have IATA format code
df_missed.columns = ['ariport_code']  # rename the column

location2 = []  # empty list to store all the airports that have an airport code in 'IATA' format
missed = []  # new empty missed list

# Now we try to catch all IATA format airport codes
airports = airportsdata.load('IATA')
for index, row in df_missed.iterrows():
    try:
        location2.append(airports[row['ariport_code']])
    except KeyError:
        missed.append([row['ariport_code']])
        continue

df_lid = pd.DataFrame(location)  # Dataframe with all LID format codes
df_iata = pd.DataFrame(location2)  # Dataframe with all IATA format codes
df_missed = pd.DataFrame(missed)  # Dataframe with all missed airport codes

# We created a new df that merges the coordinates with the airport id and the total number of flights
# Repeated both for LID and IATA format code
df_flights_all.columns = ['lid', 'n_flights']
df_lid = pd.merge(df_lid, df_flights_all, on='lid', how='outer')
df_lid.dropna(subset=['icao'], inplace=True)

df_flights_all.columns = ['iata', 'n_flights']
df_iata = pd.merge(df_iata, df_flights_all, on='iata', how='outer')
df_iata.dropna(subset=['icao'], inplace=True)


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

# Convert the "Date" column to a datetime object
df_airplanes['Date'] = pd.to_datetime(df_airplanes['Date'], format='%m/%d/%Y')

# Group the data by airport and date and aggregate the total number of flights
flights_per_airport = df_airplanes.groupby(['US_airport_code', 'Date'])['Total_Flights'].sum().reset_index()
flight_per_airp_foreign = df_airplanes.groupby(['Foreign_airport_code', 'Date'])['Total_Flights'].sum().reset_index()
flight_per_airp_foreign.columns = ('US_airport_code', 'Date' , 'Total_Flights')

flights_per_airport = pd.concat([flights_per_airport,flight_per_airp_foreign])
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




##### ------ Network analysis ---------- ########

import pandas as pd
import networkx as nx

# Get the top 15 airports for total flights
top_15_airports = df_flights_count.head(15)

# Create a subset of the original DataFrame containing only the top 15 airports
mask_us = df_airplanes['US_airport_code'].isin(top_15_airports['index'])
mask_foreign = df_airplanes['Foreign_airport_code'].isin(top_15_airports['index'])
subset_airplanes = df_airplanes[mask_us & mask_foreign]
sub_us = subset_airplanes.iloc[:,[4,-1]]
sub_us.columns=('code','flights')
sub_foreign = subset_airplanes.iloc[:,[7,-1]]
sub_foreign.columns=('code','flights')

df_names = df_heat_map.iloc[:,[1,3]]
df_names.columns=('code','flights')


sub_us = pd.merge(sub_us, df_names, on='code')
sub_foreign = pd.merge(sub_foreign, df_names, on='code')

airports_names = pd.concat([sub_us, sub_foreign])
airports_names = airports_names.iloc[:,[0,2]]
airports_names.drop_duplicates(inplace=True)
airports_names.columns=('code', 'city')






# Create an empty graph
G = nx.Graph()

# Add edges between airports based on the subset DataFrame
for _, row in subset_airplanes.iterrows():
    G.add_edge(row['US_airport_code'], row['Foreign_airport_code'], weight=row['Total_Flights'])

# Define the layout for the nodes
pos = nx.circular_layout(G)

# Calculate the node sizes based on the degree of the node
degree = dict(G.degree)
node_sizes = [degree[node] * 7 for node in G.nodes]

# Function to map edge weights to colors
def weight_to_color(weight):
    if weight <= 400:
        return 'black', 0.25
    elif weight <= 700:
        return 'green', 1.5
    else:
        return 'red', 1.5

# Create the edge traces
edge_traces = []

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = edge[2]['weight']
    color, width = weight_to_color(weight)

    edge_trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        line=dict(color=color, width=width),
        hoverinfo='none',
        mode='lines',
        showlegend=False,
    )

    edge_traces.append(edge_trace)

# Create the node trace
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    hovertext=[],
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        color='#1D92ED',
        size=node_sizes,
        line=dict(width=2, color='#6EC1E4'),
    ),
    textposition='middle center',
    textfont=dict(size=8, color='white', family="Arial"),
    showlegend=False
)

us_airport_total_flights = subset_airplanes.groupby(['US_airport_code']).sum()['Total_Flights']
foreign_airport_total_flights = subset_airplanes.groupby(['Foreign_airport_code']).sum()['Total_Flights']
airport_total_flights = pd.concat([us_airport_total_flights, foreign_airport_total_flights])

for node in G.nodes():
    x, y = pos[node]
    total_flights = airport_total_flights[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['text'] += tuple([node])
    node_trace['hovertext'] += tuple([f'{node}: {total_flights} total flights'])

    
# Create the legend traces
legend_trace_1 = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(color='black', size=10),
    legendgroup="edge_colors",
    showlegend=True,
    name="Weight <= 400",
)

legend_trace_2 = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(color='green', size=10),
    legendgroup="edge_colors",
    showlegend=True,
    name="Weight <= 700",
)

legend_trace_3 = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(color='red', size=10),
    legendgroup="edge_colors",
    showlegend=True,
    name="Weight > 700",
)

# Create the layout
layout = go.Layout(
    title=dict(text="Top 15 Airports by Total Flights", x=0.5, y=0.93, font=dict(size=21)),
    showlegend=True,
    legend=dict(title=dict(text="Legend")),
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)

# Create the figure
fig = go.Figure(data=edge_traces + [node_trace, legend_trace_1, legend_trace_2, legend_trace_3], layout=layout)

# Show the plot!
fig.write_html('output_graph.html')







### LET'S CREATE THE APP !!! ###

import plotly.graph_objects as go
import pandas as pd



# Define the external stylesheet
external_stylesheets = [dbc.themes.BOOTSTRAP]

# Define the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout
app.layout = html.Div([
    
    # First chart - Time series
    html.Div([
        html.H1("Airports Data Visualization"),
        html.P("Select Airports:", className="control_label"),
        dcc.Dropdown(
            id="airport_selector",
            options=[{"label": i, "value": i} for i in sorted(flights_per_airport['US_airport_code'].unique())],
            multi=True,
            value=["ATL", "ORD", "DFW", "DEN", "LAX"],  # Preselect the top 5 airports with the highest total flights
            className="dcc_control",
        ),
        html.P("Select Year:", className="control_label"),
        dcc.Dropdown(
            id="year_selector",
            options=[{"label": year, "value": year} for year in flights_per_airport['Date'].dt.year.unique()] + [{"label": "All Years", "value": "all"}],
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
    ),
    
    # Second chart - Boxplot
    html.Div([
        dcc.Dropdown(
            id="airport-dropdown",
            options=[{"label": i, "value": i} for i in sorted(df_flights_count['index'].unique())],
            value=df_flights_count['index'].head(10).tolist(),  # Default value is the top 10 airports
            multi=True
        ),
    ],
    className="pretty_container six columns",
    ),
    html.Div(
        [dcc.Graph(id="boxplot")],
        className="pretty_container six columns",
    ),
    
    # Third chart - Barplot
    html.Div([
        dcc.Dropdown(
            id="airport-dropdown2",
            options=[{"label": i, "value": i} for i in sorted(df_flights_count['index'].unique())],
            value=df_flights_count['index'].head(10).tolist(),  # Default value is the top 10 airports
            multi=True
        ),
    ],
    className="pretty_container six columns",
    ),
    html.Div(
        [dcc.Graph(id="barplot")],
        className="pretty_container six columns",
    ),
])

# Define the callback for the time series graph
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
        height=400,  # Set a fixed height for the plot in pixels
    )

    fig.update_traces(
        line=dict(width=2),
        marker=dict(size=6, symbol='circle', line=dict(width=1, color='black'))
    )

    return fig

# Define callback for updating the boxplot based on selected airports
@app.callback(
    Output("boxplot", "figure"),
    [Input("airport-dropdown", "value")]
)
def update_boxplot(selected_airports):
    filtered_df = df_airplanes[df_airplanes["US_airport_code"].isin(selected_airports) | df_airplanes["Foreign_airport_code"].isin(selected_airports)]

    fig = px.box(filtered_df, x="Total_Flights", title="Boxplot of Total Flights")

    return fig

# Define callback for updating the barplot based on selected airports
@app.callback(
    Output("barplot", "figure"),
    [Input("airport-dropdown2", "value")]
)
def update_barplot(selected_airports):
    filtered_df = df_flights_count[df_flights_count["index"].isin(selected_airports)]

    fig = px.bar(filtered_df, x="index", y="Total_Flights", title="Barplot of Total Flights", 
                 labels={"index": "Airports"},  # Change x-axis title
                 color='index',  # Use different colors for each airport
                 color_discrete_sequence=px.colors.qualitative.Set2 # Choose a color palette
                 )

    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug= True ,port=8055)







