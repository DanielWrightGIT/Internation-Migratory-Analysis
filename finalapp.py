import os
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from datetime import datetime
import plotly.express as px

migrant_App = pd.read_csv('migrant_clean.csv', index_col=None)
timeseriesdf = pd.read_csv('timeseriesdf.csv')
timeseriesdf['date'] = pd.to_datetime(timeseriesdf['date'])
timeseriesdf.set_index('date', inplace=True)

column_names = migrant_App.columns
print(column_names)

st.set_page_config(page_title="Migrant Web App", page_icon=":tada:", layout="wide")

# Create a multiselect for navigation
selected_sections = st.sidebar.selectbox("Select Sections", ["Introduction", "Modeling"])

introduction ="""
        
         Migration is a globally significant phenomenon with far-reaching impacts on 
         economies and societies. Our project analyzes the Global Missing Migrants 
         dataset to uncover the key factors contributing to migration incidents and 
         develop predictions for future occurrences. This application empowers users 
         to understand the likelihood of such incidents, providing valuable insights 
         for a safer migration experience and informed policy development.
         
         Introducing our app, it offers two core features: Exploratory Data Analysis 
         (EDA) and Time Series Analysis. Our EDA section provides insights into the 
         dataset, while the Time Series component allows users to predict future 
         migration incidents based on input parameters. """

if "Data Exploration (EDA)" in selected_sections:
    st.write("""To the right, you will find a menu of user inputs where you can explore and analyze the historical data related to missing migrants. 
    You can filter the data based on different criteria such as location, category, number of people, gender, and more. This analysis will help you 
    gain insights into the patterns and characteristics of incidents involving missing migrants. Additionally, you can visualize the main causes of 
    incidents in various migration routes. """)

    # Sidebar
    st.sidebar.header("User Inputs")
    incident_year = st.sidebar.slider("Select Incident Year", 2014, 2023)
    region_of_origin = st.sidebar.selectbox("Select Region of Origin", migrant_App["region of origin"].unique())
    number_of_males = st.sidebar.number_input("Number of Males", min_value=0)

    # Main content
    st.header("Migrant Data Analysis")

    # Group data by "Region of Origin"
    grouped_data = migrant_App.groupby("region of origin").agg({
        "number of males": "sum",
        "number of females": "sum",
        "number of children": "sum",
        "total number of dead and missing": "sum"
    })

    # Display the aggregated data
    st.write("Aggregated Data by Region:")
    st.write(grouped_data)

    # Plot the results (we can customize othr chart type)
    st.bar_chart(grouped_data)

    # User inputs
    st.write("User Inputs:")
    st.write(f"Incident Year: {incident_year}")
    st.write(f"Region of Origin: {region_of_origin}")
    st.write(f"Number of Males: {number_of_males}")

    # Filter data based on user inputs
    filtered_data = migrant_App[(migrant_App["incident year"] == incident_year) &
                            (migrant_App["region of origin"] == region_of_origin) &
                            (migrant_App["number of males"] == number_of_males)]

    if not filtered_data.empty:
        # Display filted data
        st.subheader("Analysis Results")
        st.write("Number of Dead:", filtered_data["number of dead"].values[0])
        st.write("Estimated number of Missings:", filtered_data["minimum estimated number of missing"].values[0])
        st.write("Number of Dead and Missing:", filtered_data["total number of dead and missing"].values[0])
        st.write("Number of Survivors:", filtered_data["number of survivors"].values[0])
        st.write("Number of Females:", filtered_data["number of females"].values[0])
        st.write("Number of Children:", filtered_data["number of children"].values[0])
        st.write("Cause of Death:", filtered_data["cause of death category"].values[0])
        st.write("Country of Death:", filtered_data["extracted country"].values[0])
    else:
        st.warning("No data available for the selected inputs.")

# Additional features..............
# Check if "Modeling" is in the selected sections
elif "Modeling" in selected_sections:
    # Header
    st.write("To the right you will see a menu of user inputs where you can input the time and migration route you are planning on taking, this will then output an estimation of number of incidents based on the historicals. To the right of this timeseries you will also see the main causes of death in the migration route")
    # Sidebar
    st.sidebar.header("User Inputs")
    planned_migration_date = st.sidebar.date_input("Input planned migration date", value="today", min_value=None, max_value=None, format="YYYY-MM-DD")

    migration_route = st.sidebar.selectbox("Select Migration Route", timeseriesdf["migration route"].unique())
    st.subheader("Time Series Model")
    #function that returns the level of the migration route inputted
    def getLevelOfRoute(route, timeseriesdf):
        level = timeseriesdf[timeseriesdf['migration route'] == route]['label_level'].values[0]
        return level

    #run the function with the migration route inputted   
    ts_level = getLevelOfRoute(migration_route, timeseriesdf)
    
    #function to get all the df entries with the same level 
    def getClusterLabel(level, timeseries):
        return (timeseries[timeseries['label_level'] == level])

    #function that extracts all the routes in the same cluster and groups them by the target variable.
    def preprocess_level_timeseries(level, timeseries):
        # Get the 'level' timeseries
        level_timeseries = getClusterLabel(level, timeseries)
    
        # Drop the 'date' column
        level_timeseries = level_timeseries.drop(['date.1'], axis=1)
    
        # Group by date and sum the 'total number of dead and missing'
        level_timeseries = level_timeseries.groupby(level_timeseries.index)['total number of dead and missing'].sum()
    
        return level_timeseries

    leveldf = preprocess_level_timeseries(ts_level, timeseriesdf)
    
    #the parameters will depend on the level selected 
    #if ts_level = 'level1' or then app_order = (0, 1, 1) 
    #if ts_level = 'level2' or ts_level = 'level5' then app_order = (1, 0, 0) 
    #if ts_level = 'level3' or ts_level = 'level4' then app_order = (0, 1, 1) 
    if ts_level == 'level1':
        app_order = (1, 1, 1)
    elif ts_level == 'level2' or ts_level == 'level5':
        app_order = (1, 0, 0)
    elif ts_level == 'level3' or ts_level == 'level4':
        app_order = (0, 1, 1)

    #function that gets the number of periods for the sarima model 

    def calculate_months_difference(date1, date2):
        date1 = pd.to_datetime(date1, format="%Y-%m-%d")
        date2 = pd.to_datetime(date2, format="%Y-%m-%d")

        rdelta = relativedelta(date2, date1)
        months_difference = rdelta.years * 12 + rdelta.months

        return months_difference

    last_date = leveldf.index[-1]
    target_date = planned_migration_date
    months_difference = calculate_months_difference(last_date, target_date)

    #function that sets the sarima timeseries model 
    def sarima_forecast(level_timeseries,forecast_months, order, seasonal_order, plot_title="Forecast"):
        # Create the SARIMA model
        level_sarima_model = sm.tsa.SARIMAX(level_timeseries, order=order, seasonal_order=seasonal_order)
        level_sarima_model_fit = level_sarima_model.fit()

        # Make forecasts
        forecasts = level_sarima_model_fit.get_forecast(steps=forecast_months)
        predicted_values = forecasts.predicted_mean
        predicted_values.index = pd.date_range(start=last_date, periods=forecast_months, freq='M')

        return predicted_values
    #run the sarima function
    predicted_values = sarima_forecast(leveldf, months_difference, order=app_order, seasonal_order=(1, 1, 1, 12), plot_title="Migrant Incident Forecast")

    #function to retrieve the output of the timeseries for the inputted date
    def get_values_for_year_month(indexes, values, year, month):
        matching_values_string = ""  # Initialize an empty string
        for i, date_index in enumerate(indexes):
            if date_index.year == year and date_index.month == month:
                matching_values_string += str(values[i])  # Convert values to strings and append to the string
        return matching_values_string

    # Example usage:
    year_to_find = planned_migration_date.year
    month_to_find = planned_migration_date.month
    matching_values = get_values_for_year_month(predicted_values.index, predicted_values, year_to_find, month_to_find)
    st.write(f"Estimated number of incidents for the planned migration date {month_to_find}/{year_to_find}: {matching_values}")

    #declaration of columns
    col1, col2 = st.columns(2)
    
    col1.subheader(f'{migration_route} Number of Incidents Forecast')
    col1.write(f"The migration route was classified as a danger {ts_level}.")

    # Create a new figure using Plotly Express
    fig = px.line()
    # Add historical data to the plot
    fig.add_scatter(x=leveldf.index, y=leveldf, name="Historical Data")
    # Add SARIMA forecast to the plot
    fig.add_scatter(x=predicted_values.index, y=predicted_values, name="SARIMA Forecast", line=dict(color='red'))
    # Customize the layout (titles, labels, etc.)
    fig.update_layout(
        title="Migrant Incident Forecast",
        xaxis_title="Year",
        yaxis_title="Total Number of Incidents",
        width=500,  # Set the width in pixels
        height=400  # Set the height in pixels
    )
    
    # Show the interactive plot
    col1.plotly_chart(fig)

    # Filters the data based on the selected migration route
    filtered_data_ts = timeseriesdf[timeseriesdf['migration route'] == migration_route]
    # filtered_data = migrant_App[migrant_App['cause of death category'] == cause_of_death]

    # Creates the histogram
    col2.subheader(f'Causes of Death for {migration_route}')
    fig = px.histogram(filtered_data_ts, x='cause of death category')
    fig.update_layout(
        width=600,  # Set the width in pixels
        height=400  # Set the height in pixels
    )
    col2.plotly_chart(fig)

else: 
    # Header Section
    st.header("Migration Data Analysis")
    st.subheader("Welcome to the Migration Incident Prevention App")
    st.write(introduction)
    
    
# Footer
st.sidebar.text("© 2023 Migrant Data Analysis App")

#!streamlit run appteam3.py --server.port=8080 --browser.serverAddress='0.0.0.0'
