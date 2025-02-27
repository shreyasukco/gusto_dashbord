GUSTO DASHBOARD

Overview

Gusto Dashboard is a data visualization and analysis tool built using Streamlit. It provides insights into sales, collections, and pending invoices by processing and filtering Excel and JSON datasets.

Features

Data Processing & Cleaning: Handles missing values, date formatting, and data transformation.

Dynamic Filters: Users can filter data by date, month, and other relevant parameters.

Charts & Visualizations: Uses seaborn, matplotlib, and plotly to generate insightful plots.

Geospatial Mapping: Displays outlet locations on a Folium map.

Pending Invoice Analysis: Tracks pending collections and generates reports.

Interactive UI: Uses Streamlit components for enhanced user interaction.

Dependencies

Ensure you have the following Python libraries installed:

pip install streamlit pandas seaborn matplotlib plotly openpyxl scikit-learn folium babel geopy rapidfuzz streamlit_option_menu streamlit_dynamic_filters

File Structure

Gusto_my_record_18012025.xlsx - Main dataset.

collectiondata.xlsx - Collection details.

outleterpidwithmonths.xlsx - Monthly ERP data.

collection_map_unmatched_date.xlsx - Unmatched collection records.

data.json - Order count details.

app.py - Streamlit application script.

Running the Dashboard

To start the Streamlit dashboard, use:

streamlit run app.py

Usage

Upload Data: Ensure all Excel files are available in the working directory.

Apply Filters: Use sidebar filters to select a date range and months.

View Insights: Analyze charts, tables, and maps.

Export Data: Download filtered reports for further analysis.

Error Handling

If a required file is missing, an error message is displayed.

Any invalid or missing data is handled using fillna() or errors='coerce' where applicable.

Future Enhancements

Add ML-based predictive analytics.

Improve UI with advanced filtering options.

Integrate API endpoints for real-time data updates.

Contributors

Developer: Shreyas S P