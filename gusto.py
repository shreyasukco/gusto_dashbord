import streamlit as st 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
import re
import plotly.graph_objects as go
import streamlit_dynamic_filters as DynamicFilters
from streamlit_dynamic_filters import DynamicFilters
from datetime import datetime, timedelta
import pandas as pd
from sklearn.cluster import KMeans
import folium 
from streamlit_folium import st_folium 
from scipy.spatial.distance import cdist 
from babel.numbers import format_currency 
from datetime import datetime
from geopy.distance import geodesic
from rapidfuzz.fuzz import ratio 
import time
from streamlit_option_menu import option_menu 
import streamlit.components.v1 as components
from rapidfuzz import fuzz
import itertools
from datetime import datetime

st.set_page_config(
    page_title='GUSTO DASHBOARD',          
    page_icon=None,           
    layout="wide",        
    initial_sidebar_state="auto"  
)

@st.cache_data
def read_data1(file):
    df = pd.read_excel(file)
    return df
try:
    dff = read_data1('Gusto_my_record_18012025.xlsx')
    dp=read_data1("collectiondata.xlsx")
    dm=read_data1("outleterpidwithmonths.xlsx")
    dk=read_data1("collection_map_unmatched_date.xlsx")

except FileNotFoundError:
    st.error("The file  was not found. Please check the file path and try again.")
    st.stop()
ordercount = pd.read_json("data.json")    
dp['Collection date'] = pd.to_datetime(dp['Collection date'], dayfirst=True, errors='coerce')
dm = dm[dm["Month"].notna() & (dm["Month"] != "")]

dp['Timestamp'] = pd.to_datetime(dp['Date'])
dp['Month_Year'] = dp['Timestamp'].dt.strftime('%B %Y')


dp['Particulars'] = dp['Particulars'].str.strip().str.lower()
dk['Particulars'] = dk['Particulars'].str.strip().str.lower()
# df3=pd.read_excel("beat_total11022025withnovdec.xlsx") 
store_mapping = {
    '43876506  happy mart super market': '438765036 Happy mart super market',
    '49600875   i fersh': '496000875  I Fersh',
    'new  arafa stores': '478667139 New Arafa Stores',
    'supershavanath  provision':"477424641 Supershavanath Provision",
    'supershavanath  provision':"477424641 Supershavanath Provision"
}
dp.drop(dp[(dp["Particulars"] == "(cancelled)") | (dp["Particulars"] == "(cancelled )")].index, inplace=True)
mapping = dk.set_index('Particulars')['particulars2']
# st.write(mapping)
# dp['Particulars'] = dp['Particulars'].map(dk.set_index('Particulars')['particulars2']).fillna(dp['Particulars'])
dp['Particulars_2'] = dp['Particulars'].map(mapping).fillna(dp['Particulars'])
dp['Particulars_2'] = dp['Particulars_2'].replace(store_mapping)
df1 = dp
# st.write(df1) 

ordercount= ordercount.rename(columns={"Outlets Erp Id": "Outlet Erp Id"})
# st.write(ordercount)
current_date = datetime.now().date()
formatted_date = current_date.strftime("%d/%m/%y")

# Ensure 'Collection date' is in datetime format
df1['Collection date'] = pd.to_datetime(df1['Collection date'], dayfirst=True, errors='coerce')

# Convert 'Collection date' to date only (remove time)
df1['Collection date'] = df1['Collection date'].dt.date
# st.write(df1)
start_date = datetime.strptime("11-01-2025", "%d-%m-%Y").date()
end_date = datetime.today().date()


filtered_df = df1[(df1['Collection date'] >= start_date) & (df1['Collection date'] <= end_date)]

grouped = filtered_df.groupby('Collection date')[['Return', 'Total Collection','Bad Debts','Canceled Bills']].sum().reset_index()
# st.write(grouped)
grouped["Total Collection"] -= (grouped["Bad Debts"] + grouped["Return"] + grouped["Canceled Bills"])

# Sort the data by 'Collection date' in descending order
grouped = grouped.sort_values(by='Collection date', ascending=False)

start_date_sidebar = st.sidebar.date_input("Start Date", grouped['Collection date'].min())
end_date_sidebar = st.sidebar.date_input("End Date", grouped['Collection date'].max())
month_mapping = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
    "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"
}
dm['Full_Month'] = dm['Month'].map(month_mapping) + " 2024"

months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_year_options = dp['Month_Year'].unique()

month_year_options = sorted(dp['Month_Year'].unique(), key=lambda x: pd.to_datetime(x))

start_month, end_month = st.select_slider(
    "Select Month-Year Range",
    options=month_year_options,
    value=(month_year_options[0],month_year_options[-1])  # Set default value"December 2024"s
)

dp = dp[
    (dp['Month_Year'].apply(lambda x: month_year_options.index(x)) >= month_year_options.index(start_month)) &
    (dp['Month_Year'].apply(lambda x: month_year_options.index(x)) <= month_year_options.index(end_month))
]
selected_months = [
    month for month in month_year_options 
    if month_year_options.index(start_month) <= month_year_options.index(month) <= month_year_options.index(end_month)
]

dm = dm[dm['Full_Month'].isin(selected_months)]

filtdf = grouped[(grouped['Collection date'] >= start_date_sidebar) & (grouped['Collection date'] <= end_date_sidebar)]

# Calculate total Return and Total Collection for the filtered data
amount = filtdf['Return'].sum()
collections = filtdf['Total Collection'].sum()

# st.write(f"Date: {selected_date_str}, Amount: {amount}, Collections: {collections}")
return_amount = format_currency(amount, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
collection_amount = format_currency(collections, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
dp.columns = dp.columns.str.strip()

dff["Google Maps Link"] = dff.apply(
    lambda row: f"https://www.google.com/maps?q={row['Latitude']},{row['Longitude']}",
    axis=1
)
dff["Last Modified On"] = pd.to_datetime(dff["Last Modified On"])

cutoff_date = datetime.now() - timedelta(days=9*30)

dff["Is Active"] = dff["Last Modified On"].apply(lambda x: "Active" if x >= cutoff_date else "Inactive")

dff['Particulars_number'] = dff['Outlet Erp Id'].str.extract(r'(\d{9})')

dff = dff[dff['Particulars_number'].notna()]
dff = dff.drop_duplicates(subset='Particulars_number', keep='first')

dp['Particulars_number'] = dp['Particulars_2'].str.extract(r'(\d{9})')

df_missing_particulars = dp[dp["Particulars_number"].isna()]


dff['LAT'] = dff['Latitude'].astype(float)
dff['LONG'] = dff['Longitude'].astype(float)

dff.dropna(subset=['LAT', 'LONG'], inplace=True)

dp['Sum of Diff'] = pd.to_numeric(dp['Diff'], errors='coerce')
# st.write(dp)
dp['invoice clear'] = dp['Sum of Diff'].apply(lambda x: 0 if x <= 0  else 1)
# st.write(dp)
monthvisecountofnoneclrbills=dp.groupby("Month",as_index=False)[["Gross Total",'Sum of Diff', 'invoice clear']].sum()
monthvisecountofnoneclrbills["Month"]=pd.Categorical(monthvisecountofnoneclrbills['Month'], categories=months_list, ordered=True)
monthvisecountofnoneclrbills = monthvisecountofnoneclrbills.sort_values('Month')
# st.write(monthvisecountofnoneclrbills)
sumofgraosstotall=monthvisecountofnoneclrbills["Gross Total"].sum()
v=monthvisecountofnoneclrbills["invoice clear"].sum()
dp['Sum of Diff'] = dp['Sum of Diff'].fillna(0)
grouped_dp = dp.groupby('Particulars_number', as_index=False)[['Sum of Diff', 'invoice clear']].sum()
m=grouped_dp['invoice clear'].sum()
grouped_dp.rename(columns={'Sum of Diff': 'Total Pending'}, inplace=True)

result = grouped_dp.merge(dff, on='Particulars_number', how='left')
sum_empty_pending = result.loc[result['MyBeat Plan'].isna(), 'Total Pending'].sum()

dff = dff.merge(grouped_dp, on='Particulars_number', how='left')
dff["invoice clear"] = dff["invoice clear"].fillna(0)

dff['Pending_Status'] = dff['Total Pending'].apply(
    lambda x: "Pending" if pd.notna(x) and x > 0 else "No pending"
)

gdp = dm.groupby('Outlet Erp Id', as_index=False)['Total Pending(11/01/25)'].sum()
dff = dff.merge(gdp, on='Outlet Erp Id', how='left')
dff["Total Pending(11/01/25)"] = dff["Total Pending(11/01/25)"].fillna(0)
dff["Total Pending"] = dff["Total Pending"].fillna(0)
def process_dataframe(df):
    try:
        # Ensure the required columns exist
        if 'Total Pending(11/01/25)' in df.columns and 'Total Pending' in df.columns:
            # Convert columns to numeric to avoid errors
            df['Total Pending(11/01/25)'] = pd.to_numeric(df['Total Pending(11/01/25)'], errors='coerce').fillna(0)
            df['Total Pending'] = pd.to_numeric(df['Total Pending'], errors='coerce').fillna(0)

            # Calculate 'Collected Amount' (must not be negative)
            df['Collected Amount'] = (df['Total Pending(11/01/25)'] - df['Total Pending']).clip(lower=0)

            def calculate_pending_percent(row):
                initial_pending = row['Total Pending(11/01/25)']
                current_pending = row['Total Pending']

                if initial_pending <= 0:
                    return "0.00%" if current_pending <= 0 else "100.00%"  # If no initial pending, consider as 100%

                if current_pending <= 0:
                    return "0.00%"  # No pending amount left
                elif current_pending < initial_pending:
                    percent = (current_pending / initial_pending) * 100
                    return f"{percent:.2f}%"  # Actual pending percentage
                else:
                    return "100.00%"  # Everything still pending or over-pending

            # Apply function to DataFrame
            df['Pending_Percent_diff'] = df.apply(calculate_pending_percent, axis=1)

        else:
            return "Required columns are missing in the DataFrame."

    except Exception as e:
        return f"An error occurred while processing the DataFrame: {e}"

    return df

dff = process_dataframe(dff)    
dd=dff
mm=dff['invoice clear'].sum() 
# st.write(mm) 
dff = dff.merge(ordercount, on='Outlet Erp Id', how='left')
dff["repeated_orders"] = dff.apply(lambda x: "Repeated" if (x["Pending_Status"] == "Pending") or (x["invoice clear"] > 2) else "No reapeat", axis=1)
dynamic_filters = DynamicFilters(dff, filters=["Territory","Final_Beats","Is Active","Pending_Status","Outlets Name"])    
dynamic_filters.display_filters(location='sidebar')
df = dynamic_filters.filter_df()

center_lat = np.mean(df['LAT']) 
center_lon = np.mean(df['LONG'])
# filtered_df['Combined Category'] = filtered_df['Brand Presence'].astype(str) + ' | ' + filtered_df['Milk Products SKUs'].astype(str)
custom_color_map = {
    "Pending": "red",        
    "No pending": "green",    
      
}
fig = px.scatter_map(
    df, 
    lat="LAT", 
    lon="LONG", 
    hover_name="Final_Beats",
    hover_data=["Outlets Name","Zone","Region","Territory","Outlets Type"] ,
    zoom=10,  # Default zoom level; you can adjust this for a better view
    center={"lat": center_lat, "lon": center_lon},  # Center the map at a specific lat/lon
    height=600,  # Set the height of the map
    color="Pending_Status",
    # color_discrete_sequence=px.colors.qualitative.Set1,
    color_discrete_map=custom_color_map,    
)
fig.update_traces(marker=dict(size=12, opacity=0.8))

# Set the map style (open street map is interactive with drag/zoom)
fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(
    mapbox=dict(
        center={"lat": center_lat, "lon": center_lon},
        zoom=10,  # Initial zoom level, user can zoom in/out
        accesstoken="your_mapbox_access_token"  # Optional: If using Mapbox's proprietary styles
    ),
    margin={"r":0,"t":0,"l":0,"b":0},
    # Remove margins to ensure full map area
)

total_rows = dff.shape[0]
total_rows1=df.shape[0]


outlet_counts = df['Beats'].value_counts()
outlet_counts1 = dff['Final_Beats'].value_counts()
pending_outlets_count=df[df["Pending_Status"]=="Pending"].shape[0]

non_null_count = int(result['Total Pending'].sum())
nn=int(df["Total Pending(11/01/25)"].sum())
# st.write(nn)
formatted_amount5 = format_currency(nn, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
total_matched_amount=int(non_null_count-sum_empty_pending)
formatted_amount = format_currency(total_matched_amount, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
# formatted_amount = f"₹{total_matched_amount:,}".replace(',', ',')
n= int(df['Total Pending'].sum() )

formatted_amount3 = format_currency(n, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
dff = dff.drop_duplicates(subset="Outlets Name", keep="first")
new_df = dp[dp['Particulars_number'].isnull() | (dp['Particulars_number'] == '')]
new_df = new_df.reset_index(drop=True)
new_df["Particulars_number"] = new_df["Particulars"].map(
    dff.set_index("Outlets Name")["Particulars_number"]
)
non_null_count1 = new_df['Sum of Diff'].sum()
r=int(non_null_count1+sum_empty_pending)
formatted_amount2 = format_currency(r, "INR", locale="en_IN", currency_digits=False, format="¤#,##,##0")
# collected_amount=[1442607,]
cleaned_amount = int(formatted_amount.replace('₹', '').replace(',', ''))

json_file_path = 'matched_amount.json'
today_date = datetime.now().strftime('%Y-%m-%d')

def get_formatted_date(date_obj):
    return date_obj.strftime('%Y-%m-%d')   

# yesterday_date = get_formatted_date(datetime.now() - timedelta(1)) 
col1, col2, col3, col4, col5 = st.columns(5, gap="small", vertical_alignment="top")   

flattened_list=[]
if len(df) > 500:
    per=22
else:
    outlet_names = df["Outlets Name"].tolist()

    def group_similar_names(names, threshold=65):
        grouped = []
        used_indices = set()  

        for i, name in enumerate(names):
            if i in used_indices:
                continue  # Skip if already grouped

            group = [name]  # Start a new group
            used_indices.add(i)  # Mark this index as used

            for j, other in enumerate(names[i+1:], start=i+1):  # Check remaining names
                if j not in used_indices and fuzz.ratio(name, other) >= threshold:
                    group.append(other)
                    used_indices.add(j)  # Mark this index as used

            grouped.append(group)

        return grouped
    similar_outlet_groups = group_similar_names(outlet_names, threshold=65)
    similar_outlet_groups = [group for group in similar_outlet_groups if len(group) > 1]

    def filter_nearby_outlets(group, df, max_distance=15):
        filtered_group = []
        
        for i, outlet in enumerate(group):  
            lat1, lon1 = df.loc[df.index[df["Outlets Name"] == outlet][0], ["Latitude", "Longitude"]].values
            nearby_outlets = [outlet]  

            for j, other_outlet in enumerate(group):  
                if i == j:  
                    continue
                
                lat2, lon2 = df.loc[df.index[df["Outlets Name"] == other_outlet][0], ["Latitude", "Longitude"]].values
                distance = geodesic((lat1, lon1), (lat2, lon2)).meters  # Distance in meters
                
                if distance <= max_distance:
                    nearby_outlets.append(other_outlet)

            
            if len(nearby_outlets) > 1 and nearby_outlets not in filtered_group:
                filtered_group.append(nearby_outlets)
        
        return filtered_group

    filtered_groups = [filter_nearby_outlets(group, df) for group in similar_outlet_groups]
    filtered_groups = [group for sublist in filtered_groups for group in sublist]

    # Remove duplicate groups
    filtered_groups = [sorted(group) for group in filtered_groups]

    # Remove duplicate groups
    filtered_groups = list(map(list, set(map(tuple, filtered_groups))))
    flattened_list = list(itertools.chain.from_iterable(filtered_groups))

    count1=0
    for i in filtered_groups:
        a=len(i)
        count1 +=a
    per = round((count1 / total_rows1) * 100, 2) if count1 > 0 and total_rows1 > 0 else 0
    
with col1:
    st.metric(label=f"Matched({formatted_date})", value=str(formatted_amount))
with col2:
    st.metric(label=f"Unmatched({formatted_date})", value=str(formatted_amount2))
with col3:
    st.metric(label="Universal Outlets", value=f"{total_rows:,}")
with col4:
    st.metric(label=f"Returned Amount", value=str(return_amount))
with col5:
    st.metric(label=f"Collected Amount", value=str(collection_amount))  

col1, col2, col3, col4,col5 = st.columns(5)
with col1:
    st.metric(label="Beat Outlets", value=f"{total_rows1:,}")
with col2:
    st.metric(label="Pending Outlets", value=f"{pending_outlets_count:,}")
with col3:
    # st.metric(label="TOTAL pending amount / beat", value=str(formatted_amount4))
    st.metric(label="TOTAL Pending(11/01/25)", value=str(formatted_amount5))
with col4:
    st.metric(label=f"Current Pending ({formatted_date})", value=str(formatted_amount3))
with col5:
    st.metric(label="Dupliucate Per", value=f"{per}%")          
counts = (df['repeated_orders'] == "Repeated").sum()

filtered_df = df[df["repeated_orders"] == "Repeated"]

# Count the total number of bills where 'repeated_orders' is "Repeated"
total_bills_count = filtered_df.shape[0]  # Count of rows (bills)
invoice_clear_count = filtered_df["invoice clear"].sum()
st.plotly_chart(fig)

d = dff.groupby("Final_Beats", as_index=False)[["Total Pending(11/01/25)", "Total Pending",]].sum()

selected_beats = st.multiselect(
    "Select Beat Names:",
    options=dff["Final_Beats"].unique(),  
    default=[]  
)
# dff = dff.merge(grouped_dp, on='Final_Beats', how='left')

d.index = range(1, len(d) + 1)  # Set index starting from 1
d.index.name = "Beat No"

d = process_dataframe(d)
d = d.rename(columns={"Total Pending": f"Total Pending({formatted_date})"})
col1, col2 = st.columns([2, 1])
if selected_beats: 
    fi = dff[dff["Final_Beats"].isin(selected_beats)]
    pending_outlets_df = fi[fi["Pending_Status"] == "Pending"]

    # Select only the required columns
    pending_outlets_df = pending_outlets_df[["Final_Beats", "Outlets Name","Total Pending(11/01/25)","Collected Amount" ,"Total Pending","Pending_Percent_diff"]]
    pending_outlets_df = pending_outlets_df.rename(columns={"Total Pending": f"Total Pending({formatted_date})"})
    pending_outlets_df = pending_outlets_df.sort_values(by="Final_Beats")
    pending_outlets_df.index = range(1, len(pending_outlets_df) + 1)  # Set index starting from 1
    pending_outlets_df.index.name = "SI No"
    with col1:
        st.write(pending_outlets_df)
else:
    with col1:
        st.write(d)
with col2:
    st.write(grouped)
st.write(flattened_list)
long_df = pd.melt(grouped, id_vars=['Collection date'], value_vars=['Return', 'Total Collection'], 
                  var_name='Category', value_name='Amount')

# Create the line chart using Plotly Express
fig = px.line(
    long_df, 
    x='Collection date', 
    y='Amount', 
    color='Category', 
    title="Return Amount & Collection Amount over Time", 
    markers=True, 
    color_discrete_map={"Total Collection": "green", "Return": "blue"},  # Assign custom colors
    template="plotly_white",
    hover_data={'Collection date': True, 'Amount': True}, 
    orientation="v"
)

# Updating layout with custom styles
fig.update_layout(
    xaxis_title="Collection Date",
    yaxis_title="Amount",
    font=dict(size=14),
    legend_title="Category",
    paper_bgcolor="#f7f9fc",
    hoverlabel=dict(
        font=dict(
            family="tahoma,geneva",  # Set font style
            size=20,  # Set font size
            color="black"
        ),
        bgcolor="white",  # Set background color
        bordercolor="black",  # Set border color
    )
)

# Display the plot in Streamlit
st.plotly_chart(fig)       

filtered_df = df

filtered_df["Lat_Long_Sum"] = filtered_df["Latitude"] + filtered_df["Longitude"]

# Sort the DataFrame by Lat_Long_Sum
filtered_df = filtered_df.sort_values("Lat_Long_Sum").reset_index(drop=True)
f_df=filtered_df

# st.write(f_df)
# # Check the length of the DataFrame
if len(filtered_df) > 500:
    st.warning("The data contains more than 500 records. Mapping is skipped.")
else:
    # Initialize a single map
    map_center = [filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=14)

    # Group by Beat
    grouped = filtered_df.groupby("Final_Beats")
    
    for beat, df in grouped:
        coordinates = df[["Latitude", "Longitude"]].values

        # Skip if only one record
        if len(df) <= 1:
            st.warning(f"Not enough points to create a route for {beat}.")
            continue

        # Calculate a nearest-neighbor route from the sorted data
        def sorted_route(coords):
            n = len(coords)
            visited = [False] * n
            route = [0]  # Start at the first point
            visited[0] = True

            for _ in range(n - 1):
                last = route[-1]
                distances = cdist([coords[last]], coords)[0]
                distances[visited] = np.inf  # Ignore already visited points
                next_point = np.argmin(distances)
                route.append(next_point)
                visited[next_point] = True

            return route

        # Optimize the route
        route = sorted_route(coordinates)
        optimized_df = df.iloc[route]

        # Add markers and numbers for each point
        for i, (idx, row) in enumerate(optimized_df.iterrows()):
            color = "blue"
            if i == 0:
                color = "red"  # Starting point
            elif i == len(optimized_df) - 1:
                color = "green"  # Ending point
            
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=row["Outlets Name"],
                tooltip=f"{row['Outlets Name']} ({i+1}, {beat})",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)

        # Add a polyline to connect the route
        folium.PolyLine(
            locations=optimized_df[["Latitude", "Longitude"]].values,
            color="blue" if beat == "Beat1" else "purple",  # Different color for each beat
            weight=5,
            opacity=0.7,
            tooltip=f"Route for {beat}"
        ).add_to(m)

    # Display the map in Streamlit
    st_folium(m, width=900, height=600)