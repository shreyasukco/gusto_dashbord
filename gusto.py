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
st.markdown(
    "<h4 style='text-align: center;'>Beat Performance & Collection Analytics</h4>",
    unsafe_allow_html=True
)    
# st.write(dp.columns.tolist())
dp.columns = dp.columns.str.strip()

dp.drop(dp[dp["Particulars"].isin(["(cancelled)", "(cancelled )"])].index, inplace=True)

dp.drop(["Diff","Total Collection"], axis=1, inplace=True)

dp[["Old Collection", "Recent", "Hemant", "My", 
    "Return", "Bad Debts", "Canceled Bills"]] = (
    dp[["Old Collection", "Recent", "Hemant", "My", 
        "Return", "Bad Debts", "Canceled Bills"]]
    .replace({None: 0, "": 0})  # Replace None and empty strings with 0
    .apply(pd.to_numeric, errors="coerce")  # Convert to numeric, replacing invalid values with NaN
    .fillna(0)  # Fill NaN values with 0
)
dp["Total Collection"] = dp[["Old Collection", "Recent", "Hemant", "My", 
                             "Return", "Bad Debts", "Canceled Bills"]].sum(axis=1)
    
dp["Diff"] = pd.to_numeric(dp["Gross Total"], errors="coerce").fillna(0) - pd.to_numeric(dp["Total Collection"], errors="coerce").fillna(0)

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

grouped["Total Collection"] -= (grouped["Bad Debts"] + grouped["Return"] + grouped["Canceled Bills"])

grouped = grouped.sort_values(by='Collection date', ascending=False)

# Reset index and start from 1 instead of 0
grouped.reset_index(drop=True, inplace=True)
grouped.index += 1
grouped.index.name = "SI No"
# start_date_sidebar = st.sidebar.date_input("Start Date", grouped['Collection date'].min())
# end_date_sidebar = st.sidebar.date_input("End Date", grouped['Collection date'].max())
mmm=grouped
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
    "Select a Month-Year Range to View the Pending Amount",
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

filtdf = grouped

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
    lambda x: "Pending" if pd.notna(x) and x > 0 else "No Dues"
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
    "No Dues": "green",          
}

fig = px.scatter_mapbox(
    df, 
    lat="LAT", 
    lon="LONG", 
    hover_name="Final_Beats",
    hover_data=["Outlets Name", "Zone", "Region", "Territory", "Outlets Type"],
    zoom=10,  
    center={"lat": center_lat, "lon": center_lon},  
    height=600,  
    color="Pending_Status",
    color_discrete_map=custom_color_map,    
)

fig.update_traces(marker=dict(size=12, opacity=0.8))

# Set the map style (open street map is interactive with drag/zoom)
fig.update_layout(mapbox_style="open-street-map")

# Add background color (gray)
fig.update_layout(
    mapbox=dict(
        center={"lat": center_lat, "lon": center_lon},
        zoom=10,
        accesstoken="your_mapbox_access_token"  # Optional for Mapbox styles
    ),
    margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Remove margins for full view
    paper_bgcolor="#D3D3D3",  # Light gray background
    plot_bgcolor="#D3D3D3" # Also set plot background color to gray
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
 

# flattened_list=[]
# if len(df) > 500:
#     per=22
# else:
#     outlet_names = df["Outlets Name"].tolist()

#     def group_similar_names(names, threshold=65):
#         grouped = []
#         used_indices = set()  

#         for i, name in enumerate(names):
#             if i in used_indices:
#                 continue  # Skip if already grouped

#             group = [name]  # Start a new group
#             used_indices.add(i)  # Mark this index as used

#             for j, other in enumerate(names[i+1:], start=i+1):  # Check remaining names
#                 if j not in used_indices and fuzz.ratio(name, other) >= threshold:
#                     group.append(other)
#                     used_indices.add(j)  # Mark this index as used

#             grouped.append(group)

#         return grouped
#     similar_outlet_groups = group_similar_names(outlet_names, threshold=65)
#     similar_outlet_groups = [group for group in similar_outlet_groups if len(group) > 1]

#     def filter_nearby_outlets(group, df, max_distance=15):
#         filtered_group = []
        
#         for i, outlet in enumerate(group):  
#             lat1, lon1 = df.loc[df.index[df["Outlets Name"] == outlet][0], ["Latitude", "Longitude"]].values
#             nearby_outlets = [outlet]  

#             for j, other_outlet in enumerate(group):  
#                 if i == j:  
#                     continue
                
#                 lat2, lon2 = df.loc[df.index[df["Outlets Name"] == other_outlet][0], ["Latitude", "Longitude"]].values
#                 distance = geodesic((lat1, lon1), (lat2, lon2)).meters  # Distance in meters
                
#                 if distance <= max_distance:
#                     nearby_outlets.append(other_outlet)

            
#             if len(nearby_outlets) > 1 and nearby_outlets not in filtered_group:
#                 filtered_group.append(nearby_outlets)
        
#         return filtered_group

#     filtered_groups = [filter_nearby_outlets(group, df) for group in similar_outlet_groups]
#     filtered_groups = [group for sublist in filtered_groups for group in sublist]

#     # Remove duplicate groups
#     filtered_groups = [sorted(group) for group in filtered_groups]

#     # Remove duplicate groups
#     filtered_groups = list(map(list, set(map(tuple, filtered_groups))))
#     flattened_list = list(itertools.chain.from_iterable(filtered_groups))

#     count1=0
#     for i in filtered_groups:
#         a=len(i)
#         count1 +=a
#     per = round((count1 / total_rows1) * 100, 2) if count1 > 0 and total_rows1 > 0 else 0
flattened_list = []

if len(df) > 500:
    per = 13
else:
    outlet_details = df[["Outlets Name", "Outlet Erp Id", "Final_Beats"]].values.tolist()  

    def group_similar_names(details, threshold=65):
        grouped = []
        used_indices = set()

        for i, (name, erp_id, beat) in enumerate(details):
            if i in used_indices:
                continue  

            group = [(name, erp_id, beat)]  
            used_indices.add(i)  

            for j, (other_name, other_erp_id, other_beat) in enumerate(details[i+1:], start=i+1):
                if j not in used_indices and fuzz.ratio(name, other_name) >= threshold:
                    group.append((other_name, other_erp_id, other_beat))
                    used_indices.add(j)  

            grouped.append(group)

        return grouped

    similar_outlet_groups = group_similar_names(outlet_details, threshold=65)
    similar_outlet_groups = [group for group in similar_outlet_groups if len(group) > 1]

    def filter_nearby_outlets(group, df, max_distance=15):
        filtered_group = []
        
        for i, (outlet, erp_id, beat) in enumerate(group):
            lat1, lon1 = df.loc[df.index[df["Outlets Name"] == outlet][0], ["Latitude", "Longitude"]].values
            nearby_outlets = [(outlet, erp_id, beat)]

            for j, (other_outlet, other_erp_id, other_beat) in enumerate(group):
                if i == j:
                    continue

                lat2, lon2 = df.loc[df.index[df["Outlets Name"] == other_outlet][0], ["Latitude", "Longitude"]].values
                distance = geodesic((lat1, lon1), (lat2, lon2)).meters  

                if distance <= max_distance:
                    nearby_outlets.append((other_outlet, other_erp_id, other_beat))

            if len(nearby_outlets) > 1 and nearby_outlets not in filtered_group:
                filtered_group.append(nearby_outlets)
        
        return filtered_group

    filtered_groups = [filter_nearby_outlets(group, df) for group in similar_outlet_groups]
    filtered_groups = [group for sublist in filtered_groups for group in sublist]

    # Corrected sorting and duplicate removal
    filtered_groups = [sorted(g, key=lambda x: x[1]) for g in filtered_groups]  
    filtered_groups = list(map(list, {tuple(g) for g in filtered_groups}))

    # Flatten the list
    flattened_list = list(itertools.chain.from_iterable(filtered_groups))

    # Calculate percentage
    total_rows1 = len(df)  # Ensure this is defined
    count1 = sum(len(i) for i in filtered_groups)
    per = round((count1 / total_rows1) * 100, 2) if count1 > 0 and total_rows1 > 0 else 0
df_filtered = pd.DataFrame(flattened_list, columns=["Outlets Name", "Outlet Erp Id", "Final_Beats"])
df_filtered.reset_index(drop=True, inplace=True)
df_filtered.index += 1
df_filtered.index.name = "SI No"

import streamlit as st

# Custom CSS for styling KPIs with fixed height
st.markdown("""
    <style>
    .metric-box {
        background-color:#000080;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        height: 120px; /* Fixed height for uniformity */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 16px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# First row of KPIs
col1, col2, col3, col4, col5 = st.columns(5, gap="small")

with col1:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Matched Amount ({formatted_date})</div>
            <div class="metric-value">{formatted_amount}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Unmatched Amount ({formatted_date})</div>
            <div class="metric-value">{formatted_amount2}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Universal Outlets</div>
            <div class="metric-value">{total_rows:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Pending Amount on (11/01/25)</div>
            <div class="metric-value">{formatted_amount5}</div>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Current Pending ({formatted_date})</div>
            <div class="metric-value">{formatted_amount3}</div>
        </div>
    """, unsafe_allow_html=True)

# Second row of KPIs
st.write("")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Beat Outlets</div>
            <div class="metric-value">{total_rows1:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Pending Outlets</div>
            <div class="metric-value">{pending_outlets_count:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Returned Stock Value</div>
            <div class="metric-value">{return_amount}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Collection</div>
            <div class="metric-value">{collection_amount}</div>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Duplicate Percentage</div>
            <div class="metric-value">{per}%</div>
        </div>
    """, unsafe_allow_html=True)
st.write("")
counts = (df['repeated_orders'] == "Repeated").sum()

filtered_df = df[df["repeated_orders"] == "Repeated"]

# Count the total number of bills where 'repeated_orders' is "Repeated"
total_outlets = filtered_df.shape[0]  # Count of rows (bills)
invoice_clear_count = filtered_df["invoice clear"].sum()
# st.write(total_outlets)
st.plotly_chart(fig)

d = dff.groupby("Final_Beats", as_index=False)[["Total Pending(11/01/25)", "Total Pending",]].sum()

long_df = pd.melt(grouped, id_vars=['Collection date'], value_vars=['Return', 'Total Collection'], 
                  var_name='Category', value_name='Amount')
# Create the line chart using Plotly Express
fig = px.line(
    long_df, 
    x='Collection date', 
    y='Amount', 
    color='Category', 
    title="Collection Amount & Return Amount over Time", 
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
    st_folium(m, width=1200, height=600)
    
col1, col2 = st.columns([2, 2])

with col2:    
    st.markdown("Duplicate Outlets list in the Selected Beats")
    st.dataframe(df_filtered, use_container_width=True)  # Expands table fully

with col1:
    st.markdown("Date-wise Collection and Return Data")
    st.dataframe(mmm, use_container_width=True)
  # Use st.dataframe() for consistency
    

selected_beats = st.multiselect(
    "Select Beat Names here to Check Pending Amount Outlets:",
    options=dff["Final_Beats"].unique(),  
    default=[]  
)
# dff = dff.merge(grouped_dp, on='Final_Beats', how='left')

d.index = range(1, len(d) + 1)  # Set index starting from 1
d.index.name = "Beat No"

d = process_dataframe(d)
d = d.rename(columns={"Total Pending": f"Total Pending({formatted_date})"})

if selected_beats: 
    fi = dff[dff["Final_Beats"].isin(selected_beats)]
    pending_outlets_df = fi[fi["Pending_Status"] == "Pending"]
    # Select only the required columns
    pending_outlets_df = pending_outlets_df[["Final_Beats", "Outlets Name","Total Pending(11/01/25)","Collected Amount" ,"Total Pending","Pending_Percent_diff"]]
    pending_outlets_df = pending_outlets_df.rename(columns={"Total Pending": f"Total Pending({formatted_date})"})
    pending_outlets_df = pending_outlets_df.sort_values(by="Final_Beats")
    pending_outlets_df.index = range(1, len(pending_outlets_df) + 1)  # Set index starting from 1
    pending_outlets_df.index.name = "SI No"
    
    st.dataframe(pending_outlets_df, use_container_width=True)
else:
    st.dataframe(d, use_container_width=True)
  