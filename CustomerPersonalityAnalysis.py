import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import warnings
warnings.filterwarnings('ignore')

# Read the file
df = pd.read_csv('D:/Python/ML Unsupervised project/marketing_campaign.csv')
print(df)

# Read the tab-separated CSV file and display the first few rows of the DataFrame
df = pd.read_csv('D:/Python/ML Unsupervised project/marketing_campaign.csv', sep="\t")
df.head()

'''
This specifies that the file uses a tab character (\t) as the 
separator between columns instead of the default comma (,). 
This is common in TSV (Tab-Separated Values) files'''

# Shape of the data
df.shape

# Check for duplicates 
df.duplicated().sum()

# DEscriptive statistics
df.describe()

# Information about the dataset
df.info()

# 2.Data Cleaning


# Check for missing values
df.isnull().sum()

# Remove the unnecessary columns
df.drop(["ID","Z_CostContact","Z_Revenue"],axis=1,inplace=True)

# Visualize the misssing values
plt.figure(figsize=(10,8))
sns.heatmap(df.isnull(),cbar_kws={'label':'Missing_Values'},cmap='Oranges')
plt.title('Missing_values in the Datasets')
plt.show()
plt.savefig('fig1.png')

# Drop the null values
df = df.dropna()

print(f"Missing values in data: {df.isnull().sum().sum()}")

# 3. Explore Data

df['Age'] = 2024 - df['Year_Birth']

plt.figure(figsize=(10, 6))

# Normalize color by the bar height
n, bins, patches = plt.hist(df['Age'], bins=20, color='blue', alpha=0.7, edgecolor='black')
for i in range(len(patches)):
    patches[i].set_facecolor(sns.color_palette("rocket_r", as_cmap=True)(n[i] / max(n)))
    if n[i] > 0:
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, n[i], int(n[i]), 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Age distribution of customers', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
plt.savefig('fig2.png')

plt.figure(figsize=(10, 6))
# Count occurrences of each education level
education_counts = df['Education'].value_counts()
# Normalize color by the bar height
bars = plt.bar(education_counts.index, education_counts.values, color='blue', alpha=0.7, edgecolor='black')
for i, bar in enumerate(bars):
    bar.set_facecolor(sns.color_palette("rocket_r", as_cmap=True)(education_counts.values[i] / max(education_counts.values)))
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()), 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Education Level Distribution of Customers', fontsize=16)
plt.xlabel('Education Level', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
plt.savefig('fig3.png')


# Create a figure for the Marital_Status plot
plt.figure(figsize=(10, 6))

# Marital status distribution
marital_counts = df['Marital_Status'].value_counts()
bars_marital = plt.bar(marital_counts.index, marital_counts.values, color='blue', alpha=0.7, edgecolor='black')

# Apply the same color palette from 'Education' to 'Marital_Status'
for i, bar in enumerate(bars_marital):
    bar.set_facecolor(sns.color_palette("rocket_r", as_cmap=True)(marital_counts.values[i] / max(marital_counts.values)))
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()), 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Marital Status Distribution of Customers', fontsize=16)
plt.xlabel('Marital Status', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
plt.savefig('fig4.png')

# Create a figure for the Income swarm plot
plt.figure(figsize=(10, 6))

# Swarm plot for Income distribution
sns.swarmplot(x=df['Income'], color=sns.color_palette("Oranges", 1)[0])

# Adding title and labels
plt.title('Income Distribution of Customers', fontsize=16)
plt.xlabel('Income', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
plt.savefig('fig5.png')


# Calculate counts and align indices
kidhome_counts = df['Kidhome'].value_counts().sort_index()
teenhome_counts = df['Teenhome'].value_counts().sort_index()
all_indices = sorted(set(kidhome_counts.index).union(set(teenhome_counts.index)))
kidhome_counts = kidhome_counts.reindex(all_indices, fill_value=0)
teenhome_counts = teenhome_counts.reindex(all_indices, fill_value=0)
# Create traces for Kidhome and Teenhome with the specified colors
trace_kidhome = go.Bar(
    x=all_indices,
    y=kidhome_counts.values,
    name='Kidhome',
    marker=dict(color='#FF6500')  # Using color #FA812F for Kidhome
)
trace_teenhome = go.Bar(
    x=all_indices,
    y=teenhome_counts.values,
    name='Teenhome',
    marker=dict(color='#CC2B52')  # Using color #FA4032 for Teenhome
)
# Create the figure
fig = go.Figure(data=[trace_kidhome, trace_teenhome])

# Update layout
fig.update_layout(
    title='Comparison of Kidhome and Teenhome Distributions',
    xaxis_title='Number of Kids/Teens at Home',
    yaxis_title='Frequency',
    barmode='group',
    bargap=0.2,
    bargroupgap=0.1
)

# Show the figure
fig.show()
plt.savefig('fig6.png')

product_columns = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2']
# Sum of each product category
# Sum of each product category
total_spend = df[product_columns].sum()

# Colors and plot size
colors = sns.color_palette("rocket_r", len(product_columns))
plt.figure(figsize=(8, 8))

# Bar plot for spending distribution
sns.barplot(x=product_columns, y=total_spend, palette=colors)

# Labels and title
plt.xlabel('Product Categories', fontsize=14)
plt.ylabel('Total Spending', fontsize=14)
plt.title('Distribution of spending across product categories', fontsize=16)

plt.show()
plt.savefig('fig7.png')

product_columns = ['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
# Sum of each product category
total_spend = df[product_columns].sum()
# Create a DataFrame for Plotly
data = {
    'Product Category': product_columns,
    'Total Spending': total_spend
}
# Plotly Donut Chart with labels outside
fig = px.pie(data, 
             names='Product Category', 
             values='Total Spending', 
             title='Distribution of Purchases and Web Visits across Product Categories',
             color_discrete_sequence=px.colors.sequential.Reds)
# Adjust to make it a donut chart and set labels outside
fig.update_traces(hole=0.4, textinfo='percent', textposition='outside', textfont_size=14)
# Update title
fig.update_layout(title_font_size=16)

fig.show()
plt.savefig('fig8.png')

# Create a contingency table
contingency_table = pd.crosstab(df['Response'], df['Complain'])

plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Oranges', cbar=True)
plt.title('Heatmap of Response vs Complain')
plt.xlabel('Complain')
plt.ylabel('Response')
plt.show()
plt.savefig('fig9.png')

product_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
total_spend = df[product_columns].sum()


colors = sns.color_palette("rocket_r", len(product_columns))
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(total_spend, labels=product_columns, autopct='%1.1f%%', 
                                   startangle=90, colors=colors, wedgeprops=dict(width=0.4), pctdistance=0.75)
for autotext in autotexts:
    autotext.set_color('white')  # white for better contrast
    autotext.set_fontsize(14)    
    autotext.set_weight('bold')  
    # Add title
plt.title('Distribution of spending across product categories', fontsize=16)

plt.show()

plt.savefig('fig10.png')


#   4. Remove Outliers & Feature Engineering

# Calculate the number of days each customer has been with the company 

df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer, format="%d-%m-%Y")
latest_date = df['Dt_Customer'].max()
df['Days_of_client'] = (latest_date - df['Dt_Customer']).dt.days

df.head()

# Calculate the number of days each customer has been with the company 

df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer, format="%d-%m-%Y")
latest_date = df['Dt_Customer'].max()
df['Days_of_client'] = (latest_date - df['Dt_Customer']).dt.days

df.head()

# Replacing 'PhD', '2n Cycle', 'Graduation', and 'Master' with 'PG' in the 'Education' column
df['Education'] = df['Education'].replace(['PhD', '2n Cycle', 'Graduation', 'Master'], 'PG')
# Replacing 'Basic' with 'UG' in the 'Education' column
df['Education'] = df['Education'].replace(['Basic'], 'UG')

# Grouping the 'Married', 'Together' as "relationship
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'relationship')
# Grouping the 'Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd', as Single
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')

# Combining columns together to reduce number of dimensions
df['Kids'] = df['Kidhome'] + df['Teenhome']
df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']

df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']

df['TotalNumPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']

# Select the necessary columns

df = df[['Education', 'Marital_Status', 'Income', 'Kids', 
         'Days_of_client', 'Recency', 'Expenses', 'TotalNumPurchases', 
         'TotalAcceptedCmp', 'Complain', 'Response']]

# Categorize columns into three groups based on their data type

binary_columns = [col for col in df.columns if df[col].nunique() == 2]
categorical_columns = [col for col in df.columns if 2 < df[col].nunique() < 10]
numerical_columns = [col for col in df.select_dtypes(include=['number']).columns 
                     if col not in binary_columns + categorical_columns]


# Calculate the z-scores for each column
z_scores = np.abs(stats.zscore(df[['Income', 'Days_of_client', 'Recency', 'Expenses', 'TotalNumPurchases']]))

# Identify rows where any of the z-scores exceed the threshold
outliers = df[(z_scores > 3).any(axis=1)]

# Remove the outliers from the DataFrame
df = df.drop(outliers.index)

# Print the shape of the original and filtered DataFrames
print("Filtered DataFrame shape:", df.shape[0])


# Set the aesthetic style of the plots
sns.set(style='whitegrid')

# Define the color to be used for all plots
plot_color = '#FA812F'

# Create a figure with subplots
num_cols = len(df.columns)
cols_per_row = 3  # Adjust this for the number of plots per row
num_rows = (num_cols // cols_per_row) + (num_cols % cols_per_row > 0)

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, num_rows * 4))

# Flatten the axes array for easy indexing
axes = axes.flatten()

# Iterate through the columns and create a distplot for each
for i, column in enumerate(df.columns):
    sns.histplot(df[column], kde=True, ax=axes[i], color=plot_color, bins=30)
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')

# Remove empty subplots if the number of columns is not a perfect multiple of cols_per_row
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
plt.savefig('fig11.png')

    
    #      5. DATA PREPROCESSING
# list of categorical columns
cat_df = list(df.select_dtypes(include=["object"]).columns)

# print the list of categorical columns
print(cat_df)

# label encoding for the categorical columns
LE = LabelEncoder()
# encoding the categorical columns
for col in df:
    if df[col].dtype == 'object':
        df[col] = LE.fit_transform(df[col])
        

#Creating a copy of data
ds = df.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['Complain', 'Response']
ds = ds.drop(cols_del, axis=1)
#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
print("All features are now scaled")

#Scaled data to be used for reducing the dimensionality
print("Dataframe to be used for further modelling:")
scaled_ds.head()

#  6. DIMENSIONALITY REDUCTION

#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["Expenses","Income", "TotalNumPurchases"]))
PCA_ds.describe().T


#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["Expenses"]
y =PCA_ds["Income"]
z =PCA_ds["TotalNumPurchases"]
#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()

plt.savefig('fig12.png')

# 7. CLUSTERING

model = KMeans()
visualizer = KElbowVisualizer(model, k=10)

visualizer.fit(PCA_ds)        # Fit the data to the visualizer
visualizer.show() 

plt.savefig('fig13.png')

#Initiating the Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
df["Clusters"]= yhat_AC

#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o' )
ax.set_title("The Plot Of The Clusters")
plt.show()

plt.savefig('fig14.png')

income = PCA_ds["Income"]
expenses = PCA_ds["Expenses"]
clusters = PCA_ds["Clusters"]

# Create a 2D scatter plot using Plotly
fig = go.Figure(data=go.Scatter(
    x=income,
    y=expenses,
    mode='markers',
    marker=dict(
        size=6,
        color=clusters,  # Coloring by cluster
        colorscale='Viridis',  # Adjust colors as needed
        opacity=0.8
    )
))

# Update layout to match the style of your image
fig.update_layout(
    title='Clusters of Customers',
    xaxis_title='Income',
    yaxis_title='Expenses',
    width=800,
    height=600,
    plot_bgcolor='rgba(235, 235, 235, 0.8)',  # Light background color
)

# Show the figure
fig.show()
plt.savefig('fig15.png')

#   8. EVALUATING MODELS

#Plotting countplot of clusters
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
pl = sns.countplot(x=df["Clusters"], palette= pal)
pl.set_title("Distribution Of The Clusters")
plt.show()
plt.savefig('fig16.png')

pl = sns.scatterplot(data = df,x=df["Expenses"], y=df["Income"],hue=df["Clusters"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()
plt.savefig('fig17.png')

plt.figure()
pl=sns.swarmplot(x=df["Clusters"], y=df["Expenses"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=df["Clusters"], y=df["Expenses"])
plt.show()
plt.savefig('fig18.png')

#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=df["TotalAcceptedCmp"],hue=df["Clusters"])
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()
plt.savefig('fig19.png')

#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=df["TotalNumPurchases"],x=df["Clusters"])
pl.set_title("Number of Deals Purchased")
plt.show()
plt.savefig('fig20.png')


# Plotting the correlation matrix
cmap = mcolors.LinearSegmentedColormap.from_list("", ["#F8EDED","#FFD09B", "#FF9100"])
# Create a figure with a size of 16x9 inches
plt.figure(figsize=(16,9))
# Create a heatmap of the correlation matrix
ax = sns.heatmap(df.corr(), annot=True, cmap=cmap)
# Display the plot
plt.show()
plt.savefig('fig21.png')


