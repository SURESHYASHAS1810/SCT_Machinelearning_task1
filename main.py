import pandas as pd

# Load the dataset
# Replace 'your_dataset.csv' with your file path
df = pd.read_csv('/content/Mall_Customers.csv')

# Examine the dataset
print(df.head())  # Preview the first few rows
print(df.info())  # Get data types and missing values
# Identify numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Fill missing values in numeric columns with their mean
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
# Fill missing values in non-numeric columns with 'Unknown' or mode
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
for column in non_numeric_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
  # Handle numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Handle non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
for column in non_numeric_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Verify no missing values remain
print(df.isnull().sum())
# Check the data types of all columns
print(df.dtypes)

# Identify non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_columns)

for column in non_numeric_columns:
    print(f"Unique values in {column}:", df[column].unique())

from sklearn.preprocessing import LabelEncoder

for column in non_numeric_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
  df = df.drop(non_numeric_columns, axis=1)

print(df.columns)

# Strip leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

if 'Gender' in df.columns:
    print(df['Gender'].unique())
elif 'gender' in df.columns:
    print(df['gender'].unique())

df.rename(columns=lambda x: x.strip(), inplace=True)

for column in non_numeric_columns:
    if column in df.columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

  print("Remaining missing values:")
print(df.isnull().sum())
# Check for missing values
print(df.isnull().sum())

# Fill missing values (example: fill with the mean)
df.fillna(df.mean(), inplace=True)

df.drop_duplicates(inplace=True)
# Drop unnecessary columns
df = df.drop(['CustomerID'], axis=1)

# Preprocess the Data

from sklearn.preprocessing import StandardScaler

# Normalize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Optional: Convert scaled data back to a DataFrame for easier visualization
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print(scaled_df.head())

# Determine the Optimal Number of Clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ssd = []
for k in range(1, 11):  # Test cluster sizes from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    ssd.append(kmeans.inertia_)

# Plot SSD to find the "elbow point"
plt.plot(range(1, 11), ssd, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Distances (SSD)")
plt.title("Elbow Method for Optimal k")
plt.show()

#Apply K-Means Clustering
# Set optimal number of clusters based on Elbow Method
optimal_clusters = 3  # Replace with the value you determined

# Apply K-Means
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# View dataset with clusters
print(df.head())

# Analyze the Clusters

# Analyze clusters
cluster_analysis = df.groupby('Cluster').mean()
print(cluster_analysis)


#Visualize the Clusters

from sklearn.decomposition import PCA
import seaborn as sns

# Reduce dimensions for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Add PCA components and cluster labels to DataFrame
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

# Plot clusters
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis')
plt.title("Customer Clusters")
plt.show()

# Save to a new CSV file
df.to_csv('clustered_customers.csv', index=False)
