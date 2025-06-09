import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample customer data: Age, Annual Income (k$), Spending Score
data = {
    'Age': [25, 34, 22, 45, 52, 23, 40, 60, 48, 33],
    'Annual_Income': [40, 70, 30, 85, 120, 35, 75, 150, 110, 65],
    'Spending_Score': [60, 65, 45, 20, 10, 80, 30, 5, 15, 55]
}
df = pd.DataFrame(data)

# Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Elbow Method to find optimal K
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot elbow graph
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.show()

# Apply KMeans with optimal K (e.g., K=3 based on elbow)
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Show resulting clusters
print(df)
