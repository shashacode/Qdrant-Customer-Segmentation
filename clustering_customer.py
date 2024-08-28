import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance


# Initialize the Qdrant client
qdrant = QdrantClient(host="localhost", port=6333)

# Define a collection in Qdrant
collection_name = "customer_vectors"

# Check if the collection exists
if qdrant.collection_exists(collection_name):
    # Optionally delete the existing collection if you want to recreate it
    qdrant.delete_collection(collection_name)

# Create the collection
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=5, distance=Distance.COSINE),
)

# qdrant.recreate_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=50, distance=Distance.COSINE),
# )
merged_df = pd.read_csv("C:/Users/ADMIN/Documents/Data_Science/Qdrant/customers_vect.csv")

points = []
for index, row in merged_df.iterrows():
    vector = json.loads(row['reduced_vector'].replace('\n', ''))
    point = PointStruct(
        id=index,
        vector=vector,
        payload={
            "customer_id": row['customer_id'],
            "age": row['age'],
            "gender": row['gender'],
            "location": row['location'],
            "signup_date": row['signup_date'],
            "page_viewed": row['page_viewed'],
            "view_duration": row['view_duration'],
            "purchase_amount": row['purchase_amount']
        }
    )
    points.append(point)

# Insert the points into the collection
qdrant.upsert(
    collection_name="customer_vectors",
    points=points
)



# Extract the vector data from the dataframe
# Assuming the vectors are stored as JSON strings
vectors = merged_df['reduced_vector'].apply(lambda x: np.array(json.loads(x)))

# Stack the vectors into a numpy array suitable for clustering
X = np.stack(vectors)

# Apply K-means clustering
# Define the number of clusters (k). You can experiment with different values.
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
merged_df['cluster'] = kmeans.fit_predict(X)

# Analyze the clusters by grouping or summarizing the data
# For example, you can see the average purchase amount per cluster
# cluster_summary = merged_df.groupby('cluster').agg({
#     'purchase_amount': 'mean',
#     'age': 'mean',
#     'view_duration': 'mean',
#     'page_viewed':'mean'
#     # Add more features to summarize as needed
# })

# print(cluster_summary)

# Example summary statistics by cluster
cluster_summary = merged_df.groupby('cluster').agg({
    'age': 'mean',
    'gender': 'mean',  # Assuming gender is binary encoded (0/1)
    'purchase_amount': 'mean',
    'page_viewed': 'mean',
    'view_duration': 'mean',
    # Add other features as needed
})

# Print the summary for interpretation
print(cluster_summary)


# Save the dataframe with the cluster labels
# merged_df.to_csv("path_to_save_clustered_dataframe.csv", index=False)


# # Assuming 2D vectors for simplicity
# plt.scatter(X[:, 0], X[:, 1], c=merged_df['cluster'], cmap='viridis')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.title('K-means Clustering of Customers')
# plt.show()


# import matplotlib.pyplot as plt

# Example: Plot average purchase amount by cluster
# sns.barplot(x='cluster', y='purchase_amount', data=merged_df, estimator=np.mean)
# plt.title('Average Purchase Amount by Cluster')
# plt.show()

# Example: Age distribution by cluster
sns.boxplot(x='cluster', y='age', data=merged_df)
plt.title('Age Distribution by Cluster')
plt.show()
