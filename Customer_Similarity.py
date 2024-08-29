import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# Initialize the Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Define a collection in Qdrant
collection_name = "customer_vectors"

# Check if the collection exists
if client.collection_exists(collection_name):
    # Optionally delete the existing collection if you want to recreate it
    client.delete_collection(collection_name)

# Create the collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=5, distance=Distance.COSINE),
)

# Load data from CSV
data = pd.read_csv("C:/Users/ADMIN/Documents/Data_Science/Qdrant/customers_segment.csv")

points = []
for index, row in data.iterrows():
    vector = json.loads(row['reduced_vector'].replace('\n', ''))
    
    # Verify the vector structure before storing it
    if not isinstance(vector, list) or not all(isinstance(i, float) for i in vector):
        print(f"Invalid vector format for row {index}: {vector}")
        continue  # Skip this entry if the vector format is incorrect

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
client.upsert(
    collection_name="customer_vectors",
    points=points
)

# Use the first customer's vector for the search as an example
query_vector = json.loads(data.iloc[0]['reduced_vector'].replace('\n', ''))

# Perform the search
search_result = client.search(
    collection_name='customer_vectors',
    query_vector=query_vector,
    limit=20,
    with_vectors=True  # Ensure vectors are included in the results
)

# Rerank the results based on 'view_duration' and 'purchase_amount'
reranked_results = sorted(
    search_result,
    key=lambda x: (x.payload['view_duration'], x.payload['purchase_amount']),
    reverse=True  # Sort descending by view_duration and purchase_amount
)

# Display the reranked results
for result in reranked_results:
    print(f"Customer ID: {result.payload['customer_id']}, Age: {result.payload['age']}, Gender: {result.payload['gender']}, "
          f"Purchase Amount: {result.payload['purchase_amount']}, View Duration: {result.payload['view_duration']}, Location: {result.payload['location']}, "
          f"Similarity Score: {result.score}")


# Create a list of dictionaries with the relevant data
results_data = [
    {
        "Customer ID": result.payload['customer_id'],
        "Age": result.payload['age'],
        "Gender": result.payload['gender'],
        "Purchase Amount": result.payload['purchase_amount'],
        "View Duration": result.payload['view_duration'],
        "Location": result.payload['location'],
        "Similarity Score": result.score,
        "Vector": result.vector 
    }
    for result in reranked_results
]

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results_data)

# Display the DataFrame
print(results_df)


from sklearn.cluster import KMeans

# Extract the vectors from the DataFrame
vectors = np.vstack(results_df['Vector'].values)

# Apply K-Means clustering with 5 clusters
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(vectors)

# Add the cluster labels to the DataFrame
results_df['Cluster'] = cluster_labels

# Store the cluster labels back in Qdrant by using upsert
for index, row in results_df.iterrows():
    point_id = index
    updated_payload = row.drop(['Vector']).to_dict()  # Drop Vector from the payload, keep the rest
    updated_payload['cluster'] = row['Cluster']
    
    client.upsert(
        collection_name="customer_vectors",
        points=[
            PointStruct(
                id=point_id,
                vector=row['Vector'],  # Reuse the existing vector
                payload=updated_payload
            )
        ]
    )

# Display the updated DataFrame with cluster labels
# print(results_df)

# Generate a summary of each cluster
cluster_summary = results_df.groupby('Cluster').agg({
    'Customer ID': 'count',  # Count of customers in each cluster
    'Age': 'mean',  # Summary statistics for age
    'Purchase Amount': 'mean',  # Summary statistics for purchase amount
    'View Duration': 'mean', # Summary statistics for view duration
    'Similarity Score': 'mean' # Summary statistics for similarity score
}).reset_index()


print(cluster_summary)

plt.figure(figsize=(10, 7))

# Plot each cluster with a different color
for cluster_label in sorted(results_df['Cluster'].unique()):
    cluster_data = results_df[results_df['Cluster'] == cluster_label]
    plt.scatter(
        np.vstack(cluster_data['Vector'].values)[:, 0],
        np.vstack(cluster_data['Vector'].values)[:, 1],
        label=f"Cluster {cluster_label}",
        s=100  # Marker size
    )

plt.title("Customer Segments Based on Clustering")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)
plt.show()



