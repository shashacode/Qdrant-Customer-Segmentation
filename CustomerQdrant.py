import pandas as pd
import ast
import json
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
merged_df = pd.read_csv("C:/Users/ADMIN/Documents/Data_Science/Qdrant/customers.csv")

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

# Function to search for similar customers
def search_similar_customers(vector, top_n=5):
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_n,
        with_payload=True
    )
    return results

# Example of searching for similar customers using a specific vector
example_vector = json.loads(merged_df.iloc[0]['reduced_vector'].replace('\n', ''))
similar_customers = search_similar_customers(vector=example_vector, top_n=5)

# Print out the results
for customer in similar_customers:
    print(f"Customer ID: {customer.payload['customer_id']}, Similarity Score: {customer.score}")
    print(f"Other Details: Age: {customer.payload['age']}, Gender: {customer.payload['gender']}")
    print("-" * 30)