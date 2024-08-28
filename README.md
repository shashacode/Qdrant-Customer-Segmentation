This is a project that shows how to carry out Customer Segmentation using Qdrant. 
The dataset used is for a fictitious company consisting of Customer's details, Purchases, Reviews and Internet sessions. The dataset will be segmented using Qdrant which is useful for different business applications including Market targeting, Revenue Optimization and other business use cases.  

The Segmentation_Preprocess.ipynb file covers preprocessing of the dataset. This is to prepare the dataset to be used in Qdrant. The dataset is normalized to vector format.  
The CustomerQdrant.py file covers the Customer Similarity Search using Qdrant. The data used is the customers.csv file which comprises the customer details and the combined vectors.
The clustering_customer.py file covers the Clustering Analysis using Qdrant. The data used is the customer_vect.csv file which consists of the normalized features.
