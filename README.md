This project is documented on my Medium page: https://medium.com/@floraoladipupo/customer-segmentation-in-retail-using-vector-search-qdrant-ec4c5740fed9

This is a project that shows how to carry out Customer Segmentation using Qdrant. 
The dataset used is for a fictitious company consisting of Customer's details, Purchases, Reviews and Internet sessions CSV files. The aim is to segment customers' data using Qdrant which is useful for different business applications including Market targeting, Revenue Optimization and other business use cases.  

The Segmentation_Pre.ipynb file covers the preprocessing of the dataset. This is to prepare the dataset to be used in Qdrant. The dataset is normalized to vector format.    

The Customer_Similarity.py file covers carrying out Customer Similarity Search using Qdrant query and payload reranking. The result of this search was used to carry out customer segmentation via clustering. The data used is the customers_segment.csv file which comprises the customer details and the embedded vectors.  


