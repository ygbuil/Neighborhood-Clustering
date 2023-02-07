# Neighborhood-Clustering

## Introduction
This project consists of a clustering of all neighborhoods in Spain based on the following demographic information:
* Neighborhood population.
* Mean age.
* Percentage of foreign residents.
* Average salary.
Disclaimer: Postcodes have been anonimized with an incremental integer.

## Algorithm
The algorithm used is a K-means clustering algorithm. This algorithm requires to determine the most optimal value of K (number of clusters) so that the clusters are distinct enough from each other. Using a combination of both the Elbow Method and Silhouette Coefficient, the most optimal number of clusters found was 4.

![alt_text](https://github.com/ygbuil/Neighborhood-Clustering/blob/master/images/elbow_method.png)   ![alt_text](https://github.com/ygbuil/Neighborhood-Clustering/blob/master/images/silhouette_coefficients.png)