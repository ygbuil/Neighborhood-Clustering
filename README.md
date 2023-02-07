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

## Results
The final clusters can be observed in the following plots. From the pairplots it can be observed that the data has been reasonably segmented for each variable, and both the train and test set are clustered equally, which indicates that the model is properly trained with no under/overfitting.

![alt_text](https://github.com/ygbuil/Neighborhood-Clustering/blob/master/images/train_clusters.png)
![alt_text](https://github.com/ygbuil/Neighborhood-Clustering/blob/master/images/test_clusters.png)

Finally, from the parallel lines plot of the cluster centers, the central tendency of each cluster can be observed, and some interesting conclusions can be extrated, such as that the high populated neighborhoods are populated by people of younger age and the salaries are higher in those regions.

![alt_text](https://github.com/ygbuil/Neighborhood-Clustering/blob/master/images/center_clusters.png)