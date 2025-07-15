# Exploring diffusion maps for Timeseries Clustering on the UCI-HAR dataset

Code notebook along with results: https://www.tanishyelgoe.tech/blog/2025/Diffusion-Maps-Clustering/

## Overview
This project demonstrates unsupervised clustering of time series data using diffusion maps, a nonlinear dimensionality reduction technique. The approach is benchmarked on the UCI-HAR (Human Activity Recognition) dataset and compared against PCA, t-SNE, and clustering in the raw feature space. The goal is to show how diffusion maps can reveal meaningful clusters in complex, high-dimensional time series data.

### Visualization of clusters using different dimensionality reduction techniques

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/903c69e0-2bf2-4fbf-bd6d-d609193ad999" />
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/404ae391-999e-46ae-bcab-57609d2ba09d" />
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/8c909aa2-79bc-47ea-9aa9-1e296db2011e" />
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/faa26b74-4279-471c-a94d-2200c7227d64" />


## Introduction
Time series clustering is challenging due to high dimensionality and complex, nonlinear relationships. Diffusion maps address this by constructing a Markov process over the data, capturing both local and global structures, and providing an embedding where similar time series are closer together. This project applies diffusion maps to the UCI-HAR dataset and compares the results to PCA, t-SNE, and raw feature clustering.

## Key Concepts
Diffusion Maps: Nonlinear dimensionality reduction using Markov chains and random walks to uncover the manifold structure of data.

Dynamic Time Warping (DTW): A distance metric suited for time series, capturing similarities even when sequences are out of phase.

Clustering Metrics: Adjusted Rand Index (ARI) for accuracy, Silhouette Score for cluster quality.

## Dataset
Source: [UCI-HAR Dataset]

Description: Multivariate time series from smartphone sensors, labeled with six human activities (walking, sitting, etc.).

Shape: (7352, 9, 128) — 7352 samples, 9 sensor signals, 128 time points per sample.


## Methodology
### Distance Metrics
Euclidean Distance: For basic similarity.

DTW Distance: For robust time series comparison.

### Diffusion Maps
Compute Pairwise Distance Matrix: Using DTW for first 1000 samples.

Construct Similarity Matrix: With a Gaussian kernel, normalized by median sigma.

Build Markov Transition Matrix: Row-normalized similarity matrix.

Eigen Decomposition: Sort eigenvalues/eigenvectors to get diffusion coordinates.

Embedding: Use top 2–3 diffusion coordinates (excluding the trivial first).

### Clustering & Evaluation
Clustering: KMeans (k=6) and DBSCAN on the diffusion embedding.

Metrics: Adjusted Rand Index (ARI), Silhouette Score.

### Comparisons
PCA: Linear dimensionality reduction.
t-SNE: Nonlinear, focuses on local neighborhoods.
Raw Feature Space: Baseline clustering.

## Results
Method	ARI (Clustering Accuracy)	Silhouette Score (Separation)
Diffusion Maps	~0.47	~0.61–0.69
PCA	~0.39	~0.46
t-SNE	~0.26	~0.30
Raw Features	~0.30	~0.13
Diffusion maps outperform other methods in both clustering accuracy and cluster separation.

## Observations
Diffusion maps capture the nonlinear manifold structure of time series data, outperforming PCA and t-SNE in both ARI and silhouette score.

DTW distance is crucial for meaningful similarity between time series.

Visualization: Embeddings in diffusion space reveal clear, well-separated clusters corresponding to human activities.

## References
UCI Machine Learning Repository, Human Activity Recognition Using Smartphones.

Wikipedia: Markov Chain, Diffusion Map.

Coifman, R.R. et al., "Geometric Diffusions as a Tool for Harmonics Analysis and Structure Definition of Data: Diffusion Maps," PNAS, 2005.

NPTEL IIT Guwahati, "Lec 49: Diffusion maps," YouTube, 2022.

## Acknowledgements
Inspired by academic lectures and open-source implementations.

Special thanks to the maintainers of the UCI-HAR dataset.
