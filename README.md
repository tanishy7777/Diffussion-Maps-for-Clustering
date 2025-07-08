## Exploring diffusion maps for Timeseries Clustering on the UCI-HAR dataset

You can view this on https://www.tanishyelgoe.tech/blog/2025/Diffusion-Maps-Clustering/

Diffusion maps are a nonlinear dimensionality reduction technique that models a dataset as a Markov chain, where the probability of moving from one point to another is based on their similarity. This is done by simulating random walks on the dataset, (called the diffusion process).

Glossary:

**Markov Chain**: If you are familiar with Finite State automata this is very similar to that. It is a system that moves between different states step by step, where the next step depends only on the current state (not past history).
![image](https://github.com/user-attachments/assets/7d85e926-3e00-41ce-b4d5-23ce53b10bf1)

Random Walk: It’s a type of Markov Chain where we randomly move from one place to another based on probabilities.

For diffusion maps, the probabilities are based on the distances, closer points have higher transition probability.

Consider n datapoints, we can create a transition matrix P where `Pij` denotes the probability of sum of all paths going from point to

in the dataset X.

This transition probability can be given by the Diffusion kernel, where d(xi, xj) is the DTW distance in case of time series data.
![image](https://github.com/user-attachments/assets/3d2011ce-c629-4af4-86a0-ee8979af62e9)


Using the Perron-Frobenius Theorem: Which states that Any matrix M that is positive (aij > 0), column stochastic matrix (

= 1 ∀j)

Hence, We can say that the matrix P gives 1 as the largest eigen value and all other eigen values are <1.

Another thing is if we take $$P^t$$ then the elements $P_{ij}$ represent the probability of all paths of length t going from $x_i$ to $x_j$


# Timeseries Clustering Using Diffusion Maps
## Overview
This project demonstrates unsupervised clustering of time series data using diffusion maps, a nonlinear dimensionality reduction technique. The approach is benchmarked on the UCI-HAR (Human Activity Recognition) dataset and compared against PCA, t-SNE, and clustering in the raw feature space. The goal is to show how diffusion maps can reveal meaningful clusters in complex, high-dimensional time series data.


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

## How to Run
Clone the repository:

```bash
git clone https://github.com/yourusername/diffusion-maps-timeseries.git
cd diffusion-maps-timeseries
```
Install dependencies:

```bash
pip install -r requirements.txt
Download the UCI-HAR dataset and place it in the data/ directory as described above.
```
Run the main script:

```bash
python diffusion_maps_clustering.py
```
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
