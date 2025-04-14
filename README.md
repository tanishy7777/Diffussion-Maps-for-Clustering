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
