# Experiments in *Private Summation in the Multi-Message Shuffle Model* (CCS 2020).

## Synthetic Experiments

We considered both a dataset of uniformly random samples from [0,1] (which we denote "ur") and a sample from a normal distribution with mean 0.573
and standard deviation .1, which we refer to as "normal". 
A set of plots provides a comparison between the different protocols in the shuffle model, while the other set of plots compares the best protocol in the shuffle model with standard protocols in the central and local model.
The same trends are observed for both types of synthetic data.
As expected, the obtained accuracies are clustered according to their analytical errors,
which in turn correspond to the model they operate in. It is worth mentioning that IKOS
gives better error than CentralLaplace in these datasets. We speculate that this is due to the fact that the Geometric mechanism
works well with counts, as it is discrete.

## Real-world data Experiments

As a real-word dataset we used the Adult dataset. This dataset contains n = 32561 curated records from the 1994 US Census database.
This is a dataset commonly used evaluate classification algorithms. We focused on the task of differentially private
estimation the mean of the "age" attribute normalized by the maximum age (with parameters $\epsilon = 1, \delta = 1/n^2$). The generated table shows the mean and standard deviation of the error obtained by each algorithm over 20 executions.
As expected, IKOS incurs error close to the one of CentralLaplace, and LocalLaplace has significantly worse error than the rest.
Our recursive protocols outperform the single-message protocol Balle et al. (CRYPTO2019) in this task, and
numerically optimized variants outperform their analytical counterparts.
