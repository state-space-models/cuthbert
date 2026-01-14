# Discrete Hidden Markov Models

<!--discrete-hidden-markov-models-start-->
`cuthbert` provides exact filtering and smoothing for state-space models with discrete states (aka hidden Markov models):

In this case, we assume the latent states $x_t$ are discrete, i.e. $x_t \in \{1, 2, \ldots, K\}$ and therefore distributions over a state can be stored as an array of normalized probabilities.

Observations are handled via a general log likelihood functions that can also be
stored as an array of log likelihoods $b_i = \log p(y_t \mid x_t = i)$ of size $K$.

<!--discrete-hidden-markov-models-end-->

The core atomic functions can be found in [`cuthbertlib.discrete`](../../cuthbertlib/discrete).

