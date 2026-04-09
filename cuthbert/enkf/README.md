# Ensemble Kalman Filter

The core atomic functions can be found in [`cuthbertlib.enkf`](../../cuthbertlib/enkf/filtering.py). The high-level filter is in [`cuthbert.enkf.ensemble_kalman_filter`](ensemble_kalman_filter.py).

<!-- --8<-- [start:enkf] -->
The EnKF treats the filtering distribution as **Gaussian**, but represents it with an ensemble of $N$ members $x^{(i)}$ instead of storing a mean and covariance and linearizing $f$ or $h$. The implied mean and covariance are the usual sample mean and sample covariance of the members.

**Predict.** Each member is advanced with the dynamics and process noise. **Multiplicative inflation** (optional) rescales deviations from the new ensemble mean by a factor $(1+\delta)$ to combat underspread ensembles.

**Update.** From deviations in state and observation space, form empirical cross-covariance $C_{xy}$ and innovation covariance $S$ in observation space (including observation noise). The Kalman gain $K \approx C_{xy} S^{-1}$ gives a Kalman-like correction to each member (e.g. stochastic EnKF with random observation perturbations).

See Algorithm 2 in Appendix A in [Calvello, Reich, and Stuart., Ensemble Kalman Methods: A Mean Field Perspective](https://arxiv.org/abs/2209.11371) for the EnKF algorithm which accomodates non-linear observation functions $h$. Note that this algorithm corresponds to the `perturbed_obs = True (Default)` option in the EnKF implementation. This boolean flag is represented by `s` in Algorithm 10.2 of [Sanz-Alonso et al., *Inverse Problems and Data Assimilation*](https://arxiv.org/abs/1810.06191), which was only written for linear $h$.
<!-- --8<-- [end:enkf] -->
