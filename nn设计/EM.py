"""
Functional HMM: EM Algorithm in Recursive Functional Style
=========================================================

This module implements the HMM forward-backward (Baum-Welch) algorithm using
pure recursive functions with memoization. Each function corresponds directly
to a mathematical formula in the EM algorithm.

All formulas are written in Markdown-compatible comments for rich rendering
in editors like VS Code, Jupyter, etc.
"""

from functools import lru_cache
from typing import List, Dict, Tuple, Optional
import numpy as np


class FunctionalHMM:
    """
    Hidden Markov Model implemented with recursive functional style.

    Uses memoized recursion to compute:
    - Forward (α)
    - Backward (β)
    - State posterior (γ)
    - Transition posterior (ξ)
    - EM parameter updates

    All based on:
    - π: initial probabilities
    - A: transition matrix
    - B: emission matrix
    - x: observation sequence
    """

    def __init__(self, π: List[float], A: List[List[float]], B: List[List[float]], x: List[int]):
        """
        Initialize HMM parameters.

        Args:
            π: Initial state probabilities, shape [N]
            A: Transition matrix, shape [N, N]
            B: Emission matrix, shape [N, M]
            x: Observation sequence, length T, values in {0, ..., M-1}
        """
        self.π = π
        self.A = A
        self.B = B
        self.x = x

        self.N = len(π)           # Number of hidden states
        self.M = len(B[0])        # Number of observation symbols
        self.T = len(x)           # Length of observation sequence

    def clear_caches(self):
        """Clear all memoization caches."""
        self.alpha.cache_clear()
        self.beta.cache_clear()
        self.gamma.cache_clear()
        self.xi.cache_clear()

    # -------------------------------------------------------------------------------------
    # 1. Forward Variable α_t(j) = P(x₁..ₜ, zₜ = j)
    # -------------------------------------------------------------------------------------
    """
    **Mathematical Formula:**

    $$
    \\alpha_t(j) = 
    \\begin{cases}
    \\pi_j b_j(x_1) & t = 1 \\\\
    \\left( \\sum_{i=1}^N \\alpha_{t-1}(i) a_{ij} \\right) b_j(x_t) & t > 1
    \\end{cases}
    $$
    """

    @lru_cache(maxsize=None)
    def alpha(self, t: int, j: int) -> float:
        if t == 1:
            return self.π[j] * self.B[j][self.x[0]]
        else:
            sum_term = sum(
                self.alpha(t - 1, i) * self.A[i][j]
                for i in range(self.N)
            )
            return sum_term * self.B[j][self.x[t - 1]]

    # -------------------------------------------------------------------------------------
    # 2. Backward Variable β_t(i) = P(x_{t+1:T} | z_t = i)
    # -------------------------------------------------------------------------------------
    """
    **Mathematical Formula:**

    $$
    \\beta_t(i) = 
    \\begin{cases}
    1 & t = T \\\\
    \\sum_{j=1}^N a_{ij} b_j(x_{t+1}) \\beta_{t+1}(j) & t < T
    \\end{cases}
    $$
    """

    @lru_cache(maxsize=None)
    def beta(self, t: int, i: int) -> float:
        if t == self.T:
            return 1.0
        else:
            return sum(
                self.A[i][j] * self.B[j][self.x[t]] * self.beta(t + 1, j)
                for j in range(self.N)
            )

    # -------------------------------------------------------------------------------------
    # 3. State Posterior γ_t(i) = P(z_t = i | x_{1:T})
    # -------------------------------------------------------------------------------------
    """
    **Mathematical Formula:**

    $$
    \\gamma_t(i) = \\frac{\\alpha_t(i) \\beta_t(i)}{\\sum_{j=1}^N \\alpha_t(j) \\beta_t(j)}
    $$
    """

    @lru_cache(maxsize=None)
    def gamma(self, t: int, i: int) -> float:
        numerator = self.alpha(t, i) * self.beta(t, i)
        denominator = sum(
            self.alpha(t, j) * self.beta(t, j)
            for j in range(self.N)
        )
        return numerator / denominator if denominator != 0 else 0.0

    # -------------------------------------------------------------------------------------
    # 4. Transition Posterior ξ_t(i,j) = P(z_t=i, z_{t+1}=j | x_{1:T})
    # -------------------------------------------------------------------------------------
    """
    **Mathematical Formula:**

    $$
    \\xi_t(i,j) = \\frac{\\alpha_t(i) a_{ij} b_j(x_{t+1}) \\beta_{t+1}(j)}{P(x_{1:T})}
    \\quad \\text{where} \\quad P(x_{1:T}) = \\sum_{k=1}^N \\alpha_T(k)
    $$
    """

    @lru_cache(maxsize=None)
    def xi(self, t: int, i: int, j: int) -> float:
        if t >= self.T:
            raise ValueError("ξ_t is only defined for t < T")
        numerator = (
            self.alpha(t, i) *
            self.A[i][j] *
            self.B[j][self.x[t]] *          # x[t] = x_{t+1}
            self.beta(t + 1, j)
        )
        Z = sum(self.alpha(self.T, k) for k in range(self.N))  # P(x)
        return numerator / Z if Z != 0 else 0.0

    # -------------------------------------------------------------------------------------
    # 5. M-Step: Update Initial Probabilities π_i^{new}
    # -------------------------------------------------------------------------------------
    """
    **Mathematical Formula:**

    $$
    \\pi_i^{\\text{new}} = \\gamma_1(i)
    $$
    """

    def update_pi(self, i: int) -> float:
        return self.gamma(1, i)

    # -------------------------------------------------------------------------------------
    # 6. M-Step: Update Transition Probabilities a_{ij}^{new}
    # -------------------------------------------------------------------------------------
    """
    **Mathematical Formula:**

    $$
    a_{ij}^{\\text{new}} = \\frac{
    \\sum_{t=1}^{T-1} \\xi_t(i,j)
    }{
    \\sum_{t=1}^{T-1} \\gamma_t(i)
    }
    $$
    """

    def update_A(self, i: int, j: int) -> float:
        numerator = sum(self.xi(t, i, j) for t in range(1, self.T))
        denominator = sum(self.gamma(t, i) for t in range(1, self.T))
        return numerator / denominator if denominator != 0 else 0.0

    # -------------------------------------------------------------------------------------
    # 7. M-Step: Update Emission Probabilities b_j(k)^{new}
    # -------------------------------------------------------------------------------------
    """
    **Mathematical Formula:**

    $$
    b_j(k)^{\\text{new}} = \\frac{
    \\sum_{t: x_t = k} \\gamma_t(j)
    }{
    \\sum_{t=1}^T \\gamma_t(j)
    }
    $$
    """

    def update_B(self, j: int, k: int) -> float:
        numerator = sum(
            self.gamma(t, j)
            for t in range(1, self.T + 1)
            if self.x[t - 1] == k
        )
        denominator = sum(self.gamma(t, j) for t in range(1, self.T + 1))
        return numerator / denominator if denominator != 0 else 0.0

    # -------------------------------------------------------------------------------------
    # 8. Full EM Step
    # -------------------------------------------------------------------------------------
    def em_step(self) -> Dict[str, List]:
        """
        Perform one EM iteration: E-step (compute expectations) + M-step (update parameters).

        Returns:
            New parameters: {'π': [...], 'A': [[...]], 'B': [[...]]}
        """
        # E-step: trigger all caches
        for t in range(1, self.T + 1):
            for i in range(self.N):
                self.gamma(t, i)
        for t in range(1, self.T):
            for i in range(self.N):
                for j in range(self.N):
                    self.xi(t, i, j)

        # M-step: update parameters
        π_new = [self.update_pi(i) for i in range(self.N)]
        A_new = [[self.update_A(i, j) for j in range(self.N)] for i in range(self.N)]
        B_new = [[self.update_B(j, k) for k in range(self.M)] for j in range(self.N)]

        return {'π': π_new, 'A': A_new, 'B': B_new}


# -------------------------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: 2 states, 2 observations
    π_init = [0.5, 0.5]
    A_init = [[0.7, 0.3],
              [0.4, 0.6]]
    B_init = [[0.9, 0.1],
              [0.2, 0.8]]
    x_obs = [0, 0, 1, 0, 1]  # observations

    hmm = FunctionalHMM(π_init, A_init, B_init, x_obs)

    # Test single query
    print(f"γ_3(1) = P(z_3=1 | x) = {hmm.gamma(3, 1):.4f}")

    # One EM step
    params_new = hmm.em_step()
    print("After one EM step:")
    print("π =", params_new['π'])
    print("A =", params_new['A'])
    print("B =", params_new['B'])

    # Remember to clear caches if reusing with new params
    hmm.clear_caches()
