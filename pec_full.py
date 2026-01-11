"""
Full Probabilistic Error Cancellation (PEC)

Standard PEC: unbiased, exponential overhead, IID sampling.
"""

import numpy as np
from pec_shared import NoisySimulator, Circuit, error_locations, quasi_probs

def full_pec_estimate(simulator, circuit, obs, init, n_samples, seed=None):
    """Full PEC estimator. Returns (estimate, std_error, overhead)."""
    if seed is not None:
        np.random.seed(seed)
    
    locs = error_locations(circuit, simulator.noise)
    if not locs:
        return simulator.ideal(circuit, obs, init), 0.0, 1.0
    
    n_locs = len(locs)
    qs = np.array([quasi_probs(p) for (_, _, p) in locs])
    N1s = np.abs(qs).sum(axis=1)
    N_total = np.prod(N1s)
    pis = np.abs(qs) / N1s[:, None]
    
    all_ss = np.array([np.random.choice(4, p=pi, size=n_samples) for pi in pis])
    selected_qs = qs[np.arange(n_locs)[:, None], all_ss]
    all_signs = np.prod(np.sign(selected_qs), axis=0)
    
    obs_matrix = simulator._build_obs(obs)
    estimates = np.empty(n_samples)
    for i in range(n_samples):
        insertions = {(l, q): all_ss[j, i] for j, (l, q, _) in enumerate(locs)}
        o = simulator.run(circuit, obs_matrix, init, insertions)
        estimates[i] = N_total * all_signs[i] * o
    
    return estimates.mean(), estimates.std() / np.sqrt(n_samples), N_total

if __name__ == "__main__":
    from pec_shared import (STANDARD_GATES, random_circuit, random_noise_model,
                            random_product_state, random_observable)
    rng = np.random.default_rng(42)
    circuit = random_circuit(3, 2, rng)
    noise = random_noise_model(rng)
    init = random_product_state(3, rng)
    obs = random_observable(3, rng)
    sim = NoisySimulator(STANDARD_GATES, noise)
    ideal = sim.ideal(circuit, obs, init)
    est, err, overhead = full_pec_estimate(sim, circuit, obs, init, 2000, seed=42)
    print(f"Full PEC Test: Ideal={ideal:.6f}, Est={est:.6f}+/-{err:.6f}, OH={overhead:.2f}, Bias={est-ideal:.6f}")