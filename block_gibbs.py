"""
Block Gibbs sampling for non-product threshold filters with Qiskit benchmarks.

We sample from pi(sigma) proportional to |q(sigma)| * h(|sigma|) using block Gibbs
updates, then estimate:
    E = Z_h * mean(sign(sigma) * O_sigma)
where Z_h = sum_sigma |q(sigma)| h(|sigma|) = qp_norm_total * E_base[h(|sigma|)].
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from exp_window_pec import (
    exp_window_quasi_prob,
    pec_estimate_qiskit,
    _build_qiskit_circuit,
    _qiskit_noise_instructions,
    _require_qiskit,
    _to_qiskit_statevector,
)
from pec_shared import (
    STANDARD_GATES,
    NoisySimulator,
    error_locations,
    random_circuit,
    random_noise_model,
    random_observable,
    random_product_state,
)


def h_threshold(weight: int, beta: float, w0: int) -> float:
    if weight <= w0:
        return 1.0
    return math.exp(-beta * (weight - w0))


def h_softplus(weight: int, beta: float, w0: float, tau: float = 1.0) -> float:
    x = (weight - w0) / tau
    x = float(np.clip(x, -20.0, 20.0))
    sp = math.log1p(math.exp(x)) if x <= 20.0 else x
    return math.exp(-beta * sp * tau)


def _weight_pmf(p_nonid: Sequence[float]) -> np.ndarray:
    pmf = np.array([1.0])
    for p in p_nonid:
        nxt = np.zeros(pmf.size + 1)
        nxt[:-1] += pmf * (1.0 - p)
        nxt[1:] += pmf * p
        pmf = nxt
    return pmf


def _configs_for_block(k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = 4 ** k
    configs = np.zeros((n, k), dtype=int)
    for j in range(k):
        configs[:, j] = (np.arange(n) // (4 ** j)) % 4
    weights = np.count_nonzero(configs, axis=1)
    return configs, weights


class BlockGibbsSampler:
    def __init__(
        self,
        base_probs: np.ndarray,
        h_target: Callable[[int], float],
        block_size: int,
        rng: np.random.Generator,
    ):
        self.base_probs = base_probs
        self.h_target = h_target
        self.block_size = max(1, int(block_size))
        self.rng = rng
        self.n_locs = base_probs.shape[0]
        self._configs_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _configs(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if k not in self._configs_cache:
            self._configs_cache[k] = _configs_for_block(k)
        return self._configs_cache[k]

    def initialize(self) -> Tuple[np.ndarray, int]:
        sigma = np.empty(self.n_locs, dtype=int)
        for i in range(self.n_locs):
            sigma[i] = self.rng.choice(4, p=self.base_probs[i])
        weight = int(np.count_nonzero(sigma))
        return sigma, weight

    def sweep(self, sigma: np.ndarray, weight: int) -> Tuple[np.ndarray, int]:
        for start in range(0, self.n_locs, self.block_size):
            k = min(self.block_size, self.n_locs - start)
            configs, block_weights = self._configs(k)
            old_block = sigma[start : start + k]
            w_rest = weight - int(np.count_nonzero(old_block))

            base = np.ones(len(configs))
            for j in range(k):
                base *= self.base_probs[start + j][configs[:, j]]
            h_vals = np.array([self.h_target(w_rest + w) for w in block_weights])
            cond = base * h_vals
            total = cond.sum()
            if total <= 0 or not np.isfinite(total):
                cond = np.full(len(cond), 1.0 / len(cond))
            else:
                cond /= total

            idx = self.rng.choice(len(cond), p=cond)
            sigma[start : start + k] = configs[idx]
            weight = w_rest + int(block_weights[idx])
        return sigma, weight

    def samples(
        self, n_samples: int, burn_in: int = 100, thin: int = 1
    ) -> Iterable[Tuple[np.ndarray, int]]:
        sigma, weight = self.initialize()
        for _ in range(max(0, burn_in)):
            sigma, weight = self.sweep(sigma, weight)
        for _ in range(n_samples):
            for _ in range(max(1, thin)):
                sigma, weight = self.sweep(sigma, weight)
            yield sigma.copy(), weight


@dataclass
class GibbsPECEstimate:
    mean: float
    std: float
    qp_norm: float
    z_h: float
    n_samples: int
    weight_mean: float
    weight_std: float


def _local_quasi_data(
    error_locs: List[Tuple],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not error_locs:
        empty = np.zeros((0, 4))
        return empty, empty, np.array([]), empty

    q_list = [exp_window_quasi_prob(p, beta=0.0) for (_, _, p) in error_locs]
    q_arr = np.array(q_list, dtype=float)
    abs_q = np.abs(q_arr)
    qp_norm = abs_q.sum(axis=1)
    base_probs = abs_q / qp_norm[:, None]
    signs = np.sign(q_arr)
    return base_probs, signs, qp_norm, abs_q


def _normalizer_from_base(
    base_probs: np.ndarray,
    qp_norm: np.ndarray,
    h_target: Callable[[int], float],
) -> Tuple[float, float]:
    p_nonid = 1.0 - base_probs[:, 0]
    pmf = _weight_pmf(p_nonid)
    weights = np.arange(len(pmf))
    e_h = float(np.sum(pmf * np.array([h_target(int(w)) for w in weights])))
    qp_norm_total = float(np.prod(qp_norm)) if qp_norm.size else 1.0
    return qp_norm_total * e_h, e_h


def gibbs_pec_estimate_qiskit(
    circuit,
    observable: str,
    initial_state: np.ndarray,
    noise: dict,
    error_locs: List[Tuple],
    h_target: Callable[[int], float],
    n_samples: int,
    block_size: int = 2,
    burn_in: int = 100,
    thin: int = 1,
    seed: int = 0,
) -> GibbsPECEstimate:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if not error_locs:
        QuantumCircuit, UnitaryGate, Pauli, DensityMatrix, Kraus, AerSimulator = _require_qiskit()
        init_state = _to_qiskit_statevector(initial_state, circuit.n_qubits)
        init_density = DensityMatrix(init_state)
        pauli_obs = Pauli(observable[::-1])
        noise_instr = _qiskit_noise_instructions(noise, Kraus)
        backend = AerSimulator(method="density_matrix")

        qc = _build_qiskit_circuit(
            circuit=circuit,
            init_density=init_density,
            noise_instr=noise_instr,
            insertions={},
            QuantumCircuit=QuantumCircuit,
            UnitaryGate=UnitaryGate,
        )
        qc.save_density_matrix()
        result = backend.run(qc).result()
        rho = result.data(0)["density_matrix"]
        dm = DensityMatrix(rho)
        base_val = float(np.real(dm.expectation_value(pauli_obs)))
        z_h = float(h_target(0))
        return GibbsPECEstimate(
            mean=z_h * base_val,
            std=0.0,
            qp_norm=1.0,
            z_h=z_h,
            n_samples=n_samples,
            weight_mean=0.0,
            weight_std=0.0,
        )

    base_probs, signs, qp_norm, _ = _local_quasi_data(error_locs)
    z_h, _ = _normalizer_from_base(base_probs, qp_norm, h_target)

    rng = np.random.default_rng(seed)
    sampler = BlockGibbsSampler(base_probs, h_target, block_size, rng)

    QuantumCircuit, UnitaryGate, Pauli, DensityMatrix, Kraus, AerSimulator = _require_qiskit()
    init_state = _to_qiskit_statevector(initial_state, circuit.n_qubits)
    init_density = DensityMatrix(init_state)
    pauli_obs = Pauli(observable[::-1])
    noise_instr = _qiskit_noise_instructions(noise, Kraus)
    backend = AerSimulator(method="density_matrix")

    sum_vals = 0.0
    sum_sq = 0.0
    w_sum = 0.0
    w_sq = 0.0

    for sigma, weight in sampler.samples(n_samples, burn_in=burn_in, thin=thin):
        insertions = {}
        sign = 1.0
        for idx, (layer, qubit, _) in enumerate(error_locs):
            s = int(sigma[idx])
            if s:
                insertions[(layer, qubit)] = s
            sign *= float(signs[idx][s])

        qc = _build_qiskit_circuit(
            circuit=circuit,
            init_density=init_density,
            noise_instr=noise_instr,
            insertions=insertions,
            QuantumCircuit=QuantumCircuit,
            UnitaryGate=UnitaryGate,
        )
        qc.save_density_matrix()
        result = backend.run(qc).result()
        rho = result.data(0)["density_matrix"]
        dm = DensityMatrix(rho)
        val = sign * float(np.real(dm.expectation_value(pauli_obs)))

        sum_vals += val
        sum_sq += val * val
        w_sum += weight
        w_sq += weight * weight

    mean_raw = sum_vals / n_samples
    if n_samples > 1:
        var_raw = (sum_sq - n_samples * mean_raw * mean_raw) / (n_samples - 1)
        std_raw = math.sqrt(max(0.0, var_raw))
    else:
        std_raw = 0.0

    mean = z_h * mean_raw
    std = z_h * std_raw / math.sqrt(n_samples)

    weight_mean = w_sum / n_samples
    if n_samples > 1:
        weight_var = (w_sq - n_samples * weight_mean * weight_mean) / (n_samples - 1)
        weight_std = math.sqrt(max(0.0, weight_var))
    else:
        weight_std = 0.0

    qp_norm_total = float(np.prod(qp_norm)) if qp_norm.size else 1.0
    return GibbsPECEstimate(
        mean=float(mean),
        std=float(std),
        qp_norm=float(qp_norm_total),
        z_h=float(z_h),
        n_samples=n_samples,
        weight_mean=float(weight_mean),
        weight_std=float(weight_std),
    )


def _summarize(errors: np.ndarray) -> Tuple[float, float, float]:
    bias = float(np.mean(errors))
    std = float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0
    rmse = float(np.sqrt(np.mean(errors**2)))
    return bias, std, rmse


def run_benchmark(
    n_qubits: int,
    depth: int,
    n_samples: int,
    n_trials: int,
    beta: float,
    w0: int,
    block_size: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> None:
    full_errors: List[float] = []
    gibbs_errors: List[float] = []

    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        circuit = random_circuit(n_qubits, depth, rng)
        noise = random_noise_model(rng)
        init = random_product_state(n_qubits, rng)
        obs = random_observable(n_qubits, rng)
        locs = error_locations(circuit, noise)

        sim = NoisySimulator(STANDARD_GATES, noise)
        ideal = sim.ideal(circuit, obs, init)

        full = pec_estimate_qiskit(
            circuit,
            obs,
            init,
            noise,
            locs,
            beta=0.0,
            n_samples=n_samples,
            seed=seed + 1000 * t + 1,
        )
        full_errors.append(full.mean - ideal)

        h_target = lambda w: h_threshold(w, beta, w0)
        gibbs = gibbs_pec_estimate_qiskit(
            circuit,
            obs,
            init,
            noise,
            locs,
            h_target=h_target,
            n_samples=n_samples,
            block_size=block_size,
            burn_in=burn_in,
            thin=thin,
            seed=seed + 1000 * t + 2,
        )
        gibbs_errors.append(gibbs.mean - ideal)

    full_bias, full_std, full_rmse = _summarize(np.array(full_errors))
    gibbs_bias, gibbs_std, gibbs_rmse = _summarize(np.array(gibbs_errors))

    print(f"{n_qubits} qubits, depth {depth}, samples={n_samples}, trials={n_trials}")
    print(f"threshold: w0={w0}, beta={beta}, block={block_size}, burn_in={burn_in}, thin={thin}")
    print(f"{'Method':<22} {'Bias':>10} {'Std':>10} {'RMSE':>10}")
    print("-" * 56)
    print(f"{'Full PEC (beta=0)':<22} {full_bias:>10.5f} {full_std:>10.5f} {full_rmse:>10.5f}")
    print(f"{'Threshold Gibbs':<22} {gibbs_bias:>10.5f} {gibbs_std:>10.5f} {gibbs_rmse:>10.5f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block Gibbs threshold PEC vs full PEC (Qiskit).")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--w0", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_benchmark(
        n_qubits=args.n_qubits,
        depth=args.depth,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        beta=args.beta,
        w0=args.w0,
        block_size=args.block_size,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
