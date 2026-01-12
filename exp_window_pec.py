"""
Exponential Window PEC Sampler
==============================

PEC with exponential suppression of high-weight Pauli corrections.

The filter in Fourier domain:
    h[σ] = e^{-β·𝟙[σ≠0]} / (η·p)[σ]

For β = 0, this reduces to full PEC: h[σ] = 1/(η·p)[σ].
For β > 0, non-identity corrections are suppressed by e^{-β}.

The quasi-probability:
    q[s] = (1/4) (η · h)[s]
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from pec_shared import (
    ETA, NoisySimulator, Circuit
)

from rich import box
from rich.console import Console
from rich.table import Table

DIRECT_STATEVECTOR_QUBITS_MAX = 1
QISKIT_METHOD = "statevector"
console = Console()
VERIFY_TEST_CASES = (
    ("Depolarizing 5%", np.array([0.95, 0.05 / 3, 0.05 / 3, 0.05 / 3])),
    ("Depolarizing 10%", np.array([0.90, 0.10 / 3, 0.10 / 3, 0.10 / 3])),
    ("Asymmetric", np.array([0.90, 0.05, 0.03, 0.02])),
    ("Z-dominated", np.array([0.85, 0.02, 0.03, 0.10])),
)


# =============================================================================
# EXPONENTIAL WINDOW FILTER
# =============================================================================

def exp_window_quasi_prob(p: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute local quasi-probability with exponential window.
    
    Args:
        p: Noise probabilities [p_I, p_X, p_Y, p_Z]
        beta: Suppression parameter (β = 0 gives full PEC)
    
    Returns:
        q: Quasi-probability [q_I, q_X, q_Y, q_Z]
    
    The filter h[σ] = e^{-β·𝟙[σ≠0]} / (η·p)[σ] gives:
        q = (1/4) η · h
    """
    eigenvalues = ETA @ p  # (η·p)[σ] for σ ∈ {0,1,2,3}
    
    decay = np.exp(-beta)
    h = np.array([
        1.0 / eigenvalues[0],  # σ=0: no suppression
        decay / eigenvalues[1],  # σ=1: suppressed
        decay / eigenvalues[2],  # σ=2: suppressed
        decay / eigenvalues[3],  # σ=3: suppressed
    ])

    return 0.25 * (ETA @ h)


def qp_norm(q: np.ndarray) -> float:
    """Sampling overhead qp_norm = ||q||₁"""
    return np.abs(q).sum()

# =============================================================================
# PEC ESTIMATOR
# =============================================================================

@dataclass
class PECEstimate:
    """
    Result of PEC estimation.
    
    Attributes:
        mean: Estimated expectation value
        std: Standard error of the mean
        qp_norm: Total sampling overhead Π_v qp_norm_v
        n_samples: Number of samples used
    """
    mean: float
    std: float
    qp_norm: float
    n_samples: int


def pec_estimate(
    sim: NoisySimulator,
    circuit: Circuit,
    observable: str,
    initial_state: np.ndarray,
    error_locs: List[Tuple],
    beta: float,
    n_samples: int,
    seed: int = 0
) -> PECEstimate:
    """
    PEC estimation with exponential window filter.
    
    Args:
        sim: Noisy simulator instance
        circuit: Quantum circuit
        observable: Pauli string (e.g., "XZI")
        initial_state: Initial state vector
        error_locs: List of (layer, qubit, noise_probs) tuples
        beta: Exponential suppression parameter (0 = full PEC)
        n_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
    
    Returns:
        PECEstimate with mean, std, qp_norm, n_samples
    """
    rng = np.random.default_rng(seed)
    
    # Compute local quasi-probabilities at each error location
    local_q = [exp_window_quasi_prob(p, beta) for (_, _, p) in error_locs]
    
    # Total qp_norm = Π_v ||q_v||₁
    local_qp_norm = [qp_norm(q) for q in local_q]
    total_qp_norm = np.prod(local_qp_norm)
    
    # Sampling distributions: π_v[s] = |q_v[s]| / qp_norm_v
    sampling_probs = [np.abs(q) / g for q, g in zip(local_q, local_qp_norm)]
    sampling_signs = [np.sign(q) for q in local_q]
    
    # Monte Carlo sampling
    estimates = np.empty(n_samples)
    
    for i in range(n_samples):
        # Sample Pauli insertions independently at each location
        insertions = {}
        sign = 1.0
        
        for v, (layer, qubit, _) in enumerate(error_locs):
            s = rng.choice(4, p=sampling_probs[v])
            insertions[(layer, qubit)] = s
            sign *= sampling_signs[v][s]
        
        # Run noisy simulation with insertions, weight by sign
        estimates[i] = sign * sim.run(circuit, observable, initial_state, insertions)
    
    # PEC estimate: qp_norm × mean(raw estimates)
    mean = total_qp_norm * estimates.mean()
    std = total_qp_norm * estimates.std() / np.sqrt(n_samples)
    
    return PECEstimate(mean=mean, std=std, qp_norm=float(total_qp_norm), n_samples=n_samples)


# =============================================================================
# QISKIT INTEGRATION
# =============================================================================

def _require_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        from qiskit.quantum_info import Operator, Pauli, Statevector, Kraus
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise ImportError(
            "Qiskit and qiskit-aer are required for Qiskit simulation."
        ) from exc
    return QuantumCircuit, UnitaryGate, Operator, Pauli, Statevector, Kraus, AerSimulator


def _to_qiskit_statevector(state: np.ndarray, n_qubits: int) -> np.ndarray:
    if n_qubits <= DIRECT_STATEVECTOR_QUBITS_MAX:
        return state.copy()
    reshaped = state.reshape([2] * n_qubits)
    return reshaped.transpose(list(reversed(range(n_qubits)))).reshape(-1)


def _qiskit_noise_instructions(noise: dict, Kraus):
    instr_map = {}
    for gate_name, (kraus1, kraus2) in noise.items():
        instr_map[gate_name] = (
            Kraus(list(kraus1)).to_instruction(),
            Kraus(list(kraus2)).to_instruction(),
        )
    return instr_map


def _build_qiskit_circuit(
    circuit: Circuit,
    init_statevector,
    noise_instr: dict,
    insertions: dict,
    QuantumCircuit,
    UnitaryGate,
):
    n = circuit.n_qubits
    qc = QuantumCircuit(n)
    qc.set_statevector(init_statevector)

    for l, layer in enumerate(circuit.layers):
        for gate in layer:
            if isinstance(gate.content, np.ndarray):
                qc.append(UnitaryGate(gate.content), [gate.qubits[0]])
                continue

            name = gate.content
            q0, q1 = gate.qubits
            if name == "CNOT":
                qc.cx(q0, q1)
            elif name == "CNOT_R":
                qc.cx(q1, q0)
            elif name == "CZ":
                qc.cz(q0, q1)
            elif name == "SWAP":
                qc.swap(q0, q1)
            else:
                raise ValueError(f"Unsupported 2q gate: {name}")

            if name in noise_instr:
                instr1, instr2 = noise_instr[name]
                qc.append(instr1, [q0])
                qc.append(instr2, [q1])

                if insertions:
                    for q in (q0, q1):
                        s = insertions.get((l, q))
                        if s is None or s == 0:
                            continue
                        if s == 1:
                            qc.x(q)
                        elif s == 2:
                            qc.y(q)
                        elif s == 3:
                            qc.z(q)
                        else:
                            raise ValueError(f"Bad Pauli index: {s}")

    return qc


def noisy_expectation_qiskit(
    circuit: Circuit,
    observable: str,
    initial_state: np.ndarray,
    noise: dict,
) -> float:
    """
    Noisy expectation value using Qiskit simulation (no PEC insertions).
    """
    QuantumCircuit, UnitaryGate, Operator, Pauli, Statevector, Kraus, AerSimulator = _require_qiskit()

    init_state = _to_qiskit_statevector(initial_state, circuit.n_qubits)
    pauli_obs = Operator(Pauli(observable[::-1]))
    noise_instr = _qiskit_noise_instructions(noise, Kraus)
    backend = AerSimulator(method=QISKIT_METHOD)

    qc = _build_qiskit_circuit(
        circuit=circuit,
        init_statevector=init_state,
        noise_instr=noise_instr,
        insertions={},
        QuantumCircuit=QuantumCircuit,
        UnitaryGate=UnitaryGate,
    )
    qc.save_statevector()

    result = backend.run(qc, shots=1).result()
    state = result.data(0)["statevector"]
    sv = state if isinstance(state, Statevector) else Statevector(state)
    return float(np.real(sv.expectation_value(pauli_obs)))


def pec_estimate_qiskit(
    circuit: Circuit,
    observable: str,
    initial_state: np.ndarray,
    noise: dict,
    error_locs: List[Tuple],
    beta: float,
    n_samples: int,
    seed: int = 0
) -> PECEstimate:
    """
    PEC estimation with exponential window filter using Qiskit simulation.
    """
    QuantumCircuit, UnitaryGate, Operator, Pauli, Statevector, Kraus, AerSimulator = _require_qiskit()

    rng = np.random.default_rng(seed)
    local_q = [exp_window_quasi_prob(p, beta) for (_, _, p) in error_locs]
    local_qp_norm = [qp_norm(q) for q in local_q]
    total_qp_norm = np.prod(local_qp_norm)
    sampling_probs = [np.abs(q) / g for q, g in zip(local_q, local_qp_norm)]
    sampling_signs = [np.sign(q) for q in local_q]

    init_state = _to_qiskit_statevector(initial_state, circuit.n_qubits)
    pauli_obs = Operator(Pauli(observable[::-1]))
    noise_instr = _qiskit_noise_instructions(noise, Kraus)
    backend = AerSimulator(method=QISKIT_METHOD)

    estimates = np.empty(n_samples)
    for i in range(n_samples):
        insertions = {}
        sign = 1.0
        for v, (layer, qubit, _) in enumerate(error_locs):
            s = rng.choice(4, p=sampling_probs[v])
            insertions[(layer, qubit)] = s
            sign *= sampling_signs[v][s]

        qc = _build_qiskit_circuit(
            circuit=circuit,
            init_statevector=init_state,
            noise_instr=noise_instr,
            insertions=insertions,
            QuantumCircuit=QuantumCircuit,
            UnitaryGate=UnitaryGate,
        )
        qc.save_statevector()

        result = backend.run(qc, shots=1).result()
        state = result.data(0)["statevector"]
        sv = state if isinstance(state, Statevector) else Statevector(state)
        estimates[i] = sign * np.real(sv.expectation_value(pauli_obs))

    mean = total_qp_norm * estimates.mean()
    std = total_qp_norm * estimates.std() / np.sqrt(n_samples)
    return PECEstimate(mean=mean, std=std, qp_norm=float(total_qp_norm), n_samples=n_samples)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_beta_zero_is_full_pec():
    """Verify that β=0 recovers full PEC quasi-probabilities."""
    console.rule("Verification: β=0 equals Full PEC")
    table = Table(box=box.ASCII)
    table.add_column("Case")
    table.add_column("max|diff|", justify="right")
    table.add_column("Status", justify="center")

    for name, p in VERIFY_TEST_CASES:
        # Full PEC: q = (1/4) η · (1/(η·p))
        eigenvalues = ETA @ p
        q_full = 0.25 * (ETA @ (1.0 / eigenvalues))
        
        # Exponential window with β=0
        q_exp0 = exp_window_quasi_prob(p, beta=0.0)
        
        # Check equality
        diff = np.abs(q_full - q_exp0).max()
        status = "OK" if diff < 1e-14 else "FAIL"
        table.add_row(name, f"{diff:.2e}", status)

    console.print(table)

if __name__ == "__main__":
    verify_beta_zero_is_full_pec()
