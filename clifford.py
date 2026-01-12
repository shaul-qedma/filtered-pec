# clifford.py
"""
Clifford circuit utilities for lightcone computation and stabilizer simulation.

Uses Stim for Pauli propagation and stabilizer expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

from pec_shared import Circuit, Gate, CLIFFORD_2Q_GATES, error_locations


def _require_stim():
    try:
        import stim  # type: ignore
    except ImportError as exc:
        raise ImportError("stim required for Clifford utilities. Install with: pip install stim") from exc
    return stim


def _matrix_key(mat: np.ndarray) -> Tuple[float, ...]:
    """Canonicalize a 1Q Clifford matrix up to global phase for hashing."""
    flat = mat.flatten()
    phase = 1.0 + 0.0j
    for entry in flat:
        if abs(entry) > 1e-8:
            phase = entry / abs(entry)
            break
    canonical = mat * np.conj(phase)
    return tuple(np.round(canonical, 8).flatten())


def _build_clifford_catalog() -> Tuple[List[np.ndarray], List[List[str]]]:
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    s = np.array([[1, 0], [0, 1j]], dtype=complex)
    generators = [("H", h), ("S", s)]

    queue: List[Tuple[np.ndarray, List[str]]] = [(np.eye(2, dtype=complex), [])]
    seen = {_matrix_key(queue[0][0])}
    matrices: List[np.ndarray] = []
    decomps: List[List[str]] = []

    while queue:
        mat, seq = queue.pop(0)
        matrices.append(mat)
        decomps.append(seq)
        for name, gen in generators:
            new_mat = gen @ mat
            key = _matrix_key(new_mat)
            if key in seen:
                continue
            seen.add(key)
            queue.append((new_mat, seq + [name]))

    if len(matrices) != 24:
        raise RuntimeError(f"Expected 24 single-qubit Cliffords, got {len(matrices)}")

    return matrices, decomps


CLIFFORD_1Q_MATRICES, CLIFFORD_1Q_DECOMP = _build_clifford_catalog()


@dataclass
class CliffordGate(Gate):
    """Gate with attached 1Q Clifford decomposition for Stim."""
    clifford_seq: List[str] = None


def _stim_gate_ops(gate: Gate) -> Iterable[Tuple[str, List[int]]]:
    if isinstance(gate.content, np.ndarray):
        seq = getattr(gate, "clifford_seq", None)
        if not seq:
            raise ValueError("Clifford 1Q gate missing clifford_seq decomposition.")
        q = int(gate.qubits[0])
        for op in seq:
            yield op, [q]
        return

    if isinstance(gate.content, str):
        q0, q1 = int(gate.qubits[0]), int(gate.qubits[1])
        if gate.content == "CNOT":
            yield "CX", [q0, q1]
        elif gate.content == "CNOT_R":
            yield "CX", [q1, q0]
        elif gate.content == "CZ":
            yield "CZ", [q0, q1]
        elif gate.content == "SWAP":
            yield "SWAP", [q0, q1]
        else:
            raise ValueError(f"Unsupported Clifford 2Q gate: {gate.content}")
        return

    raise ValueError("Unsupported Clifford gate content type.")


def _pauli_support(pauli, qubit: int) -> bool:
    if hasattr(pauli, "xs") and hasattr(pauli, "zs"):
        return bool(pauli.xs[qubit] or pauli.zs[qubit])

    pauli_str = str(pauli)
    if pauli_str.startswith("-"):
        pauli_str = pauli_str[1:]
    if pauli_str.startswith("+" ):
        pauli_str = pauli_str[1:]
    if pauli_str.startswith("-i"):
        pauli_str = pauli_str[2:]
    elif pauli_str.startswith("i"):
        pauli_str = pauli_str[1:]
    return pauli_str[qubit] != "I"


def _to_stim_circuit(circuit: Circuit):
    stim = _require_stim()
    stim_circuit = stim.Circuit()
    for layer in circuit.layers:
        for gate in layer:
            for op, targets in _stim_gate_ops(gate):
                stim_circuit.append(op, targets)
    return stim_circuit


# =============================================================================
# Lightcone Computation
# =============================================================================

def compute_lightcone(
    circuit: Circuit,
    observable: str,
) -> Set[Tuple[int, int]]:
    """
    Compute exact lightcone for a Clifford circuit by backwards Pauli propagation.
    """
    stim = _require_stim()

    n_qubits = circuit.n_qubits
    if len(observable) != n_qubits:
        raise ValueError(f"Observable length {len(observable)} != n_qubits {n_qubits}")

    pauli = stim.PauliString(observable)
    lightcone: Set[Tuple[int, int]] = set()

    for layer_idx in range(len(circuit.layers) - 1, -1, -1):
        layer = circuit.layers[layer_idx]
        for gate in layer:
            if isinstance(gate.content, np.ndarray):
                q = int(gate.qubits[0])
                if _pauli_support(pauli, q):
                    lightcone.add((layer_idx, q))
                    for op, targets in _stim_gate_ops(gate):
                        c = stim.Circuit()
                        c.append(op, targets)
                        pauli = pauli.after(c)
            else:
                q0, q1 = int(gate.qubits[0]), int(gate.qubits[1])
                if _pauli_support(pauli, q0) or _pauli_support(pauli, q1):
                    lightcone.add((layer_idx, q0))
                    lightcone.add((layer_idx, q1))
                    for op, targets in _stim_gate_ops(gate):
                        c = stim.Circuit()
                        c.append(op, targets)
                        pauli = pauli.after(c)

    return lightcone


def lightcone_error_locations(
    circuit: Circuit,
    noise: dict,
    observable: str,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Get error locations restricted to the lightcone.
    """
    all_locs = error_locations(circuit, noise)
    lc = compute_lightcone(circuit, observable)
    return [(l, q, p) for (l, q, p) in all_locs if (l, q) in lc]


# =============================================================================
# Stabilizer Simulation
# =============================================================================

def stabilizer_expectation(
    circuit: Circuit,
    observable: str,
    initial_state: Optional[np.ndarray] = None,
) -> float:
    """
    Compute exact expectation value for a Clifford circuit using Stim.

    If initial_state is provided, it must be a computational basis state.
    """
    stim = _require_stim()

    n_qubits = circuit.n_qubits
    stim_circuit = _to_stim_circuit(circuit)

    prep = None
    if initial_state is not None:
        idx = int(np.argmax(np.abs(initial_state)))
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        prep = stim.Circuit()
        for q, bit in enumerate(bits):
            if bit:
                prep.append("X", [q])

    sim = stim.TableauSimulator()
    if prep is not None:
        sim.do_circuit(prep)
    sim.do_circuit(stim_circuit)
    return float(sim.peek_observable_expectation(stim.PauliString(observable)))


# =============================================================================
# Random Clifford Circuit Generation
# =============================================================================

def random_clifford_circuit(
    n_qubits: int,
    depth: int,
    rng: np.random.Generator,
    twoq_gate_names: Optional[List[str]] = None,
) -> Circuit:
    """
    Generate random Clifford circuit with brickwall structure.

    - Even layers: random 1Q Clifford gates on all qubits
    - Odd layers: random 2Q Clifford gates (CNOT, CZ, SWAP, etc.)
    """
    if twoq_gate_names is None:
        twoq_gate_names = list(CLIFFORD_2Q_GATES)

    for name in twoq_gate_names:
        if name not in CLIFFORD_2Q_GATES:
            raise ValueError(f"Non-Clifford gate: {name}")

    layers = []
    for l in range(depth):
        layer = []
        if l % 2 == 0:
            for q in range(n_qubits):
                cliff_idx = int(rng.integers(0, len(CLIFFORD_1Q_MATRICES)))
                gate = Gate(qubits=(q,), content=CLIFFORD_1Q_MATRICES[cliff_idx])
                gate.clifford_idx = cliff_idx  # type: ignore
                gate.clifford_seq = CLIFFORD_1Q_DECOMP[cliff_idx]  # type: ignore
                layer.append(gate)
        else:
            start = 0 if (l % 4 == 1) else 1
            for q in range(start, n_qubits - 1, 2):
                gate_name = rng.choice(twoq_gate_names)
                layer.append(Gate(qubits=(q, q + 1), content=gate_name))
        layers.append(layer)

    return Circuit(n_qubits=n_qubits, layers=layers)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    circuit = random_clifford_circuit(4, 10, rng)
    observable = "ZIII"

    lc = compute_lightcone(circuit, observable)
    print(f"Observable: {observable}")
    print(f"Lightcone size: {len(lc)} locations")

    ideal = stabilizer_expectation(circuit, observable)
    print(f"Ideal expectation: {ideal}")
