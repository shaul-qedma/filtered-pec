# clifford.py
"""
Clifford circuit utilities for exact lightcone computation and stabilizer simulation.

This module provides:
1. Pauli algebra (multiplication, conjugation)
2. Pauli propagation through Clifford gates
3. Exact lightcone computation via backwards propagation
4. Stabilizer simulation for exact expectation values
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from pec_shared import Circuit, Gate, STANDARD_GATES


# =============================================================================
# Pauli Algebra
# =============================================================================

# Pauli multiplication: P1 * P2 = phase * P3
# Phase encoded as power of i: 0=1, 1=i, 2=-1, 3=-i
PAULI_MULT: Dict[Tuple[str, str], Tuple[str, int]] = {
    ('I', 'I'): ('I', 0), ('I', 'X'): ('X', 0), ('I', 'Y'): ('Y', 0), ('I', 'Z'): ('Z', 0),
    ('X', 'I'): ('X', 0), ('X', 'X'): ('I', 0), ('X', 'Y'): ('Z', 1), ('X', 'Z'): ('Y', 3),
    ('Y', 'I'): ('Y', 0), ('Y', 'X'): ('Z', 3), ('Y', 'Y'): ('I', 0), ('Y', 'Z'): ('X', 1),
    ('Z', 'I'): ('Z', 0), ('Z', 'X'): ('Y', 1), ('Z', 'Y'): ('X', 3), ('Z', 'Z'): ('I', 0),
}


def pauli_mult(p1: str, p2: str) -> Tuple[str, int]:
    """Multiply two single-qubit Paulis. Returns (result, phase_power)."""
    return PAULI_MULT[(p1, p2)]


def phase_to_complex(phase_power: int) -> complex:
    """Convert phase power to complex number: i^phase_power."""
    return 1j ** (phase_power % 4)


# =============================================================================
# 2-Qubit Clifford Gate Conjugation Tables
# =============================================================================
# For gate G and 2-qubit Pauli P = P0⊗P1: G† P G = phase * (P0'⊗P1')
# Each table maps (P0, P1) -> ((P0', P1'), phase_power)

# Pauli matrices for direct computation
_PAULI_MATS = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _build_gate_table(gate_matrix: np.ndarray) -> Dict[Tuple[str, str], Tuple[Tuple[str, str], int]]:
    """
    Build conjugation table for a 2-qubit gate by direct matrix computation.
    Computes G† (P0⊗P1) G for all Pauli pairs.
    """
    table = {}
    paulis = ['I', 'X', 'Y', 'Z']
    G = gate_matrix
    Gdag = G.conj().T
    
    for p0 in paulis:
        for p1 in paulis:
            # Input Pauli
            P = np.kron(_PAULI_MATS[p0], _PAULI_MATS[p1])
            
            # Conjugate: G† P G
            result = Gdag @ P @ G
            
            # Find which Pauli it is
            found = False
            for q0 in paulis:
                for q1 in paulis:
                    Q = np.kron(_PAULI_MATS[q0], _PAULI_MATS[q1])
                    
                    for phase_power in range(4):
                        phase = 1j ** phase_power
                        if np.allclose(result, phase * Q):
                            table[(p0, p1)] = ((q0, q1), phase_power)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                raise ValueError(f"Could not identify result for {p0}{p1}")
    
    return table


def _build_swap_table() -> Dict[Tuple[str, str], Tuple[Tuple[str, str], int]]:
    """SWAP just exchanges the two qubits."""
    table = {}
    for p0 in ['I', 'X', 'Y', 'Z']:
        for p1 in ['I', 'X', 'Y', 'Z']:
            table[(p0, p1)] = ((p1, p0), 0)
    return table


# Build the tables once at module load using the gate matrices from pec_shared
CNOT_TABLE = _build_gate_table(STANDARD_GATES['CNOT'])
CNOT_R_TABLE = _build_gate_table(STANDARD_GATES['CNOT_R'])
CZ_TABLE = _build_gate_table(STANDARD_GATES['CZ'])
SWAP_TABLE = _build_swap_table()  # SWAP is simple enough

GATE_TABLES = {
    'CNOT': CNOT_TABLE,
    'CNOT_R': CNOT_R_TABLE,
    'CZ': CZ_TABLE,
    'SWAP': SWAP_TABLE,
}


# =============================================================================
# Single-Qubit Clifford Gates
# =============================================================================

# The 24-element single-qubit Clifford group
# Each element maps (X, Y, Z) to (±P1, ±P2, ±P3) where P1,P2,P3 is a permutation
# We represent as: {X: (sign, P), Y: (sign, P), Z: (sign, P)}

CLIFFORD_1Q = [
    # Identity
    {'X': (1, 'X'), 'Y': (1, 'Y'), 'Z': (1, 'Z')},
    # X, Y, Z rotations by π (Pauli gates squared = I, but as conjugation they're nontrivial)
    {'X': (1, 'X'), 'Y': (-1, 'Y'), 'Z': (-1, 'Z')},  # X conjugation
    {'X': (-1, 'X'), 'Y': (1, 'Y'), 'Z': (-1, 'Z')},  # Y conjugation
    {'X': (-1, 'X'), 'Y': (-1, 'Y'), 'Z': (1, 'Z')},  # Z conjugation
    # Hadamard-like (X↔Z)
    {'X': (1, 'Z'), 'Y': (-1, 'Y'), 'Z': (1, 'X')},   # H
    {'X': (-1, 'Z'), 'Y': (-1, 'Y'), 'Z': (-1, 'X')}, # XHX = -H effectively
    {'X': (1, 'Z'), 'Y': (1, 'Y'), 'Z': (-1, 'X')},
    {'X': (-1, 'Z'), 'Y': (1, 'Y'), 'Z': (1, 'X')},
    # S-like (X→Y)
    {'X': (1, 'Y'), 'Y': (-1, 'X'), 'Z': (1, 'Z')},   # S
    {'X': (-1, 'Y'), 'Y': (1, 'X'), 'Z': (1, 'Z')},   # S†
    {'X': (1, 'Y'), 'Y': (1, 'X'), 'Z': (-1, 'Z')},
    {'X': (-1, 'Y'), 'Y': (-1, 'X'), 'Z': (-1, 'Z')},
    # Permutations X→Y→Z→X (axis rotations)
    {'X': (1, 'Y'), 'Y': (1, 'Z'), 'Z': (1, 'X')},
    {'X': (-1, 'Y'), 'Y': (-1, 'Z'), 'Z': (1, 'X')},
    {'X': (1, 'Y'), 'Y': (-1, 'Z'), 'Z': (-1, 'X')},
    {'X': (-1, 'Y'), 'Y': (1, 'Z'), 'Z': (-1, 'X')},
    # Permutations X→Z→Y→X
    {'X': (1, 'Z'), 'Y': (1, 'X'), 'Z': (1, 'Y')},
    {'X': (-1, 'Z'), 'Y': (-1, 'X'), 'Z': (1, 'Y')},
    {'X': (1, 'Z'), 'Y': (-1, 'X'), 'Z': (-1, 'Y')},
    {'X': (-1, 'Z'), 'Y': (1, 'X'), 'Z': (-1, 'Y')},
    # More permutations
    {'X': (-1, 'X'), 'Y': (1, 'Z'), 'Z': (1, 'Y')},
    {'X': (1, 'X'), 'Y': (-1, 'Z'), 'Z': (1, 'Y')},
    {'X': (-1, 'X'), 'Y': (-1, 'Z'), 'Z': (-1, 'Y')},
    {'X': (1, 'X'), 'Y': (1, 'Z'), 'Z': (-1, 'Y')},
]


def apply_clifford_1q(pauli: str, clifford_idx: int) -> Tuple[str, int]:
    """
    Apply single-qubit Clifford conjugation to a Pauli.
    Returns (new_pauli, sign) where sign is +1 or -1.
    """
    if pauli == 'I':
        return ('I', 1)
    
    cliff = CLIFFORD_1Q[clifford_idx % 24]
    sign, new_pauli = cliff[pauli]
    return (new_pauli, sign)


# =============================================================================
# Pauli String Propagation
# =============================================================================

@dataclass
class PauliString:
    """A Pauli string with phase: phase * P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1}"""
    paulis: List[str]  # List of 'I', 'X', 'Y', 'Z'
    phase: complex = 1.0
    
    @classmethod
    def from_string(cls, s: str) -> 'PauliString':
        """Create from string like 'XIZI'."""
        return cls(paulis=list(s), phase=1.0)
    
    def __str__(self) -> str:
        phase_str = ""
        if self.phase == -1:
            phase_str = "-"
        elif self.phase == 1j:
            phase_str = "i"
        elif self.phase == -1j:
            phase_str = "-i"
        return phase_str + ''.join(self.paulis)
    
    def copy(self) -> 'PauliString':
        return PauliString(paulis=self.paulis.copy(), phase=self.phase)
    
    @property
    def support(self) -> Set[int]:
        """Qubits where Pauli is not identity."""
        return {i for i, p in enumerate(self.paulis) if p != 'I'}


def propagate_through_2q_gate(
    pauli: PauliString,
    gate_name: str,
    q0: int,
    q1: int,
) -> PauliString:
    """
    Propagate Pauli string through 2-qubit Clifford gate.
    Computes G† P G.
    """
    table = GATE_TABLES.get(gate_name)
    if table is None:
        raise ValueError(f"Unknown 2Q Clifford gate: {gate_name}")
    
    p0, p1 = pauli.paulis[q0], pauli.paulis[q1]
    (new_p0, new_p1), phase_power = table[(p0, p1)]
    
    result = pauli.copy()
    result.paulis[q0] = new_p0
    result.paulis[q1] = new_p1
    result.phase *= phase_to_complex(phase_power)
    
    return result


def propagate_through_1q_clifford(
    pauli: PauliString,
    qubit: int,
    clifford_idx: int,
) -> PauliString:
    """Propagate Pauli string through single-qubit Clifford gate."""
    new_p, sign = apply_clifford_1q(pauli.paulis[qubit], clifford_idx)
    
    result = pauli.copy()
    result.paulis[qubit] = new_p
    result.phase *= sign
    
    return result


# =============================================================================
# Lightcone Computation
# =============================================================================

def compute_lightcone(
    circuit: Circuit,
    observable: str,
) -> Set[Tuple[int, int]]:
    """
    Compute exact lightcone for a Clifford circuit by backwards Pauli propagation.
    
    Args:
        circuit: Clifford circuit (must contain only Clifford gates)
        observable: Pauli string observable (e.g., "IZZI")
    
    Returns:
        Set of (layer_idx, qubit) pairs that are in the lightcone.
        These are the error locations that affect the observable.
    """
    n_qubits = circuit.n_qubits
    if len(observable) != n_qubits:
        raise ValueError(f"Observable length {len(observable)} != n_qubits {n_qubits}")
    
    pauli = PauliString.from_string(observable)
    lightcone: Set[Tuple[int, int]] = set()
    
    # Propagate backwards through circuit
    for layer_idx in range(len(circuit.layers) - 1, -1, -1):
        layer = circuit.layers[layer_idx]
        
        for gate in layer:
            if isinstance(gate.content, np.ndarray):
                # 1Q gate represented as matrix - must be Clifford
                # For now, we treat it as a random Clifford (index stored elsewhere)
                # or we can skip it if it's identity-like
                qubit = gate.qubits[0]
                
                # Check if this qubit is in the current support
                if pauli.paulis[qubit] != 'I':
                    lightcone.add((layer_idx, qubit))
                    
                    # If gate has clifford_idx attribute, propagate
                    if hasattr(gate, 'clifford_idx'):
                        pauli = propagate_through_1q_clifford(
                            pauli, qubit, gate.clifford_idx
                        )
                    # Otherwise assume identity (no change to Pauli)
                    
            elif isinstance(gate.content, str):
                # 2Q gate
                gate_name = gate.content
                q0, q1 = gate.qubits
                
                # Check if gate overlaps current Pauli support
                if pauli.paulis[q0] != 'I' or pauli.paulis[q1] != 'I':
                    # This gate is in the lightcone
                    lightcone.add((layer_idx, q0))
                    lightcone.add((layer_idx, q1))
                    
                    # Propagate Pauli through gate
                    if gate_name in GATE_TABLES:
                        pauli = propagate_through_2q_gate(pauli, gate_name, q0, q1)
                    else:
                        raise ValueError(f"Non-Clifford 2Q gate: {gate_name}")
    
    return lightcone


def lightcone_error_locations(
    circuit: Circuit,
    noise: dict,
    observable: str,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Get error locations restricted to the lightcone.
    
    Returns list of (layer, qubit, noise_probs) for locations in the lightcone.
    """
    from pec_shared import error_locations, kraus_to_probs
    
    # Get all error locations
    all_locs = error_locations(circuit, noise)
    
    # Get lightcone
    lc = compute_lightcone(circuit, observable)
    
    # Filter to lightcone
    lc_locs = [(l, q, p) for (l, q, p) in all_locs if (l, q) in lc]
    
    return lc_locs


# =============================================================================
# Stabilizer Simulation
# =============================================================================

def stabilizer_expectation(
    circuit: Circuit,
    observable: str,
    initial_state: Optional[np.ndarray] = None,
) -> float:
    """
    Compute exact expectation value for a Clifford circuit.
    
    For computational basis input |0...0⟩:
    1. Propagate observable backwards through circuit
    2. Result is ±1 if propagated Pauli is in {I,Z}^n, else 0
    
    Args:
        circuit: Clifford circuit
        observable: Pauli string (e.g., "XZIY")
        initial_state: If None, assumes |0...0⟩. Otherwise must be computational basis.
    
    Returns:
        Exact expectation value (-1, 0, or 1 for |0...0⟩ input)
    """
    n_qubits = circuit.n_qubits
    
    # Handle initial state
    if initial_state is not None:
        # Find which computational basis state it is
        idx = np.argmax(np.abs(initial_state))
        phase = initial_state[idx]
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
    else:
        bits = [0] * n_qubits
    
    # Propagate observable backwards
    pauli = PauliString.from_string(observable)
    
    for layer_idx in range(len(circuit.layers) - 1, -1, -1):
        layer = circuit.layers[layer_idx]
        
        for gate in layer:
            if isinstance(gate.content, np.ndarray):
                qubit = gate.qubits[0]
                if hasattr(gate, 'clifford_idx'):
                    pauli = propagate_through_1q_clifford(
                        pauli, qubit, gate.clifford_idx
                    )
            elif isinstance(gate.content, str):
                gate_name = gate.content
                if gate_name in GATE_TABLES:
                    q0, q1 = gate.qubits
                    pauli = propagate_through_2q_gate(pauli, gate_name, q0, q1)
                else:
                    raise ValueError(f"Non-Clifford gate: {gate_name}")
    
    # Now compute ⟨bits|pauli|bits⟩
    # For computational basis state |b⟩:
    # ⟨b|I|b⟩ = 1
    # ⟨b|X|b⟩ = 0
    # ⟨b|Y|b⟩ = 0
    # ⟨b|Z|b⟩ = (-1)^b
    
    result = pauli.phase
    for i, p in enumerate(pauli.paulis):
        if p == 'X' or p == 'Y':
            return 0.0
        elif p == 'Z':
            result *= (-1) ** bits[i]
    
    return float(np.real(result))


# =============================================================================
# Random Clifford Circuit Generation
# =============================================================================

@dataclass
class CliffordGate(Gate):
    """Gate with Clifford index for 1Q gates."""
    clifford_idx: int = 0


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
        twoq_gate_names = ['CNOT', 'CNOT_R', 'CZ', 'SWAP']
    
    # Validate all gates are Clifford
    for name in twoq_gate_names:
        if name not in GATE_TABLES:
            raise ValueError(f"Non-Clifford gate: {name}")
    
    layers = []
    
    for l in range(depth):
        layer = []
        
        if l % 2 == 0:
            # 1Q Clifford layer
            for q in range(n_qubits):
                cliff_idx = int(rng.integers(0, 24))
                # Store as matrix placeholder with clifford_idx
                gate = Gate(qubits=(q,), content=np.eye(2, dtype=complex))
                gate.clifford_idx = cliff_idx  # type: ignore
                layer.append(gate)
        else:
            # 2Q gate layer (brickwall)
            start = 0 if (l % 4 == 1) else 1
            for q in range(start, n_qubits - 1, 2):
                gate_name = rng.choice(twoq_gate_names)
                layer.append(Gate(qubits=(q, q + 1), content=gate_name))
        
        layers.append(layer)
    
    return Circuit(n_qubits=n_qubits, layers=layers)


# =============================================================================
# Verification
# =============================================================================

def _verify_tables():
    """Verify conjugation tables are correct by comparing to matrix computation."""
    from pec_shared import PAULI_I, PAULI_X, PAULI_Y, PAULI_Z
    
    paulis = {'I': PAULI_I, 'X': PAULI_X, 'Y': PAULI_Y, 'Z': PAULI_Z}
    
    for gate_name, gate_matrix in STANDARD_GATES.items():
        if gate_name not in GATE_TABLES:
            continue
            
        table = GATE_TABLES[gate_name]
        G = gate_matrix
        
        for (p0, p1), ((exp_p0, exp_p1), phase_power) in table.items():
            # Compute G† (P0⊗P1) G
            P = np.kron(paulis[p0], paulis[p1])
            result = G.conj().T @ P @ G
            
            # Expected result
            expected = phase_to_complex(phase_power) * np.kron(paulis[exp_p0], paulis[exp_p1])
            
            if not np.allclose(result, expected):
                print(f"FAIL: {gate_name} {p0}{p1} -> {exp_p0}{exp_p1} (phase {phase_power})")
                print(f"  Result:\n{result}")
                print(f"  Expected:\n{expected}")
                return False
    
    print("All conjugation tables verified ✓")
    return True


if __name__ == "__main__":
    _verify_tables()
    
    # Quick test
    print("\nTest: Simple circuit")
    rng = np.random.default_rng(42)
    circuit = random_clifford_circuit(4, 10, rng)
    observable = "ZIII"
    
    lc = compute_lightcone(circuit, observable)
    print(f"Observable: {observable}")
    print(f"Lightcone size: {len(lc)} locations")
    
    # Compute ideal expectation
    ideal = stabilizer_expectation(circuit, observable)
    print(f"Ideal expectation: {ideal}")