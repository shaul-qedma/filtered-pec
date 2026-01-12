# noise_model.py
"""
Noise model configuration and generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from constants import DEFAULT_INFIDELITY
from pec_shared import (
    CLIFFORD_2Q_GATES,
    pauli_to_kraus,
    random_pauli_probs,
    depolarizing_probs,
)


@dataclass
class NoiseConfig:
    """
    Noise model configuration.
    
    Four modes:
    1. kraus: Explicit Kraus operators per gate per qubit
    2. pauli: Pauli probabilities [p_I, p_X, p_Y, p_Z] per gate per qubit  
    3. per_gate: Per-gate infidelity specification
    4. depolarizing: Global depolarizing noise with isotropic/random distribution
    
    YAML examples:
    
    # Mode 1: Explicit Kraus (rarely used)
    noise_model:
      type: kraus
      gates:
        CZ:
          qubit_0: [[[1,0],[0,1]], [[0,1],[1,0]], ...]  # 4 matrices as nested lists
          qubit_1: [[[1,0],[0,1]], [[0,1],[1,0]], ...]
    
    # Mode 2: Pauli probabilities
    noise_model:
      type: pauli
      gates:
        CZ:
          qubit_0: [0.99, 0.003, 0.004, 0.003]  # [p_I, p_X, p_Y, p_Z]
          qubit_1: [0.98, 0.01, 0.005, 0.005]
        CNOT:
          qubit_0: [0.985, 0.005, 0.005, 0.005]
          qubit_1: [0.985, 0.005, 0.005, 0.005]
    
    # Mode 3: Per-gate infidelity
    noise_model:
      type: per_gate
      gates:
        CZ:
          infidelity: 0.01
        CNOT:
          infidelity: 0.02
    
    # Mode 4: Depolarizing
    noise_model:
      type: depolarizing
      infidelity: 0.01
      isotropic: true   # true for uniform X/Y/Z, false for random
    """
    type: str = "depolarizing"
    infidelity: Optional[float] = DEFAULT_INFIDELITY
    isotropic: bool = True
    gates: Optional[Dict[str, Any]] = None


def parse_noise_config(config: Optional[Dict]) -> NoiseConfig:
    """Parse noise configuration from dict (e.g., from YAML)."""
    if config is None:
        return NoiseConfig()
    
    return NoiseConfig(
        type=config.get("type", "depolarizing"),
        infidelity=config.get("infidelity", DEFAULT_INFIDELITY),
        isotropic=config.get("isotropic", True),
        gates=config.get("gates"),
    )


def generate_noise_model(
    rng: np.random.Generator,
    config: Optional[Union[Dict, NoiseConfig]] = None,
    gate_names: Optional[List[str]] = None,
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Generate noise model based on configuration.
    
    Args:
        rng: Random number generator (used for random/infidelity modes)
        config: Noise configuration dict or NoiseConfig object
        gate_names: List of gate names to generate noise for
    
    Returns:
        Dict mapping gate name -> (kraus_q0, kraus_q1)
        where kraus_qi is a list of 4 Kraus operators for qubit i
    """
    if gate_names is None:
        gate_names = list(CLIFFORD_2Q_GATES)
    
    if config is None:
        cfg = NoiseConfig()
    elif isinstance(config, dict):
        cfg = parse_noise_config(config)
    else:
        cfg = config
    
    noise: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    
    if cfg.type == "kraus":
        noise = _generate_kraus_noise(cfg, gate_names)
    elif cfg.type == "pauli":
        noise = _generate_pauli_noise(cfg, gate_names)
    elif cfg.type == "per_gate":
        noise = _generate_per_gate_noise(rng, cfg, gate_names)
    elif cfg.type == "depolarizing":
        noise = _generate_depolarizing_noise(rng, cfg, gate_names)
    else:
        raise ValueError(f"Unknown noise model type: {cfg.type}")
    
    return noise


def _generate_kraus_noise(
    cfg: NoiseConfig,
    gate_names: List[str],
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Generate noise from explicit Kraus operators."""
    if cfg.gates is None:
        raise ValueError("kraus noise type requires 'gates' configuration")
    
    noise = {}
    for gate in gate_names:
        if gate not in cfg.gates:
            raise ValueError(f"Missing Kraus operators for gate '{gate}'")
        
        gate_cfg = cfg.gates[gate]
        kraus_q0 = [np.array(k, dtype=complex) for k in gate_cfg["qubit_0"]]
        kraus_q1 = [np.array(k, dtype=complex) for k in gate_cfg["qubit_1"]]
        
        _validate_kraus(kraus_q0, f"{gate}/qubit_0")
        _validate_kraus(kraus_q1, f"{gate}/qubit_1")
        
        noise[gate] = (kraus_q0, kraus_q1)
    
    return noise


def _generate_pauli_noise(
    cfg: NoiseConfig,
    gate_names: List[str],
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Generate noise from Pauli probability distributions."""
    if cfg.gates is None:
        raise ValueError("pauli noise type requires 'gates' configuration")
    
    noise = {}
    for gate in gate_names:
        if gate not in cfg.gates:
            raise ValueError(f"Missing Pauli probabilities for gate '{gate}'")
        
        gate_cfg = cfg.gates[gate]
        
        # Support both per-qubit and symmetric specification
        if "qubit_0" in gate_cfg:
            probs_q0 = np.array(gate_cfg["qubit_0"], dtype=float)
            probs_q1 = np.array(gate_cfg.get("qubit_1", gate_cfg["qubit_0"]), dtype=float)
        elif "probs" in gate_cfg:
            # Symmetric: same for both qubits
            probs_q0 = np.array(gate_cfg["probs"], dtype=float)
            probs_q1 = probs_q0.copy()
        else:
            # Direct list format: [p_I, p_X, p_Y, p_Z]
            probs_q0 = np.array(gate_cfg, dtype=float)
            probs_q1 = probs_q0.copy()
        
        noise[gate] = (pauli_to_kraus(probs_q0), pauli_to_kraus(probs_q1))
    
    return noise


def _generate_per_gate_noise(
    rng: np.random.Generator,
    cfg: NoiseConfig,
    gate_names: List[str],
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Generate noise from per-gate infidelity specification."""
    if cfg.gates is None:
        raise ValueError("per_gate noise type requires 'gates' configuration")
    
    noise = {}
    
    for gate in gate_names:
        if gate not in cfg.gates:
            raise ValueError(f"No infidelity specified for gate '{gate}'")
        
        gate_cfg = cfg.gates[gate]
        if isinstance(gate_cfg, dict):
            infidelity = gate_cfg.get("infidelity", cfg.infidelity)
        else:
            # Direct infidelity value
            infidelity = float(gate_cfg)
        
        if infidelity is None:
            raise ValueError(f"No infidelity specified for gate '{gate}'")
        
        p_I = 1.0 - infidelity
        
        # For per-gate, we use depolarizing (isotropic) by default
        probs_q0 = depolarizing_probs(p_I)
        probs_q1 = depolarizing_probs(p_I)
        
        noise[gate] = (pauli_to_kraus(probs_q0), pauli_to_kraus(probs_q1))
    
    return noise


def _generate_depolarizing_noise(
    rng: np.random.Generator,
    cfg: NoiseConfig,
    gate_names: List[str],
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Generate depolarizing noise with global infidelity."""
    if cfg.infidelity is None:
        raise ValueError("depolarizing noise type requires 'infidelity' field")
    
    p_I = 1.0 - cfg.infidelity
    
    if cfg.isotropic:
        probs = depolarizing_probs(p_I)
    else:
        probs = random_pauli_probs(rng, p_I)
    
    noise = {}
    for gate in gate_names:
        noise[gate] = (pauli_to_kraus(probs), pauli_to_kraus(probs))
    
    return noise


def _validate_kraus(kraus: List[np.ndarray], name: str) -> None:
    """Validate Kraus operators satisfy completeness relation."""
    if len(kraus) != 4:
        raise ValueError(f"{name}: expected 4 Kraus operators, got {len(kraus)}")
    
    for i, k in enumerate(kraus):
        if k.shape != (2, 2):
            raise ValueError(f"{name}[{i}]: expected 2x2 matrix, got {k.shape}")
    
    # Check completeness: sum_i K_i^dag @ K_i = I
    total = sum(k.conj().T @ k for k in kraus)
    if not np.allclose(total, np.eye(2)):
        raise ValueError(f"{name}: Kraus operators don't satisfy completeness relation")