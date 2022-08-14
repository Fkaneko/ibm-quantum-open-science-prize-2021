import copy
import logging
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import qiskit.quantum_info as qi
from omegaconf import DictConfig, OmegaConf
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.quantum_info import TwoQubitBasisDecomposer
from qiskit.quantum_info.operators import Operator

log = logging.getLogger(__name__)


class QcKeys(Enum):
    def _generate_next_value_(name, start, count, last_values):  # type: ignore
        return name

    NORMAL = auto()
    TAIL = auto()
    HEAD = auto()
    ABSORB_PREV_Q0_RZ = auto()
    DEVICE_NATIVE_RZX = auto()
    QISKIT_KAK = auto()


@dataclass
class TrotterSingleStep:
    name: str
    num_qubits: int
    qrs_for_each_qc: Dict[int, List[int]]

    def __post_init__(self) -> None:
        if isinstance(self.qrs_for_each_qc, DictConfig):
            self.qrs_for_each_qc = OmegaConf.to_container(self.qrs_for_each_qc)
        max_reg = max([max(qr_list) for qr_list in self.qrs_for_each_qc.values()])
        if max_reg >= self.num_qubits:
            raise ValueError(f"max_reg:{max_reg} should be less than num_qubits:{self.num_qubits}")


@dataclass
class TrotterUnitQc:
    name: str
    key: QcKeys
    flip_cnot_directions: Tuple[bool, bool, bool]

    def __post_init__(self) -> None:
        if isinstance(self.key, str):
            self.key = QcKeys[self.key]


def diff_circuit(qc_1: QuantumCircuit, qc_2: QuantumCircuit, epsilon: float = 1.0e-12) -> bool:
    flag = np.all(np.array(Operator(qc_1) - Operator(qc_2)) < epsilon)
    return bool(flag)  # for mypy, explicit casting


def get_device_native_rzx(t: Parameter, name: str = "rzx_wo_echo") -> QuantumCircuit:
    """
    device native rzx gate for pulse optimization.
    Rzx gate with explicit echo

    Parameters
    -----
        t: Parameter
            rotation angle on Rzx
        name: str
            name for this sub circuit

    Returns
    -----
        qc: QuantumCircuit
            Rzx gate

    Note
        theta == pi/2 -> cnot gate
    """
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name=name)
    qc.rzx(t / 2, 0, 1)
    qc.x(0)
    qc.rzx(-t / 2, 0, 1)
    qc.x(0)
    return qc.to_instruction()


def get_Hxxx_circuit(
    t: Parameter,
    trotter_qc: TrotterUnitQc,
) -> QuantumCircuit:

    qr = QuantumRegister(2)
    order_name = "plain" if t > 0 else "inverse"
    qc = QuantumCircuit(
        # qr, name=trotter_qc.name + "_" + trotter_qc.key.value + f" t: {t:>3.2f}"
        qr,
        name=trotter_qc.name + " " + order_name + " " + f" t: {abs(t):>3.2f}",
    )

    _add_three_cnot_decomposition = partial(
        add_three_cnot_decomposition,
        qc=qc,
        t=t,
        q_0=0,
        q_1=1,
        flip_cnot_directions=trotter_qc.flip_cnot_directions,
    )
    if trotter_qc.key == QcKeys.NORMAL:
        qc = _add_three_cnot_decomposition()
    elif trotter_qc.key == QcKeys.TAIL:
        qc = _add_three_cnot_decomposition(remove_tail=True)
    elif trotter_qc.key == QcKeys.HEAD:
        qc = _add_three_cnot_decomposition(remove_head=True)
    elif trotter_qc.key == QcKeys.QISKIT_KAK:
        qc = add_qiskit_kak_decomposition(
            qc=qc,
            t=t,
            q_0=0,
            q_1=1,
        )
    else:
        raise ValueError

    return qc


def add_three_cnot_decomposition(
    qc: QuantumCircuit,
    t: Parameter,
    q_0: QuantumRegister,
    q_1: QuantumRegister,
    remove_head: bool = False,
    remove_tail: bool = False,
    flip_cnot_directions: Tuple[bool, bool, bool] = (False, True, False),
) -> QuantumCircuit:
    def _cnot(qc: QuantumCircuit, qr: QuantumRegister, flip_direction: bool = False) -> QuantumCircuit:
        # from https://qiskit.org/documentation/stubs/qiskit.transpiler.passes.GateDirection.html#qiskit.transpiler.passes.GateDirection
        if flip_direction:
            qc.h(qr[0])
            qc.h(qr[1])
            qc.cnot(qr[1], qr[0])
            qc.h(qr[0])
            qc.h(qr[1])
        else:
            qc.cnot(qr[0], qr[1])
        return qc

    if not remove_head:
        qc.rz(-np.pi * 1 / 2, q_1)
    qc = _cnot(qc=qc, qr=[q_1, q_0], flip_direction=flip_cnot_directions[0])
    qc.rz(1 / 2 * np.pi + 2 * t, q_0)
    qc.ry(-2 * t - 1 / 2 * np.pi, q_1)
    qc = _cnot(qc=qc, qr=[q_0, q_1], flip_direction=flip_cnot_directions[1])
    qc.ry(+2 * t + 1 / 2 * np.pi, q_1)
    qc = _cnot(qc=qc, qr=[q_1, q_0], flip_direction=flip_cnot_directions[2])
    if not remove_tail:
        qc.rz(np.pi * 1 / 2, q_0)

    return qc.to_instruction()


def add_qiskit_kak_decomposition(
    qc: QuantumCircuit,
    t: float,
    q_0: QuantumRegister,
    q_1: QuantumRegister,
) -> QuantumCircuit:
    qc.rzz(2 * t, q_0, q_1)
    qc.ryy(2 * t, q_0, q_1)
    qc.rxx(2 * t, q_0, q_1)
    qc_mat = qi.Operator(qc)
    kak = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
    kak_qc = kak(np.array(qc_mat))
    kak_qc.name = qc.name
    return kak_qc.to_instruction()


def make_trotter_single_step(
    t: Parameter,
    trotter_single_step: TrotterSingleStep,
    trotter_qcs: Tuple[TrotterUnitQc, ...],
) -> dict:
    """

    Parameters
    -----

    Returns
    -----
        single_trotter_steps: Dict

    """

    def _generate_step(
        qc: QuantumCircuit,
        qrs_for_each_qc: Dict[int, List[int]],
        trotter_qcs: Tuple[TrotterUnitQc, ...],
    ) -> QuantumCircuit:
        assert len(qrs_for_each_qc) == len(trotter_qcs)
        for i, trotter_qc in enumerate(trotter_qcs):
            qr = qrs_for_each_qc[i]
            circ = get_Hxxx_circuit(t=t, trotter_qc=trotter_qc)
            qc.append(circ, [qr[0], qr[1]])
        return qc

    qr = QuantumRegister(trotter_single_step.num_qubits)
    qc = QuantumCircuit(qr, name=trotter_single_step.name)
    single_trotter_step = _generate_step(
        qc=qc,
        qrs_for_each_qc=trotter_single_step.qrs_for_each_qc,
        trotter_qcs=trotter_qcs,
    )
    log.info(f"Decomposed unit trotter_step, \n {single_trotter_step.decompose().draw()}")
    log.info(f"unit trotter_step, \n {single_trotter_step.draw()}")
    return single_trotter_step.to_instruction()


def generate_trotter_n_steps_on_qc(
    initial_qc: QuantumCircuit,
    trotter_single_step: TrotterSingleStep,
    trotter_qcs: Tuple[TrotterUnitQc, ...],
    trotter_steps: int = 8,
    target_time: float = np.pi,
    is_inverse_step: Optional[List[bool]] = None,
) -> QuantumCircuit:

    # Parameterize variable t to be evaluated at t=pi later
    # t = Parameter("t")
    t = target_time / trotter_steps
    single_trotter_step = make_trotter_single_step(
        t=t, trotter_single_step=trotter_single_step, trotter_qcs=trotter_qcs
    )

    if is_inverse_step:
        assert len(is_inverse_step) == trotter_steps
        if any(is_inverse_step):
            flip_single_step = copy.deepcopy(trotter_single_step)
            flip_single_step.name = "Inverse_" + flip_single_step.name
            # reverse qc order
            qrs_for_each_qc = trotter_single_step.qrs_for_each_qc
            flip_single_step.qrs_for_each_qc = dict(zip(reversed(qrs_for_each_qc.keys()), qrs_for_each_qc.values()))
            inverse_trotter_step = make_trotter_single_step(
                t=-t, trotter_single_step=flip_single_step, trotter_qcs=trotter_qcs
            )
    else:
        is_inverse_step = [False] * trotter_steps

    qr = initial_qc.qregs[0]

    # Simulate time evolution under H_heis3 Hamiltonian
    for trotter_step in range(trotter_steps):
        if is_inverse_step[trotter_step]:
            initial_qc.append(inverse_trotter_step, [qr[1], qr[3], qr[5]])
        else:
            initial_qc.append(single_trotter_step, [qr[1], qr[3], qr[5]])

    if isinstance(t, Parameter):
        # Evaluate simulation at target_time (t=pi) meaning each trotter step
        # evolves pi/trotter_steps in time
        initial_qc = initial_qc.bind_parameters({t: target_time / trotter_steps})
    log.info(f"Full circuit, \n {str(initial_qc)}")
    return initial_qc


def add_measurement(
    input_qcs: List[QuantumCircuit],
    meas_qr_inds: List[int] = [1, 3, 5],
) -> List[QuantumCircuit]:
    meas_qcs = []
    for calib_qc in input_qcs:
        cr = ClassicalRegister(len(meas_qr_inds))
        qr = calib_qc.qregs[0]
        qc = QuantumCircuit(qr, cr)
        # qc.barrier(*calib_qc.qregs[0])
        qc = qc + calib_qc
        qc.barrier(qr)
        for i, qr_ind in enumerate(meas_qr_inds):
            qc.measure(qr[qr_ind], cr[i])
        meas_qcs.append(qc)
    return meas_qcs
