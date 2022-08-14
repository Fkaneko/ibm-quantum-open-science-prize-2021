import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, complete_meas_cal
from qiskit.ignis.verification.tomography.data import count_keys
from qiskit.providers import Job
from qiskit.result.result import Result

from src.trotter_steps import TrotterSingleStep, TrotterUnitQc, add_measurement, generate_trotter_n_steps_on_qc

# observed state
OBSERVE_STATES_3_QUBITS = ("000", "001", "010", "011", "100", "101", "110", "111")

# IBMQ job tag prefix
SINGLE_CALIB_TAG = "SINGLE_CALIB_"
INVERSE_DATA_TAG = "INVERSE_DATA_"

OPERATION_ORDER_IN_STEP_0_TAG = "1-3/3-5"
OPERATION_ORDER_IN_STEP_1_TAG = "3-5/1-3"

log = logging.getLogger(__name__)

# At TargetProcess.DATASET_GENERATION, append normal trotter step qc circuit on IBMQ job
# such a job has job.tag "add_3_plain_qc"
ADD_3_PLAIN_QC = "add_3_plain_qc"


class CalibName(Enum):
    MEAS = "meas_calibs"
    TROTTER_INVERSE = "trotter_inverse_calibs"
    NO_CALIB = "no_calibration"


class TrotterCalibParam(NamedTuple):
    time: float
    trotter_steps: int
    inverse_list: List[bool]

    @property
    def single_step_time(self) -> float:
        if self.trotter_steps < 0:
            return -1
        else:
            return self.time / self.trotter_steps

    @classmethod
    def empty(cls) -> "TrotterCalibParam":
        return cls(time=-1, trotter_steps=-1, inverse_list=[False, False])

    @classmethod
    def add_3_plain_qc(cls) -> List["TrotterCalibParam"]:
        time = np.pi
        inverse_mode = InverseMode.NO_INVERSE
        params = []
        for trotter_steps in [8, 10, 12]:
            inverse_list = inverse_mode.calc_inverse_list(trotter_steps=trotter_steps)
            params.append(cls(time=time, trotter_steps=trotter_steps, inverse_list=inverse_list))
        return params


def measurement_caribration(
    qr_size: int = 7,
    qubit_list: List[int] = [1, 3, 5],
) -> CompleteMeasFitter:
    # from https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/noise/3_measurement_error_mitigation.ipynb

    log.info(f"Preparing measurement caribration on qr:{qubit_list} ...")
    # Generate the calibration circuits
    qr = QuantumRegister(qr_size)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel="mcal")
    return meas_calibs, state_labels


class InverseMode(Enum):
    HALF = auto()
    REPEAT = auto()
    NO_INVERSE = auto()

    def calc_inverse_list(self, trotter_steps: int) -> List[bool]:
        if self == InverseMode.HALF:
            return ([False] * (trotter_steps >> 1)) + ([True] * (trotter_steps >> 1))
        elif self == InverseMode.REPEAT:
            return [False, True] * (trotter_steps >> 1)
        elif self == InverseMode.NO_INVERSE:
            return [False] * trotter_steps
        else:
            raise ValueError

    @staticmethod
    def find_mode_from_list(inverse_list: List[bool]) -> "InverseMode":
        if not any(inverse_list):
            return InverseMode.NO_INVERSE
        elif inverse_list[0] == inverse_list[1]:
            return InverseMode.HALF
        elif inverse_list[0] != inverse_list[1]:
            return InverseMode.REPEAT
        else:
            raise ValueError


def get_calib_qc_params(
    base_time: float = np.pi,
    calib_trotter_step_list: List[int] = [2, 4, 8],
    sample_times: List[float] = [0.5, 1.0, 1.5],
    max_num_circuits: int = 300,
) -> List[Tuple[float, int, List[bool]]]:

    sample_times = np.array(sample_times) * base_time
    log.info(f"Trotter calib sampling times: {sample_times/np.pi} x PI")
    trotter_calib_params = []
    for time in sample_times:
        for inverse_mode in [InverseMode.HALF, InverseMode.REPEAT]:
            for trotter_steps in calib_trotter_step_list:
                if trotter_steps % 2 != 0:
                    continue
                inverse_ = inverse_mode.calc_inverse_list(trotter_steps=trotter_steps)
                trotter_calib_params.append((time, trotter_steps, inverse_))

    if len(trotter_calib_params) > max_num_circuits:
        raise ValueError(f"too many calib patterns {len(trotter_calib_params)} > {max_num_circuits}")
    log.info(f"Total trotter calib circuits: {len(trotter_calib_params)}")
    return trotter_calib_params


def trotter_calibration(
    input_qc: QuantumCircuit,
    trotter_single_step: TrotterSingleStep,
    trotter_qcs: Tuple[TrotterUnitQc, ...],
    target_time: float = np.pi,
    calib_trotter_step_list: List[int] = [2, 4, 8],
    sample_times: List[float] = [0.5, 1.0, 1.5],
    is_inverse_step: List[bool] = [],
    qubit_list: List[int] = [1, 3, 5],
) -> Tuple[List[QuantumCircuit], List[Tuple[float, int, List[bool]]]]:

    gate_calibs: List[QuantumCircuit] = []
    trotter_steps_generator = partial(
        generate_trotter_n_steps_on_qc,
        trotter_single_step=trotter_single_step,
        trotter_qcs=trotter_qcs,
    )
    trotter_calib_params = get_calib_qc_params(
        base_time=target_time,
        sample_times=sample_times,
        calib_trotter_step_list=calib_trotter_step_list,
    )
    for time_, steps_, is_inverse_ in trotter_calib_params:
        qc = trotter_steps_generator(
            initial_qc=copy.deepcopy(input_qc),
            target_time=time_,
            trotter_steps=steps_,
            is_inverse_step=is_inverse_,
        )
        gate_calibs.append(qc)

    log.info(f"Trotter calib qc: add measurement on {qubit_list}")
    gate_calibs = add_measurement(
        input_qcs=gate_calibs,
        meas_qr_inds=qubit_list,
    )
    return gate_calibs, trotter_calib_params


@dataclass
class CsvDataset:
    job_id: List[str] = field(default_factory=list)
    job_completed_time: List[datetime] = field(default_factory=list)
    job_start_time: List[datetime] = field(default_factory=list)
    calib_name: List[CalibName] = field(default_factory=list)
    job_result_time: List[datetime] = field(default_factory=list)
    initial_state: List[str] = field(default_factory=list)
    trotter_steps: List[int] = field(default_factory=list)
    target_time: List[float] = field(default_factory=list)
    single_step_time: List[float] = field(default_factory=list)
    inverse_mode: List[InverseMode] = field(default_factory=list)
    is_single_step_calib: List[bool] = field(default_factory=list)
    operation_order_in_step: List[str] = field(default_factory=list)
    shots: List[int] = field(default_factory=list)
    global_phase: List[float] = field(default_factory=list)
    is_success_result: List[bool] = field(default_factory=list)
    result: List[Dict[str, float]] = field(default_factory=list)

    def __add__(self, other: "CsvDataset") -> "CsvDataset":  # type: ignore [override]
        # add new list
        for attrib_name in self.__annotations__.keys():
            new_value = self.__getattribute__(attrib_name) + other.__getattribute__(attrib_name)
            self.__setattr__(attrib_name, new_value)
        return self

    def parse_result(self, cr_size: int = 3) -> pd.DataFrame:
        res_df = pd.DataFrame(self.result)
        if len(res_df) == 0:
            state_labels = count_keys(num_qubits=cr_size)
            res_df = pd.DataFrame({str_: [] for str_ in state_labels})
            return res_df
        assert max([int(state, base=2) for state in res_df.columns]) < 2**cr_size
        # for zero count states, such state has Null in this df
        res_df = res_df.fillna(value=0.0)

        return res_df

    def get_dataframe(self, cr_size: int = 3) -> pd.DataFrame:
        df = pd.DataFrame(self.__dict__)
        result_df = self.parse_result(cr_size=cr_size)
        df = pd.concat([df, result_df], axis=1)
        return df

    def append_data(self, **kwargs: Any) -> None:
        assert set(kwargs.keys()) == set(
            self.__annotations__.keys()
        ), f"Missing keys {set(self.__annotations__.keys()) - set(kwargs.keys())}"
        for attrib_name in self.__annotations__.keys():
            new_value = self.__getattribute__(attrib_name) + kwargs[attrib_name]
            self.__setattr__(attrib_name, new_value)


def make_dataset_csv(
    calib_job: Job,
    calib_name: CalibName,
    calib_results: Result,
    trotter_calib_params: Optional[List[TrotterCalibParam]],
    is_single_step_calib: List[bool],
    operation_order_in_step: List[str],
    cr_size: int = 3,
    initial_states: List[str] = ["110"],
) -> CsvDataset:

    if trotter_calib_params is None:
        trotter_calib_params = [TrotterCalibParam.empty()] * len(calib_results.results)

    dataset = CsvDataset()
    for exp_idx, exp_result in enumerate(calib_results.results):
        calib_param = trotter_calib_params[exp_idx]
        inverse_mode = InverseMode.find_mode_from_list(inverse_list=calib_param.inverse_list)

        # normalize state label with bin
        result = {}
        for state_name, count in exp_result.data.counts.items():
            bin_format = "{" + f":0>{cr_size}b" + "}"
            if state_name.startswith("0x"):
                state_name = bin_format.format(int(state_name, base=16))

            result[state_name] = count

        dataset.append_data(
            job_id=[calib_job.job_id()],
            job_start_time=[calib_job.time_per_step()["RUNNING"]],
            job_completed_time=[calib_job.time_per_step()["COMPLETED"]],
            calib_name=[calib_name],
            job_result_time=[calib_results.date],
            initial_state=[initial_states[exp_idx]],
            trotter_steps=[calib_param.trotter_steps],
            target_time=[calib_param.time],
            single_step_time=[calib_param.single_step_time],
            inverse_mode=[inverse_mode],
            is_single_step_calib=[is_single_step_calib[exp_idx]],
            operation_order_in_step=[operation_order_in_step[exp_idx]],
            shots=[exp_result.shots],
            global_phase=[exp_result.header.global_phase],
            is_success_result=[exp_result.success],
            result=[result],
        )
    return dataset
