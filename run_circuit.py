import copy
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from qiskit import QuantumCircuit
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.ignis.verification.tomography.data import count_keys
from qiskit.providers import Job
from qiskit.result.result import Result

import src.open_science_prize_requirements as __pickle_patch_for_open_sci_prize
from run_error_mitigation_learning import (
    generate_error_mitigation_filter_with_xgboost,
    prepare_filter_by_xgb_prediction,
)
from src.error_mitigation_learning import get_xgboost_save_path, MAX_NUM_VAL_FOLDS
from src.carib import (
    ADD_3_PLAIN_QC,
    INVERSE_DATA_TAG,
    OBSERVE_STATES_3_QUBITS,
    OPERATION_ORDER_IN_STEP_0_TAG,
    OPERATION_ORDER_IN_STEP_1_TAG,
    SINGLE_CALIB_TAG,
    CalibName,
    CsvDataset,
    TrotterCalibParam,
    make_dataset_csv,
    measurement_caribration,
    trotter_calibration,
)
from src.logger.exp_logger import ExpLog, get_job_result, retrieve_job_from_ibmq, run_exp_logging
from src.open_science_prize_requirements import (
    QCJob,
    monitor_submitted_job_state,
    prepare_initial_state_open_science_prize,
    set_tomography,
    submit_job,
)
from src.run_mode_definitions import (
    BackendName,
    RunMode,
    TargetProcess,
    get_backend,
    update_config_and_set_evaluation_args,
)
from src.trotter_steps import TrotterSingleStep, TrotterUnitQc, generate_trotter_n_steps_on_qc
from src.utils import adjust_qiskit_logging_level, fix_seed

log = logging.getLogger(__name__)

# when load old pickle, need old module name
sys.modules["src.open_science_prize_requiments"] = __pickle_patch_for_open_sci_prize


def request_single_experiment(
    backend_name: BackendName,
    target_process: TargetProcess,
    trotter_steps: int,
    trotter_single_step: TrotterSingleStep,
    trotter_qcs: Tuple[TrotterUnitQc, ...],
    meas_calib_conf: DictConfig,
    calib_trotter_is_inverses: Optional[List[bool]] = None,
    calib_trotter_step_list: List[int] = [2, 4, 8],
    calib_trotter_qubit_list: List[int] = [1, 3, 5],
    calib_sample_times: List[float] = [0.5, 1.0, 1.5],
    optimization_level: int = 3,
    seed_transpiler: int = 42,
    seed_simulator: int = 42,
    job_tags: List[str] = [],
) -> Tuple[List[Job], OrderedDict[CalibName, Job], ExpLog]:
    """generate IBMQ/Aer Job instance

    Parameters
    ----------
    backend_name : BackendName
        backend name which is used at qiskit circuit execution
    target_process :
        specify target process for this function. This flag switch,
        NORMAL -> plain job submission, DATASET_GENERATION -> trotter step error data
    trotter_steps : int,
        the number of trotter steps for the circut
    trotter_single_step : TrotterSingleStep
        specification for a single trotter step which will be repeated "trotter_steps" times at circuit.
    trotter_qcs : Tuple[TrotterUnitQc, ...]
        specification for sub circuit in a single trotter step, usually ((q1, q3), (q3, q5)) .
    carib_conf : DictConfig, optional
        measurement error mitigation arguments for qiskit measfilter
    calib_trotter_is_inverses : Optional[List[bool]], optional
        whether generate circuits for trotter steps error data, by default None
    calib_trotter_step_list : List[int], optional
        the number of the trotter steps for trotter step error data, sweep region is specified here, by default [2, 4, 8]
    calib_trotter_qubit_list : List[int], optional
        qubits for trotter step error data, by default [1, 3, 5]
    calib_sample_times : List[float], optional
        target times for circuits which are used for trotter step error data, by default [0.5, 1.0, 1.5]
    optimization_level : int, optional
        transpile level for qiskit job execution, by default 3
    seed_transpiler : int, optional
        transpile seed for qiskit job execution, by default 42
    seed_simulator : int, optional
        qiskit simulator seed at job execution, by default 42
    job_tags : List[str], optional
        job tags at IBMQ Job, by default []

    Returns
    -------
    jobs :List[Job]
        list of generated IBMQ/Aer jobs
    calib_jobs: OrderedDict[CalibName, Job]
        calibration jobs which may contains measurement_caribration and circuit for trotter step error data
    exp_log: ExpLog
        this job submission log. For retreiving job results later.

    """
    # prepare initial qc state
    jakarta, backend = get_backend(backend_name=backend_name)
    qc, target_time, num_job_repeats = prepare_initial_state_open_science_prize(trotter_steps=trotter_steps)
    calib_job_inputs: OrderedDict[CalibName, List[QuantumCircuit]] = OrderedDict()

    if meas_calib_conf.is_measurement:
        meas_calibs, state_labels = measurement_caribration(
            qr_size=meas_calib_conf.qr_size, qubit_list=meas_calib_conf.qubit_list
        )
        calib_job_inputs[CalibName.MEAS] = meas_calibs
    else:
        state_labels = None

    trotter_calib_params = []
    if target_process == TargetProcess.DATASET_GENERATION:
        if calib_trotter_is_inverses:
            log.info("make calibration qc for trotter_steps")
            # add Hxxx interaction with n trotter_steps
            gate_calibs, trotter_calib_params = trotter_calibration(
                input_qc=qc,
                trotter_single_step=trotter_single_step,
                trotter_qcs=trotter_qcs,
                target_time=target_time,
                calib_trotter_step_list=calib_trotter_step_list,
                sample_times=calib_sample_times,
                is_inverse_step=calib_trotter_is_inverses,
                qubit_list=calib_trotter_qubit_list,
            )
            calib_job_inputs[CalibName.TROTTER_INVERSE] = gate_calibs

    # add Hxxx interaction with n trotter_steps
    qc = generate_trotter_n_steps_on_qc(
        initial_qc=qc,
        trotter_single_step=trotter_single_step,
        trotter_qcs=trotter_qcs,
        trotter_steps=trotter_steps,
        target_time=target_time,
    )

    # add state tomography for evaluating fidelity
    st_qcs = set_tomography(qc)

    # submit qc job
    job_name = str(Path.cwd())
    qc_job = QCJob(
        st_qcs=st_qcs,
        backend=backend,
        num_job_repeats=num_job_repeats,
    )
    jobs, calib_jobs = submit_job(
        qc_job=qc_job,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        seed_simulator=seed_simulator,
        job_name=job_name,
        job_tags=job_tags + [backend_name.name],
        calib_job_inputs=calib_job_inputs,
    )

    return (
        jobs,
        calib_jobs,
        ExpLog(
            backend=jakarta,
            qc_job=qc_job,
            calib_job_ids=[job.job_id() for job in calib_jobs.values()],
            calib_names=[calib_name for calib_name in calib_jobs.keys()],
            meas_state_labels=state_labels,
            meas_method=meas_calib_conf.method,
            job_ids=[job.job_id() for job in jobs],
            job_name=job_name,
            trotter_calib_params=trotter_calib_params,
        ),
    )


def fix_trotter_calib_param_in_exp_log(exp_log: ExpLog, is_single_step_calib: bool, state_labels: List[str]) -> ExpLog:
    if is_single_step_calib:
        # in this mode trotter_calib_param in exp_log does not have enough param
        new_trotter_calib_params: List[TrotterCalibParam] = []
        for trotter_calib_param in exp_log.trotter_calib_params:
            for initial_state in state_labels:
                new_trotter_calib_params.append(trotter_calib_param)
                exp_log.initial_states.append(initial_state)
        exp_log.trotter_calib_params = new_trotter_calib_params
    else:
        for _ in range(len(exp_log.trotter_calib_params)):
            exp_log.initial_states.append("110")
    return exp_log


def process_calib_jobs(
    calib_jobs: OrderedDict[CalibName, Job],
    exp_log: ExpLog,
    target_process: TargetProcess,
    cr_size: int = 3,
    is_single_step_calib: bool = False,
    operation_order_in_step: str = "1-3/3-5",
) -> Tuple[ExpLog, CsvDataset]:
    def _process_meas_filter(calib_results: Result, state_labels: List[str]) -> None:
        meas_fitter = CompleteMeasFitter(
            calib_results,
            state_labels,
            circlabel="mcal",
        )
        exp_log.meas_fitter = meas_fitter

    dataset = CsvDataset()
    for calib_name, calib_job in calib_jobs.items():
        log.info(f"{calib_name} status: {calib_job.status()} ")
        if not calib_job.done():
            continue
        if calib_name == CalibName.MEAS:
            _process_meas_filter(calib_results=calib_job.result(), state_labels=exp_log.meas_state_labels)
        elif calib_name == CalibName.TROTTER_INVERSE:
            if target_process == TargetProcess.NORMAL:
                calib_res = calib_job.result()
            elif target_process == TargetProcess.DATASET_GENERATION:
                exp_log = fix_trotter_calib_param_in_exp_log(
                    exp_log=exp_log, is_single_step_calib=is_single_step_calib, state_labels=OBSERVE_STATES_3_QUBITS
                )
                # result layout, meas_calib_qcs + dataset_qcs + plain_qcs
                end_meas_ind = len(exp_log.meas_state_labels)
                start_plain_qc = len(calib_job.result().results)
                if ADD_3_PLAIN_QC in calib_job.tags():
                    start_plain_qc -= 3
                    exp_log.trotter_calib_params, plain_qc_params = (
                        exp_log.trotter_calib_params[:-3],
                        exp_log.trotter_calib_params[-3:],
                    )
                    exp_log.initial_states, plain_qc_initial_states = (
                        exp_log.initial_states[:-3],
                        exp_log.initial_states[-3:],
                    )

                # for meas filter
                meas_res = copy.deepcopy(calib_job.result())
                meas_res.results = meas_res.results[:end_meas_ind]
                _process_meas_filter(calib_results=meas_res, state_labels=exp_log.meas_state_labels)
                # for main dataset
                calib_res = copy.deepcopy(calib_job.result())
                calib_res.results = calib_res.results[end_meas_ind:start_plain_qc]
                assert len(calib_res.results) == len(exp_log.trotter_calib_params), "wrong results slicing"

                # plain qc for validation
                plain_qc_res = copy.deepcopy(calib_job.result())
                plain_qc_res.results = plain_qc_res.results[start_plain_qc:]

            if exp_log.meas_fitter:
                log.info(f"Applying meas_fitter on {calib_name} result")
                calib_res = exp_log.meas_fitter.filter.apply(calib_res, method=exp_log.meas_method)
                if len(plain_qc_res.results) > 0:
                    log.info(f"Applying meas_fitter on {CalibName.NO_CALIB} result")
                    plain_qc_res = exp_log.meas_fitter.filter.apply(plain_qc_res, method=exp_log.meas_method)

            meas_dataset = make_dataset_csv(
                calib_job=calib_job,
                calib_name=CalibName.MEAS,
                calib_results=meas_res,
                is_single_step_calib=[False] * len(meas_res.results),
                operation_order_in_step=["none"] * len(meas_res.results),
                trotter_calib_params=None,
                initial_states=count_keys(cr_size),
            )
            calib_dataset = make_dataset_csv(
                calib_job=calib_job,
                calib_name=CalibName.TROTTER_INVERSE,
                calib_results=calib_res,
                is_single_step_calib=[is_single_step_calib] * len(calib_res.results),
                operation_order_in_step=[operation_order_in_step] * len(calib_res.results),
                trotter_calib_params=exp_log.trotter_calib_params,
                initial_states=exp_log.initial_states,
            )

            if len(plain_qc_res.results) == 0:
                plain_qc_dataset = CsvDataset()
            else:
                plain_qc_dataset = make_dataset_csv(
                    calib_job=calib_job,
                    calib_name=CalibName.NO_CALIB,
                    calib_results=plain_qc_res,
                    is_single_step_calib=[False] * len(plain_qc_res.results),
                    operation_order_in_step=[OPERATION_ORDER_IN_STEP_0_TAG] * len(plain_qc_res.results),
                    trotter_calib_params=plain_qc_params,
                    initial_states=plain_qc_initial_states,
                )

            dataset += meas_dataset + calib_dataset + plain_qc_dataset

    return exp_log, dataset


def run_calibration_data_generation(
    job_table_path: Path,
    backend_name: BackendName = BackendName.SIM,
    target_process: TargetProcess = TargetProcess.NORMAL,
) -> CsvDataset:

    assert backend_name == backend_name.JAKARTA
    assert target_process == TargetProcess.DATASET_GENERATION

    # job table from ibmq cloud
    job_df = pd.read_csv(job_table_path)
    job_df = job_df.fillna("")

    # extract jobs and load corresponding exp_log for each job
    job_df = job_df.loc[job_df["Status"] == "Completed"]
    tag_pattern = f"{INVERSE_DATA_TAG}*|{SINGLE_CALIB_TAG}*"
    # tag_pattern = f"{INVERSE_DATA_TAG}*"
    # tag_pattern = f"{SINGLE_CALIB_TAG}*"
    # tag_remove_pattern = "^(?!.*SINGLE_CALIB_[1,2]$).*$"
    tag_remove_pattern = f"{SINGLE_CALIB_TAG}[1,2,3]$"

    name_prefix = CalibName.TROTTER_INVERSE.value + "_"
    job_df["target_job"] = job_df["Tags"].apply(lambda x: any([re.search(tag_pattern, tag) for tag in x.split(",")]))
    job_df["remove_job"] = job_df["Tags"].apply(
        lambda x: any([re.search(tag_remove_pattern, tag) for tag in x.split(",")])
    )
    job_df = job_df.loc[job_df.target_job & ~job_df.remove_job].reset_index(drop=True)

    # collect cached dir and tag info
    is_single_step_calib = job_df["Tags"].apply(
        lambda x: any([re.search(SINGLE_CALIB_TAG + "*", tag) for tag in x.split(",")])
    )
    job_df["op_order"] = OPERATION_ORDER_IN_STEP_0_TAG
    op_order_1_mask = job_df["Tags"].apply(
        lambda x: any([re.search(OPERATION_ORDER_IN_STEP_1_TAG, tag) for tag in x.split(",")])
    )
    job_df.loc[op_order_1_mask, "op_order"] = OPERATION_ORDER_IN_STEP_1_TAG
    operation_order_in_step = job_df["op_order"]

    # retreive result from cloud
    exp_log_dirs = job_df["Name"].apply(lambda x: Path(x.replace(name_prefix, ""))).tolist()
    exp_logs: List[ExpLog] = []
    for exp_log_dir in exp_log_dirs:
        exp_logs.append(ExpLog.load_from_pickle(exp_log_dir / "exp_log.pickle"))
    jakarta, _ = get_backend(backend_name=BackendName.JAKARTA)

    # dataset generation
    dataset = CsvDataset()
    for job_index, exp_log_loaded in enumerate(exp_logs):
        calib_jobs_retrieved = retrieve_job_from_ibmq(backend=jakarta, job_ids=exp_log_loaded.calib_job_ids)
        calib_jobs = OrderedDict(zip(exp_log_loaded.calib_names, calib_jobs_retrieved))
        exp_log_loaded, csv_dataset = process_calib_jobs(
            calib_jobs=calib_jobs,
            exp_log=exp_log_loaded,
            target_process=target_process,
            is_single_step_calib=is_single_step_calib[job_index],
            operation_order_in_step=operation_order_in_step[job_index],
        )
        dataset += csv_dataset
    return dataset


def run_evaluation(
    exp_log: ExpLog,
    backend_name: BackendName,
    target_process: TargetProcess,
    xgboost_save_path: Optional[Path] = None,
    jobs: Optional[List[Job]] = None,
    calib_jobs: Optional[OrderedDict[CalibName, Job]] = None,
    meas_fitter_override_skip_filter: bool = False,
    meas_fitter_override_path: Optional[str] = None,
    job_retrieval_ids: Optional[List[str]] = None,
) -> ExpLog:
    """runinit evaluation of the state tomography fidelity for evaluation target specified with exp_log

    Parameters
    ----------
    exp_log: ExpLog
        evaluation target exp_log instance which contains job submission log for retreiving job results.
    backend_name : BackendName
        backend name which was used at evaluation target
    target_process : TargetProcess
        specify target process for this function. This flag switch,
        NORMAL -> plain job submission, DATASET_GENERATION -> trotter step error data
    xgboost_save_path : Optional[Path]
        xgboost_save_path
    jobs : Optional[List[Job]]
        jobs
    calib_jobs : Optional[OrderedDict[CalibName, Job]]
        calib_jobs
    meas_fitter_override_skip_filter : bool
        meas_fitter_override_skip_filter
    meas_fitter_override_path : Optional[str]
        meas_fitter_override_path
    job_retrieval_ids : Optional[List[str]]
        job_retrieval_ids

    Returns
    -------
    ExpLog

    """
    if jobs is None:
        assert backend_name == backend_name.JAKARTA
        # we need approved instance of jakarta to retrieve job from cloud
        jakarta, _ = get_backend(backend_name=BackendName.JAKARTA)
        job_ids = exp_log.job_ids if job_retrieval_ids is None else job_retrieval_ids
        # update job_ids
        exp_log.job_ids = job_ids
        # get main job
        jobs = retrieve_job_from_ibmq(backend=jakarta, job_ids=job_ids)
        monitor_submitted_job_state(jobs=jobs)
        # get carib job
        if len(exp_log.calib_job_ids) > 0:
            calib_jobs_retrieved = retrieve_job_from_ibmq(backend=jakarta, job_ids=exp_log.calib_job_ids)
            monitor_submitted_job_state(jobs=calib_jobs_retrieved)
            calib_jobs = OrderedDict(zip(exp_log.calib_names, calib_jobs_retrieved))
        if calib_jobs:
            exp_log, csv_dataset = process_calib_jobs(
                calib_jobs=calib_jobs, exp_log=exp_log, target_process=target_process
            )

    exp_log.override_meas_filter(path=meas_fitter_override_path, skip_meas_filter=meas_fitter_override_skip_filter)

    if not xgboost_save_path:
        # store the job results
        exp_log = get_job_result(jobs=jobs, exp_log=exp_log, xgb_pred_filters=None)
        return exp_log

    # xgb model error mitigation
    pred_dfs = []
    for val_fold in range(MAX_NUM_VAL_FOLDS):
        xgb_save_path = get_xgboost_save_path(save_dir_path=xgboost_save_path, val_fold=val_fold)
        if xgb_save_path.model.exists() & xgb_save_path.scaler.exists() & xgb_save_path.input_feature.exists():
            _, pred_df = generate_error_mitigation_filter_with_xgboost(
                xgb_save_path=xgb_save_path, exp_log=exp_log, jobs=jobs
            )
            pred_dfs.append(pred_df)
        else:
            break
    if len(pred_dfs) == 0:
        raise ValueError(
            f"check your xgboost_save_path settings, there are no any trained models in {xgboost_save_path}"
        )
    log.info(f"using {len(pred_dfs)} models for generating xgboost filter")
    pred_state_labels = [state + "_pred" for state in OBSERVE_STATES_3_QUBITS]

    # ensemble with different validation fold, this gives +0.03 fidelity improvement
    ensemble_preds = sum([pred_df.loc[:, pred_state_labels] for pred_df in pred_dfs]) / len(pred_dfs)
    ensemble_df = pred_dfs[0]
    ensemble_df.loc[:, pred_state_labels] = ensemble_preds
    xgb_pred_filters = prepare_filter_by_xgb_prediction(plain_qc_df=ensemble_df, pred_state_labels=pred_state_labels)

    # store the job results
    exp_log = get_job_result(jobs=jobs, exp_log=exp_log, xgb_pred_filters=xgb_pred_filters)

    return exp_log


@hydra.main(config_path="./src/config", config_name="config")
def main(conf: DictConfig) -> None:

    fix_seed(seed=conf.seed)
    adjust_qiskit_logging_level(logging_level=logging.WARNING)

    # run experiment with the specified env, which is defined as
    # open_science_prize_requirements

    # parse config
    run_mode = RunMode[conf.run_mode]
    target_process = TargetProcess[conf.target_process]
    backend_name = BackendName[conf.backend_name]

    trotter_single_step = TrotterSingleStep(**conf.trotter.single_step)
    trotter_qcs = (
        TrotterUnitQc(**conf.trotter.unit_qc.qr13),
        TrotterUnitQc(**conf.trotter.unit_qc.qr35),
    )

    # hydra automatically change cwd, so fix relative path to absolute
    hydra_cwd, cwd = Path.cwd(), Path(get_original_cwd())
    xgboost_save_path = cwd / conf.xgboost.load.path if conf.xgboost.load.path is not None else None
    job_table_path = cwd / conf.job_table_path
    exp_log_path = Path(conf.exp_log_path)
    if not exp_log_path.is_absolute():
        # relative path means new job and save under hydra working directory
        exp_log_path = hydra_cwd / conf.exp_log_path

    if run_mode & RunMode.SUBMIT_JOB:
        log.info("Start job submission...")
        jobs, calib_jobs, exp_log_wo_results = request_single_experiment(
            backend_name=backend_name,
            target_process=target_process,
            trotter_steps=conf.trotter.steps,
            trotter_qcs=trotter_qcs,
            trotter_single_step=trotter_single_step,
            meas_calib_conf=conf.carib,
            calib_trotter_is_inverses=conf.calib_trotter.is_inverse_step,
            calib_trotter_step_list=conf.calib_trotter.trotter_steps_list,
            calib_sample_times=conf.calib_trotter.sample_times,
            calib_trotter_qubit_list=conf.calib_trotter.qubit_list,
            optimization_level=conf.transpile.optimization_level,
            seed_transpiler=conf.seed,
            seed_simulator=conf.seed,
            job_tags=conf.neptune_logger.tags,
        )
        exp_log_wo_results.conf_at_request = conf
        # try to store calibration results, if jobs are DONE.
        exp_log_wo_results, _ = process_calib_jobs(
            calib_jobs=calib_jobs, exp_log=exp_log_wo_results, target_process=target_process
        )
        exp_log_wo_results.save_as_pickle(path=exp_log_path)

    if (run_mode == RunMode.ALL) & (backend_name == BackendName.JAKARTA):
        # IBMQ job needs hours or days, so check the state before move onto job evaluation
        monitor_submitted_job_state(jobs=list(calib_jobs.values()) + jobs)

    if run_mode & RunMode.EVALUATE:
        log.info("Start job evaluation...")
        if target_process == TargetProcess.DATASET_GENERATION:
            dataset = run_calibration_data_generation(
                job_table_path=job_table_path, backend_name=backend_name, target_process=target_process
            )
            df = dataset.get_dataframe()
            calib_csv_path = Path("./calib_dataset.csv")
            log.info(f"save csv dataset at {calib_csv_path.absolute()}")
            df.to_csv(calib_csv_path, index=False)

        else:
            exp_log_loaded = ExpLog.load_from_pickle(path=exp_log_path)

            conf, evaluation_kwargs = update_config_and_set_evaluation_args(
                conf_new=conf, conf_old=exp_log_loaded.conf_at_request
            )
            exp_log = run_evaluation(
                exp_log=exp_log_loaded,
                backend_name=backend_name,
                target_process=target_process,
                xgboost_save_path=xgboost_save_path,
                jobs=None if run_mode == RunMode.EVALUATE else jobs,
                calib_jobs=None if run_mode == RunMode.EVALUATE else calib_jobs,
                **evaluation_kwargs,
            )
            # loggin with neptune logger, if no neptune simply ignored
            run_exp_logging(conf=conf, exp_log=exp_log, log_level=log.getEffectiveLevel())


if __name__ == "__main__":
    main()
