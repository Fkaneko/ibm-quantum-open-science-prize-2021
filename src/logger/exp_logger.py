import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.ignis.mitigation.measurement.fitters import MeasurementFilter
from qiskit.providers import Job
from qiskit.providers.backend import BackendV1
from qiskit.result.result import Result
from tensorboardX import SummaryWriter

from src.carib import CalibName, TrotterCalibParam
from src.logger.from_pl.logger import _flatten_dict, _sanitize_params
from src.open_science_prize_requirements import QCJob, monitor_submitted_job_state, output_state_tomo

log = logging.getLogger(__name__)


# some jobs are resubmitted on IBMQ to regenerate calibration or qc parameters
JOB_ID_RESUBMISSION_DICT = {}


@dataclass
class ExpLog:
    """Storing experiment log/results

    Attributes
    ----------
    fids: List[float]
        |110> fidelities, target metric
    backend: BackendV1
        current backend instance which includes calibration info
    qc_job: QCJob
        inputs for job, which produce the result
    job_ids: List[str]
        job ids for ibmq cloud
    job_name: Optional[str]
        job name which is used ibmq cloud
    conf_at_request: DictConfig
        config at when this ExpLog was generated
    job_results: List[Result]
        job result with qiskit result class
    meas_fitter: CompleteMeasFitter
        measurement caribration
    """

    backend: BackendV1
    qc_job: QCJob
    meas_method: str
    job_ids: List[str]
    job_name: Optional[str] = None
    calib_job_ids: List[str] = field(default_factory=list)
    calib_names: List[CalibName] = field(default_factory=list)
    meas_state_labels: List[str] = field(default_factory=list)
    # these values below are stored after qc job completion
    fids: List[float] = field(default_factory=list)
    job_results: List[Result] = field(default_factory=list)
    meas_fitter: CompleteMeasFitter = field(default=None)
    trotter_calib_params: List[TrotterCalibParam] = field(default_factory=list)
    initial_states: List[str] = field(default_factory=list)

    conf_at_request: DictConfig = field(default_factory=OmegaConf.create)

    def __eq__(self, other: "ExpLog") -> bool:  # type: ignore [override]

        meas_flag = True
        if (self.meas_fitter is not None) & (other.meas_fitter is not None):
            meas_flag = bool(np.all(self.meas_fitter.cal_matrix == other.meas_fitter.cal_matrix))
        flag = (
            (self.backend.name() == other.backend.name())
            # & (self.qc_job == other.qc_job)
            & meas_flag
            & (self.meas_method == other.meas_method)
            & (self.job_ids == other.job_ids)
            & (self.job_name == other.job_name)
            & (self.fids == other.fids)
            & (self.job_results == other.job_results)
            & (self.conf_at_request == other.conf_at_request)
        )
        return flag

    def save_as_pickle(self, path: Path) -> None:
        log.info(f"saving experiment log on {path.absolute()}")
        if path.exists():
            new_path = path.with_name("exp_log_" + datetime.now().isoformat() + ".pickle")
            log.info(f"{path.absolute()}, exists, so append timestamp, {new_path.absolute()}")

        with path.open("wb") as f:
            pickle.dump(self, f)

    def __convert_calib_params_in_namedtuple(self) -> None:
        if len(self.trotter_calib_params) == 0:
            return
        elif isinstance(self.trotter_calib_params[0], TrotterCalibParam):
            return
        self.trotter_calib_params = [TrotterCalibParam(*param) for param in self.trotter_calib_params]

    @staticmethod
    def load_from_pickle(path: Path) -> "ExpLog":
        log.info(f"loading experiment log from {path.absolute()}")
        with path.open("rb") as f:
            cls = pickle.load(f)
        # for newly added attributes after loaded ExpLog pickle was saved
        # we generate a new instance
        exp_log = ExpLog(**cls.__dict__)
        exp_log.__convert_calib_params_in_namedtuple()
        return exp_log

    def override_meas_filter(self, path: Optional[str], skip_meas_filter: bool = False) -> None:
        if skip_meas_filter:
            log.info("Skipping meas_fitter processing")
            self.meas_fitter = None
            self.meas_method = "none"
        else:
            if path:
                log.info("Overriding meas_fitter of experiment log ...")
                for_meas_calc = ExpLog.load_from_pickle(path=Path(path))
                self.meas_fitter = for_meas_calc.meas_fitter
                self.meas_method = for_meas_calc.meas_method
                log.info(f"caribration matrix, \n {self.meas_fitter.cal_matrix}")


def initialize_neptune_logger(
    project: Optional[str] = "your/project",
    name: str = "optional_name",
    mode: str = "debug",
    custom_run_id: Optional[str] = None,
    tags: List[Optional[str]] = [None],
) -> Any:
    """
    initialize neptune ai logger
    """
    try:
        import neptune.new as neptune
    except Exception:
        if project is not None:
            log.warning(
                f"""
                Neptune project:"{project}" is assumed but Neptune is not installed,
                So setting project=None here """
            )
        project = None

    if project is None:
        log.info("Any Neptune project is not specified, so Neptune is not activated")
        return None

    # base neptune settings
    run = neptune.init(
        project=project,
        name=name,  # Optional
        mode=mode,
        custom_run_id=custom_run_id,
    )
    if tags[0] is not None:
        run["sys/tags"].add(list(tags))
    return run


def run_tensorboard_logging(conf: DictConfig, exp_log: ExpLog) -> None:
    metric = {}
    with SummaryWriter() as w:
        log.info(f"saving experiment result on {str(Path.cwd())}/{w.logdir}")
        # convert DictConfig into tensorboard hparams format with PL utilities
        params = OmegaConf.to_container(conf)
        params = _flatten_dict(params)  # type: ignore[arg-type]
        params = _sanitize_params(params)
        params["env/dir"] = str(Path.cwd())
        params["job/id"] = str(exp_log.job_ids)
        params["job/name"] = exp_log.job_name
        metric["fidelity/average"] = np.mean(exp_log.fids)
        metric["fidelity/std"] = np.std(exp_log.fids)
        w.add_hparams(params, metric)
        for exp_idx, fid in enumerate(exp_log.fids):
            w.add_scalars(
                "fidelities",
                {"fidelity": fid},
                exp_idx,
            )


def run_exp_logging(
    conf: DictConfig,
    exp_log: ExpLog,
    custom_run_id: Optional[str] = None,
    tag_with_first_job_id: bool = True,
    log_level: int = logging.INFO,
) -> None:
    """Starting neptune ai logging.

    If neptune is not installed tensorboard logging will be used

    Parameters
    ----------
    conf : DictConfig
        config for this experiment
    exp_results : ExpResults
        experiment results

    Returns
    -------
    None

    """

    run = initialize_neptune_logger(
        project=conf.neptune_logger.project,
        name=conf.neptune_logger.name,
        # mode="debug" if conf.is_debug else "async",
        mode="debug" if log_level == logging.DEBUG else "async",
        custom_run_id=custom_run_id,
        tags=conf.neptune_logger.tags,
    )
    if run is None:
        # saveing with tensorboardX
        run_tensorboard_logging(conf=conf, exp_log=exp_log)
        return None

    from neptune.new.types import File

    # logging each value
    run["params/conf"] = conf
    run["env/dir"] = str(Path.cwd())

    # job meta data
    run["sys/tags"].add([conf.backend_name])
    if tag_with_first_job_id:
        run["sys/tags"].add([exp_log.job_ids[0]])
    run["job/num_job_repeats"] = exp_log.qc_job.num_job_repeats
    run["job/backend_jakarta"].upload(File.as_pickle(exp_log.backend))
    run["job/results"].upload(File.as_pickle(exp_log.job_results))
    run["job/ids"] = exp_log.job_ids
    # run["prediction_example"].upload(File.as_image(numpy_array))

    # fidelity related
    for fid in exp_log.fids:
        run["fidelity/all"].log(fid)

    run["fidelity/average"] = np.mean(exp_log.fids)
    run["fidelity/std"] = np.std(exp_log.fids)

    run.stop()
    return None


def get_job_result(
    jobs: List[Job],
    exp_log: ExpLog,
    xgb_pred_filters: Optional[List[MeasurementFilter]] = None,
) -> ExpLog:
    # check qc results with fidelity over num_job_repeats
    fids, job_results = output_state_tomo(
        jobs=jobs,
        st_qcs=exp_log.qc_job.st_qcs,
        meas_fitter=exp_log.meas_fitter,
        meas_method=exp_log.meas_method,
        xgb_pred_filters=xgb_pred_filters,
    )
    exp_log.fids = fids
    exp_log.job_results = job_results
    return exp_log


def retrieve_job_from_ibmq(backend: BackendV1, job_ids: List[str]) -> List[Job]:
    # job retrieval
    jobs: List[Job] = []
    for job_id in job_ids:
        resub_job_id = JOB_ID_RESUBMISSION_DICT.get(job_id, None)
        if resub_job_id:
            log.info(f"Use resubmission job id:{resub_job_id} instead of {job_id}")
            job_id = resub_job_id
        job = backend.retrieve_job(job_id=job_id)
        jobs.append(job)
        log.info(f"retrieve job: {job_id}")
    return jobs
