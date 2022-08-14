import copy
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from hydra import compose, initialize_config_dir
from qiskit.ignis.mitigation.measurement.fitters import MeasurementFilter
from qiskit.providers import Job
from qiskit.providers.aer.jobs.aerjob import AerJob
from sklearn.preprocessing import StandardScaler

from src.carib import OBSERVE_STATES_3_QUBITS, CalibName
from src.error_mitigation_learning import (
    XgboostSavePath,
    DATASET_META,
    GATE_NAMES_IN_JAKARTA,
    GATE_PROPERTY_NAMES,
    QUBIT_PROPERTY_NAMES,
    filter_features,
    get_xgboost_save_path,
    initial_state_augmentation,
    make_col_aligned_dataframe_groupby,
    make_split,
    parse_backend_properties_as_dataframe,
    preprocess_dataframe_dataset,
    set_local_tz_from_strings,
    visualize_properties,
)
from src.logger.exp_logger import ExpLog
from src.open_science_prize_requirements import TARGET_TIME
from src.run_mode_definitions import BackendName, get_backend
from src.utils import fix_seed

REMOVE_CALIB_NAMES = [CalibName.NO_CALIB, CalibName.MEAS]


def back_to_probability(
    df: pd.DataFrame,
    state_labels: List[str],
    state_suffix: str = "_pred",
) -> pd.DataFrame:
    for state in state_labels:
        initial_state_mask = state.replace(state_suffix, "") == df["initial_state"]
        # initial_state should be 1.0
        df.loc[initial_state_mask, state] = 1.0 - df.loc[initial_state_mask, state]
        # not initial_state should be 0.0
        df.loc[~initial_state_mask, state] = 0.0 - df.loc[~initial_state_mask, state]
    return df


def _load_from_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj


def prepare_filter_by_xgb_prediction(
    plain_qc_df: pd.DataFrame, pred_state_labels: List[str]
) -> List[MeasurementFilter]:
    xgb_filters = []
    for original_index, target_df in plain_qc_df.groupby("original_index"):
        # get pred_cal_mat
        pred_cal_mat = target_df.loc[:, pred_state_labels].to_numpy()
        assert target_df["initial_state"].to_list() == list(OBSERVE_STATES_3_QUBITS)
        num_normalizations = 100
        for i in range(num_normalizations):
            pred_cal_mat = pred_cal_mat / pred_cal_mat.sum(axis=0).reshape(1, -1)
            pred_cal_mat = pred_cal_mat / pred_cal_mat.sum(axis=1).reshape(-1, 1)

        meas_filter = MeasurementFilter(cal_matrix=pred_cal_mat, state_labels=list(OBSERVE_STATES_3_QUBITS))
        xgb_filters.append(meas_filter)
    return xgb_filters


def generate_error_mitigation_filter_with_xgboost(
    xgb_save_path: XgboostSavePath, exp_log: ExpLog, jobs: List[Job]
) -> Tuple[List[MeasurementFilter], pd.DataFrame]:
    # xgb model loading
    # xgb_save_path = get_xgboost_save_path(save_dir_path=xgboost_save_path, val_fold=4)
    reg = xgb.XGBRegressor()
    reg.load_model(xgb_save_path.model)
    with xgb_save_path.scaler.open("rb") as f:
        scaler = pickle.load(f)
    with xgb_save_path.input_feature.open("r") as f:
        input_feature_names = json.load(f)

    # preprocess for xgb model
    jakarta, _ = get_backend(backend_name=BackendName.JAKARTA)
    jakarta_props = []
    job_datetimes = []
    for job in jobs:
        if isinstance(job, AerJob):
            # simply get prop from exp_log
            jakarta_prop = exp_log.backend._properties
            job_datetime = exp_log.backend._properties.last_update_date
        else:
            # get prop from IBMQ with the datetime of job
            jakarta_prop = jakarta.properties(refresh=True, datetime=job.time_per_step()["RUNNING"])
            job_datetime = job.time_per_step()["RUNNING"]

        jakarta_props.append(jakarta_prop)
        job_datetimes.append(job_datetime)

    qubit_df, gate_df = parse_backend_properties_as_dataframe(jakarta_props=jakarta_props)
    qubit_df = make_col_aligned_dataframe_groupby(row_aligned_df=qubit_df, key="qubit")
    gate_df = make_col_aligned_dataframe_groupby(row_aligned_df=gate_df, key="gate_name")
    # drop features
    gate_df, qubit_df = filter_features(gate_df=gate_df, qubit_df=qubit_df, mode="full")
    calib_df = pd.merge(left=qubit_df, right=gate_df, on=["sample_index", "last_update_date"])

    # csv dataset area
    trotter_steps = exp_log.conf_at_request.trotter.steps
    single_step_time = TARGET_TIME / trotter_steps
    dataset_df = []
    for job_datetime, job in zip(job_datetimes, jobs):
        dataset_df.append(
            {
                "job_start_time": job_datetime,
                "trotter_steps": trotter_steps,
                "single_step_time": single_step_time,
                "initial_state": "110",
                "calib_name": CalibName.NO_CALIB,
                "initial_state_q5": 1,
                "initial_state_q3": 1,
                "initial_state_q1": 0,
            }
        )
    dataset_df = pd.DataFrame(dataset_df)

    merged_df = pd.merge_asof(
        left=dataset_df,
        right=calib_df,
        left_on="job_start_time",
        right_on="last_update_date",
        direction="backward",
    )
    plain_qc_df = merged_df.loc[merged_df["calib_name"] == CalibName.NO_CALIB].reset_index(drop=True)
    plain_qc_df = initial_state_augmentation(df=plain_qc_df, state_labels=OBSERVE_STATES_3_QUBITS)
    # additional validation
    X_plain = plain_qc_df.loc[:, input_feature_names].to_numpy()
    X_plain = scaler.transform(X_plain)
    pred_plain = reg.predict(X_plain)

    pred_state_labels = [state + "_pred" for state in OBSERVE_STATES_3_QUBITS]
    pred_df = pd.DataFrame(pred_plain.astype(np.float64), columns=pred_state_labels)
    pred_df = pd.concat([plain_qc_df, pred_df], axis=1)
    # xgb prediction is not probability, so cast to probability
    pred_df = back_to_probability(df=pred_df, state_labels=pred_state_labels)

    xgb_pred_filters = prepare_filter_by_xgb_prediction(plain_qc_df=pred_df, pred_state_labels=pred_state_labels)
    return xgb_pred_filters, pred_df


def main() -> None:
    CONFIG_DIR = Path(Path.cwd(), "src", "config")
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        conf = compose("config_xgboost.yaml")

    # parse config
    gate_path = Path(conf.cached_data.backend_properties.gate_path)
    qubit_path = Path(conf.cached_data.backend_properties.qubit_path)
    cache_dataset_path = Path(conf.cached_data.dataset_csv_path)
    save_dir_path = Path(conf.save_dir_path)
    save_dir_path.mkdir(exist_ok=True)

    fix_seed(seed=conf.seed)

    if gate_path.exists() & qubit_path.exists():
        gate_df = pd.read_csv(gate_path)
        qubit_df = pd.read_csv(qubit_path)
        gate_df = set_local_tz_from_strings(df=gate_df, target_column="last_update_date")
        qubit_df = set_local_tz_from_strings(df=qubit_df, target_column="last_update_date")
    else:
        backend_cached_pickle_dir = Path(conf.cached_data.backend_pickle_dir)
        cached_dirs = list(backend_cached_pickle_dir.iterdir())
        cached_dirs = sorted(cached_dirs, key=lambda x: datetime.fromisoformat(x.name))
        jakarta_props = [_load_from_pickle(timestamped_dir / "jakarta_prop.pickle") for timestamped_dir in cached_dirs]
        qubit_df, gate_df = parse_backend_properties_as_dataframe(jakarta_props=jakarta_props)

        # col -> row conversion
        qubit_df = make_col_aligned_dataframe_groupby(row_aligned_df=qubit_df, key="qubit")

        gate_df = make_col_aligned_dataframe_groupby(row_aligned_df=gate_df, key="gate_name")

    if conf.visualize_properties:
        # output some images on a screen
        visualize_properties(
            col_aligned_df=copy.deepcopy(qubit_df),
            property_names=list(QUBIT_PROPERTY_NAMES),
        )
        gate_property_names = [
            [gate_prop + ".*_" + gate_name for gate_name in GATE_NAMES_IN_JAKARTA] for gate_prop in GATE_PROPERTY_NAMES
        ]
        gate_property_names = gate_property_names[0] + gate_property_names[1]

        visualize_properties(col_aligned_df=copy.deepcopy(gate_df), property_names=gate_property_names)

    # drop features
    gate_df, qubit_df = filter_features(gate_df=gate_df, qubit_df=qubit_df, mode="minimum")
    calib_df = pd.merge(left=qubit_df, right=gate_df, on=["sample_index", "last_update_date"])
    calib_columns = calib_df.iloc[:, 2:].columns.to_list()

    # cached dataset contains counts after CompleteMeasFilter applied
    dataset_df = pd.read_csv(cache_dataset_path)
    dataset_df = preprocess_dataframe_dataset(df=dataset_df, state_labels=OBSERVE_STATES_3_QUBITS)
    # merge with backend_properties data and counts data
    merged_df = pd.merge_asof(
        left=dataset_df, right=calib_df, left_on="job_start_time", right_on="last_update_date", direction="backward"
    )
    assert np.all(merged_df["job_start_time"] > merged_df["last_update_date"])

    # get only plain qc dataframe as additional validation data
    plain_qc_df = merged_df.loc[merged_df["calib_name"] == CalibName.NO_CALIB].reset_index(drop=True)
    # initial state augmentation for 8 x 8 matirc prediction
    plain_qc_df = initial_state_augmentation(df=plain_qc_df, state_labels=OBSERVE_STATES_3_QUBITS)

    for remove_calib_name in REMOVE_CALIB_NAMES:
        print(f"remove samples from train/val set: {remove_calib_name}")
        mask = merged_df["calib_name"] != remove_calib_name
        merged_df = merged_df.loc[mask]

    merged_df = make_split(df=merged_df, n_splits=conf.n_splits, target_key="job_id", group_key="job_id", how="group")

    # Start n_splits cross validation
    for val_fold in range(conf.n_splits):
        # train/val split
        train_df = merged_df.loc[merged_df["fold"] != val_fold, :]
        val_df = merged_df.loc[merged_df["fold"] == val_fold, :]
        # generate input/label data for model
        X_train = train_df.loc[:, DATASET_META + calib_columns].to_numpy()
        y_train = train_df.loc[:, OBSERVE_STATES_3_QUBITS].to_numpy()
        X_val = val_df.loc[:, DATASET_META + calib_columns].to_numpy()
        y_val = val_df.loc[:, OBSERVE_STATES_3_QUBITS].to_numpy()
        # scale input data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        # Train a regressor on it
        reg = xgb.XGBRegressor(tree_method="hist", n_estimators=conf.xgboost.n_estimators)
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])

        # additional validation
        X_plain = plain_qc_df.loc[:, DATASET_META + calib_columns].to_numpy()
        y_plain = plain_qc_df.loc[:, OBSERVE_STATES_3_QUBITS].to_numpy()
        X_plain = scaler.transform(X_plain)
        pred_plain = reg.predict(X_plain)
        loss = np.sqrt(((pred_plain - y_plain) ** 2).sum(axis=-1))

        pred_df = pd.DataFrame(
            pred_plain.astype(np.float64), columns=[state + "_pred" for state in OBSERVE_STATES_3_QUBITS]
        )
        pred_df = pd.concat([plain_qc_df, pred_df], axis=1)

        # save each objects, model, prediction and input configuration...
        xgb_save_path = get_xgboost_save_path(save_dir_path=save_dir_path, val_fold=val_fold)
        # Save model into JSON format.
        # Json format contains both model structure and training hyperparameter
        reg.save_model(xgb_save_path.model)
        with xgb_save_path.scaler.open("wb") as f:
            pickle.dump(scaler, f)
        with xgb_save_path.input_feature.open("w") as f:
            json.dump(DATASET_META + calib_columns, f)
        pred_df.to_csv(xgb_save_path.pred_df, index=False)


if __name__ == "__main__":
    main()
