from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from qiskit.providers.backend import BackendV1
from qiskit.providers.ibmq.utils.converters import utc_to_local
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from src.carib import CalibName

QUBIT_PROPERTY_NAMES = (
    "T1",
    "T2",
    "frequency",
    "anharmonicity",
    "readout_error",
    "prob_meas0_prep1",
    "prob_meas1_prep0",
    "readout_length",
)

MAX_NUM_VAL_FOLDS = 10
DATASET_META = ["trotter_steps", "single_step_time", "initial_state_q5", "initial_state_q3", "initial_state_q1"]
GATE_PROPERTY_NAMES = ("gate_error", "gate_length")
GATE_NAMES_IN_JAKARTA = ("cx", "id", "reset", "rz", "sx", "x")


def extract_properties_as_dataframe(prop_dict: dict, target_property_key: str = "qubits") -> pd.DataFrame:
    if target_property_key == "qubits":
        prop_dfs = []
        for qubit_idx, qubit_prop in enumerate(prop_dict[target_property_key]):
            df = pd.DataFrame(qubit_prop)
            df["qubit"] = "q" + str(qubit_idx)
            prop_dfs.append(df)

        prop_df = pd.concat(prop_dfs, axis=0)
        prop_df = prop_df.sort_values(["qubit", "name"])
    elif target_property_key == "gates":
        # There is an additional nest in this case, so flatten
        prop_dfs = []
        for gate_prop in prop_dict[target_property_key]:
            gate = gate_prop["gate"]
            gate_name = gate_prop["name"]
            for param in gate_prop["parameters"]:
                param.update({"gate": gate, "gate_name": gate_name})
                prop_dfs.append(param)
        prop_df = pd.DataFrame(prop_dfs)
        prop_df["last_update_date"] = prop_dict["last_update_date"]
        # sometimes order of cx data is changed so we use uniform order instead
        prop_df = prop_df.sort_values(["gate_name", "name"])

    prop_df["last_update_date"] = prop_dict["last_update_date"]
    return prop_df


def make_col_aligned_dataframe_groupby(row_aligned_df: pd.DataFrame, key: str = "qubit") -> pd.DataFrame:
    # add unique feature name
    unit = np.where(row_aligned_df["unit"] == "", "", "_" + row_aligned_df["unit"])
    row_aligned_df["feat_name"] = row_aligned_df["name"] + unit + "_" + row_aligned_df[key]
    # Making one sample time has features as a dataframe row
    feats_per_sample = []
    for feat_name, target_df in row_aligned_df.groupby("feat_name"):
        df = pd.DataFrame(
            {
                "sample_index": target_df["sample_index"],
                "last_update_date": target_df["last_update_date"],
                feat_name: target_df["value"],
            }
        )
        feats_per_sample.append(df)

    # merge
    base_df = feats_per_sample.pop(0)
    for qubit_index, df in enumerate(feats_per_sample, 1):
        base_df = pd.merge(
            left=base_df,
            right=df,
            on=["sample_index", "last_update_date"],
            how="outer",
        )
    assert np.all(base_df.notna())
    return base_df


def parse_backend_properties_as_dataframe(
    jakarta_props: List[BackendV1],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # cast to local timezone with qiskit converter to compare time stamp at IBMQ
    last_update_date = pytz.utc.localize(datetime(2000, 1, 1))
    last_update_date = utc_to_local(last_update_date)

    qubit_dfs = []
    gate_dfs = []
    sample_index = 0
    for jakarta_prop in jakarta_props:
        prop_dict = jakarta_prop.to_dict()
        if last_update_date >= prop_dict["last_update_date"]:
            continue

        qubit_df = extract_properties_as_dataframe(prop_dict=prop_dict, target_property_key="qubits")
        gate_df = extract_properties_as_dataframe(prop_dict=prop_dict, target_property_key="gates")
        qubit_df["sample_index"] = sample_index
        gate_df["sample_index"] = sample_index
        qubit_dfs.append(qubit_df)
        gate_dfs.append(gate_df)
        # update for next sample
        last_update_date = qubit_df["last_update_date"].iloc[0]
        sample_index += 1

    qubit_df = pd.concat(qubit_dfs, axis=0)
    gate_df = pd.concat(gate_dfs, axis=0)

    return qubit_df, gate_df


def visualize_properties(col_aligned_df: pd.DataFrame, property_names: List[str]) -> None:
    col_aligned_df.iloc[:, 2:] = col_aligned_df.iloc[:, 2:] - np.median(col_aligned_df.iloc[:, 2:], axis=0)
    for prop_name in property_names:
        filtered = col_aligned_df.filter(regex=f"{prop_name}.*")
        if filtered.shape[1] == 0:
            print(f"skip visualization for {prop_name}.*")
            continue
        fig, ax = plt.subplots()
        filtered.plot(ax=ax)
        plt.show()
        plt.close()


def filter_features(
    gate_df: pd.DataFrame,
    qubit_df: pd.DataFrame,
    mode: str = "max",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # remove constant feature
    qubit_df = qubit_df.filter(regex="^(?!.*readout_length).*$")
    qubit_df = qubit_df.filter(regex="^(?!.*anharmonicity).*$")

    gate_df = gate_df.filter(regex="^(?!.*gate_length).*$")
    gate_df = gate_df.filter(regex="^(?!.*_rz\d).*$")

    # remove unused qubit/gate feature, [0, 2, 4, 6] qubits
    qubit_df = qubit_df.filter(regex="^(?!.*readout_error_q[0,2,4,6]).*$")
    qubit_df = qubit_df.filter(regex="^(?!.*prob_meas0_prep1_q[0,2,4,6]).*$")
    qubit_df = qubit_df.filter(regex="^(?!.*prob_meas1_prep0_q[0,2,4,6]).*$")
    qubit_df = qubit_df.filter(regex="^(?!.*T2_us_q[0,2,4,6]).*$")

    # x, sx, id gate errors are all the same, so we use x only
    gate_df = gate_df.filter(regex="^(?!.*_x[0,2,4,6]).*$")
    gate_df = gate_df.filter(regex="^(?!.*_sx\d).*$")
    gate_df = gate_df.filter(regex="^(?!.*_id\d).*$")
    gate_df = gate_df.filter(regex="^(?!.*_reset[0,2,4,6]).*$")

    unused_cnots = [
        "gate_error_cx0_1",
        "gate_error_cx1_0",
        "gate_error_cx1_2",
        "gate_error_cx2_1",
        "gate_error_cx4_5",
        "gate_error_cx5_4",
        "gate_error_cx5_6",
        "gate_error_cx6_5",
    ]
    gate_feat_names = gate_df.columns.to_list()
    gate_feat_names = [name for name in gate_feat_names if name not in unused_cnots]
    gate_df = gate_df.loc[:, gate_feat_names]

    if mode == "minimum":
        print("use minimum features for calibration data")
        # remove readout/meas, because we used counts after meas complete filter
        qubit_df = qubit_df.filter(regex="^(?!.*readout_error_q\d).*$")
        qubit_df = qubit_df.filter(regex="^(?!.*prob_meas0_prep1_q\d).*$")
        qubit_df = qubit_df.filter(regex="^(?!.*prob_meas1_prep0_q\d).*$")
        qubit_df = qubit_df.filter(regex="^(?!.*T2_us_q\d).*$")
        qubit_df = qubit_df.filter(regex="^(?!.*T1_us_q[0,2,4,6]).*$")
        qubit_df = qubit_df.filter(regex="^(?!.*frequency_GHz_q[0,2,4,6]).*$")

    return gate_df, qubit_df


def set_local_tz_from_strings(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    # convert IBMQ default tz settings for merge key
    time_series = pd.to_datetime(df[target_column], utc=True)
    df[target_column] = pd.Series(time_series.apply(utc_to_local))
    return df


def preprocess_dataframe_dataset(
    df: pd.DataFrame,
    state_labels: List[str],
    epsilon: float = 1.0e-9,
    remove_calib_names: Optional[List[str]] = None,
) -> pd.DataFrame:

    cr_size = len(state_labels[0])
    # use only success result
    df = df.loc[df["is_success_result"]]

    df = set_local_tz_from_strings(df=df, target_column="job_start_time")
    df = df.sort_values(["job_id", "job_start_time"])

    # normalize state expression
    df["initial_state"] = df["initial_state"].astype(str).str.zfill(cr_size)
    df["initial_state_q5"] = df["initial_state"].apply(lambda x: int(x[0]))
    df["initial_state_q3"] = df["initial_state"].apply(lambda x: int(x[1]))
    df["initial_state_q1"] = df["initial_state"].apply(lambda x: int(x[2]))

    # parse CalibName
    df["calib_name"] = df["calib_name"].apply(lambda x: CalibName[x.split(".")[1]])

    # use probability instead of count
    shots = df["shots"].to_numpy()[:, np.newaxis]
    df.loc[:, state_labels] = df.loc[:, state_labels] / shots

    # convert in label
    for state in state_labels:
        initial_state_mask = state == df["initial_state"]
        # initial_state should be 1.0
        df.loc[initial_state_mask, state] = 1.0 - df.loc[initial_state_mask, state]
        # not initial_state should be 0.0
        df.loc[~initial_state_mask, state] = 0.0 - df.loc[~initial_state_mask, state]
    assert np.abs(df.loc[:, state_labels].sum(axis=1)).max() < epsilon / shots[0]

    return df


def make_split(
    df: pd.DataFrame,
    n_splits: int = 3,
    target_key: str = "target",
    group_key: Optional[str] = None,
    is_reset_index: bool = True,
    verbose: int = 0,
    shuffle: bool = True,
    how: str = "stratified",
) -> pd.DataFrame:

    if shuffle:
        df = df.sample(frac=1.0)

    if is_reset_index:
        df.reset_index(drop=True, inplace=True)
    df["fold"] = -1

    split_keys = {"X": df, "y": df[target_key]}
    if how == "stratified":
        cv = StratifiedKFold(n_splits=n_splits)
    elif how == "group":
        assert group_key is not None
        cv = GroupKFold(n_splits=n_splits)
        split_keys.update({"groups": df[group_key]})
    elif how == "stratified_group":
        assert group_key is not None
        cv = StratifiedGroupKFold(n_splits=n_splits)
        split_keys.update({"groups": df[group_key]})
    else:
        raise ValueError(f"how: {how}")

    for i, (train_idx, valid_idx) in enumerate(cv.split(**split_keys)):
        df.loc[valid_idx, "fold"] = i
    if verbose == 1:
        print(">> check split with target\n", pd.crosstab(df.fold, df[target_key]))
        if group_key is not None:
            print(">> check split with group\n", pd.crosstab(df.fold, df[group_key]))

    return df


def initial_state_augmentation(df: pd.DataFrame, state_labels: List[str]) -> pd.DataFrame:

    cr_size = len(state_labels[0])
    if df.calib_name.unique().tolist() != [CalibName.NO_CALIB]:
        raise ValueError(f"Only {CalibName.NO_CALIB} samples are expected")

    augmented_dfs = []
    for state_label in state_labels:
        aug_df = df.copy(deep=True)
        aug_df["is_original"] = aug_df["initial_state"] == state_label
        aug_df["initial_state"] = state_label
        aug_df["initial_state"] = aug_df["initial_state"].astype(str).str.zfill(cr_size)
        aug_df["initial_state_q5"] = aug_df["initial_state"].apply(lambda x: int(x[0]))
        aug_df["initial_state_q3"] = aug_df["initial_state"].apply(lambda x: int(x[1]))
        aug_df["initial_state_q1"] = aug_df["initial_state"].apply(lambda x: int(x[2]))
        aug_df = aug_df.reset_index(drop=False)
        aug_df = aug_df.rename(columns={"index": "original_index"})

        augmented_dfs.append(aug_df)

    augmented_dfs = pd.concat(augmented_dfs, axis=0)
    augmented_dfs = augmented_dfs.sort_values(
        [
            "original_index",
            "job_start_time",
            "initial_state_q5",
            "initial_state_q3",
            "initial_state_q1",
            "trotter_steps",
        ]
    ).reset_index(drop=True)
    return augmented_dfs


class XgboostSavePath(NamedTuple):
    model: Path
    scaler: Path
    input_feature: Path
    pred_df: Path


def get_xgboost_save_path(save_dir_path: Path, val_fold: int) -> XgboostSavePath:
    postfix = f"val_fold_{val_fold}"
    return XgboostSavePath(
        model=save_dir_path / f"error_mitigation_regressor_{postfix}.json",
        scaler=save_dir_path / f"scaler_{postfix}.pickle",
        input_feature=save_dir_path / f"input_feature_{postfix}.json",
        pred_df=save_dir_path / f"pred_df_{postfix}.csv",
    )
