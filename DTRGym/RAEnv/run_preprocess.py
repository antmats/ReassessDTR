import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from tianshou.data import Batch, ReplayBuffer
from tqdm import tqdm


def save_to_buffer(output_file_path, data):
    if isinstance(data, list):
        all_inputs, all_groups, all_targets, all_outcomes  = [], [], [], []
        for d in data:
            all_inputs.append(d["inputs"])
            all_groups.append(d["groups"])
            all_targets.append(d["targets"])
            all_outcomes.append(d["outcomes"])
        data = {
            "inputs": pd.concat(all_inputs, ignore_index=True),
            "groups": pd.concat(all_groups, ignore_index=True),
            "targets": np.concatenate(all_targets),
            "outcomes": np.concatenate(all_outcomes),
        }

    inputs = data["inputs"]
    targets = data["targets"]
    groups = data["groups"]
    outcomes = data["outcomes"]

    replay_buffer = ReplayBuffer(len(inputs))

    for id in tqdm(groups.unique(), desc=f"Saving data to buffer {output_file_path}."):
        group_indices = groups[groups == id].index
        group_inputs = inputs.loc[group_indices]
        group_targets = targets[group_indices]
        group_outcomes = outcomes[group_indices]
        max_step = len(group_indices)

        for step in range(max_step):
            obs = group_inputs.iloc[step].to_numpy()
            obs_next = group_inputs.iloc[step + 1].to_numpy() if step + 1 < max_step else obs
            act = group_targets[step]
            rew = group_outcomes[step]

            id_numeric = int(id.replace("-", ""))

            replay_buffer.add(
                Batch(
                    obs=obs,
                    act=act,
                    rew=rew,
                    terminated=(step == max_step - 1),
                    truncated=False,
                    obs_next=obs_next,
                    info={"id": id_numeric, "step": step},
                )
            )

    assert replay_buffer.done.sum() == len(np.unique(replay_buffer.info["id"]))
    assert len(replay_buffer.done.shape) == 1, "replay_buffer.done.shape must be 1"

    replay_buffer.save_hdf5(output_file_path)


if __name__ == "__main__":
    data_dir_path = os.path.join(Path(__file__).parent)
    output_dir_path = os.path.join(data_dir_path, "offline_data")

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    for split in ["train", "val", "test", "ope_train"]:
        if split == "train":
            data = joblib.load(os.path.join(data_dir_path, "train.pkl"))
        elif split == "val":
            data = joblib.load(os.path.join(data_dir_path, "val.pkl"))
        elif split == "test":
            data = joblib.load(os.path.join(data_dir_path, "test.pkl"))
        elif split == "ope_train":
            data = [
                joblib.load(os.path.join(data_dir_path, "train.pkl")),
                joblib.load(os.path.join(data_dir_path, "test.pkl")),
            ]

        output_file_path = os.path.join(output_dir_path, f"all_{split}_buffer.hdf5")
        save_to_buffer(output_file_path, data)
