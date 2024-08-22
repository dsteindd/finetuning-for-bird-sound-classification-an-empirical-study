import json
import os
import sys
from collections import defaultdict

import config
import numpy as np
import torch
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datasets import load_species_ids_species_names_and_targets, load_folds
from torchmetrics import Recall, Precision, F1Score
from tqdm import tqdm


TRANSLATIONS = {
    "Curruca communis": "Sylvia communis",
    "Curruca curruca": "Sylvia curruca",
    "Curruca nisoria" : "Sylvia nisoria",
    "Saxicola rubicola": "Saxicola torquatus"
}

MIN_CONFIDENCE = 0.1

# from https://github.com/python/cpython/issues/60009
class RedirectStdout:
    ''' Create a context manager for redirecting sys.stdout
        to another file.
    '''

    def __init__(self, new_target):
        self.new_target = new_target

    def __enter__(self):
        self.old_target = sys.stdout
        sys.stdout = self.new_target
        return self

    def __exit__(self, exctype, excinst, exctb):
        sys.stdout = self.old_target


def read_species_list_with_ids(file: str):
    species_id2name = dict()
    species_name2id = dict()

    with open(file, "r") as f:

        for row in f:
            species_id, species_name = row.split(" ")
            species_name = species_name.replace("_", " ").strip()
            species_id2name[species_id] = species_name

            species_name2id[species_name] = species_id

    return species_id2name, species_name2id


def detections_to_prob_vector(detections, species_names):
    one_hot = np.zeros(shape=(len(species_names)), dtype=np.float32)

    for detection in detections:
        scientific_name = detection["scientific_name"]

        if scientific_name in TRANSLATIONS:
            scientific_name = TRANSLATIONS[scientific_name]

        if scientific_name in species_names:
            one_hot[species_names.index(scientific_name)] = detection['confidence']

    return one_hot

def species_ids_to_one_hot(targets, species_ids):
    onehot = np.zeros(shape=(len(species_ids)))

    for target in targets:
        if target in species_ids:
            onehot[species_ids.index(target)] = 1

    return onehot


def analyze_folds(
        data_dir: str,
        metadata_path: str,
        species_list: str,
        out_directory: str = "./birdnet-analyzer-results",
        confidence=0.01
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Running on device: ", device)

    test_folds = [0, 1, 2, 3, 4]

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    species_ids, species_names, file_ids_to_targets = load_species_ids_species_names_and_targets(
        path_to_metadata=metadata_path,
        path_to_species_file=species_list
    )

    recall = Recall(task="multilabel", average="macro", num_labels=len(species_ids), threshold=confidence).to(device)
    precision = Precision(task="multilabel", average="macro", num_labels=len(species_ids), threshold=confidence).to(device)
    f1 = F1Score(task="multilabel", average="macro", num_labels=len(species_ids), threshold=confidence).to(device)

    analyzer = Analyzer()

    results_lists = defaultdict(list)

    for test_fold in test_folds:
        print(f"Fold: {test_fold}")

        test_filenames = load_folds(
            fold_directory=config.FOLDS_DIRECTORY,
            folds=[test_fold]
        )

        precision.reset()
        recall.reset()
        f1.reset()

        for filename in tqdm(test_filenames):
            targets = file_ids_to_targets[filename]
            filepath = os.path.join(data_dir, filename)

            recording = Recording(
                analyzer,
                filepath,
                min_conf=0.01,
                return_all_detections=True
            )

            with RedirectStdout(None):
                recording.analyze()

            targets = species_ids_to_one_hot(targets, species_ids)
            predictions = detections_to_prob_vector(recording.detections, species_names)

            targets = torch.from_numpy(targets).reshape((1, -1)).to(device)
            predictions = torch.from_numpy(predictions).reshape((1, -1)).to(device)

            precision.update(predictions, targets.type(torch.int64))
            recall.update(predictions, targets.type(torch.int64))
            f1.update(predictions, targets.type(torch.int64))

        results_lists["precisions"].append(precision.compute().item())
        results_lists["recalls"].append(recall.compute().item())
        results_lists["f1s"].append(f1.compute().item())


    (pmin, pmax, pmean) = (
        np.min(results_lists["precisions"]), np.max(results_lists["precisions"]), np.mean(results_lists["precisions"])
    )

    (rmin, rmax, rmean) = (
        np.min(results_lists["recalls"]), np.max(results_lists["recalls"]), np.mean(results_lists["recalls"])
    )

    (fmin, fmax, fmean) = (
        np.min(results_lists["f1s"]), np.max(results_lists["f1s"]), np.mean(results_lists["f1s"])
    )

    results = {
        "P_min": pmin,
        "P_max": pmax,
        "P_mean": pmean,
        "R_min": rmin,
        "R_max": rmax,
        "R_mean": rmean,
        "F1_min": fmin,
        "F1_max": fmax,
        "F1_mean": fmean,
    }

    out_path = os.path.join(out_directory, f"confidence={confidence}_stats.csv")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    analyze_folds(
        data_dir=config.FILE_PATHS_DOWNSTREAM,
        metadata_path=config.METADATA_PATH_DOWNSTREAM,
        out_directory="./eco2scape-results",
        species_list=config.SPECIES_LIST_DOWNSTREAM,
        confidence=0.01
    )
    analyze_folds(
        data_dir=config.FILE_PATHS_DOWNSTREAM,
        metadata_path=config.METADATA_PATH_DOWNSTREAM,
        out_directory="./eco2scape-results",
        species_list=config.SPECIES_LIST_DOWNSTREAM,
        confidence=0.5
    )
    analyze_folds(
        data_dir=config.FILE_PATHS_DOWNSTREAM,
        metadata_path=config.METADATA_PATH_DOWNSTREAM,
        out_directory="./eco2scape-results",
        species_list=config.SPECIES_LIST_DOWNSTREAM,
        confidence=0.1
    )