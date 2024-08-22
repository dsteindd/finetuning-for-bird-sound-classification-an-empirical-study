import json
import os.path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

import config


def stratified_split():
    with open(config.METADATA_PATH_DOWNSTREAM, "r") as f:
        data = json.load(f)

    species_ids = []
    species_names = []

    with open("data/species-list-finetune.txt", "r") as f:
        for line in f:
            species_ids.append(int(line.split(" ")[0]))
            species_names.append(line.split(" ")[1].replace("_", " "))

    X = [k for k in data.keys()]
    y = [v for v in data.values()]

    mlb = MultiLabelBinarizer()
    mlb.fit([species_ids])

    y = mlb.transform(y)

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    splits = mskf.split(X, y)

    train_folds = []
    test_folds = []

    for i, (train_index, test_index) in enumerate(splits):
        train_folds.append(train_index)
        test_folds.append(test_index)

    # Sanity Checks
    for i in range(len(train_folds)):
        print(f"Fold {i}")
        print(f"Num Training Samples: {len(train_folds[i])}")
        print(f"Num Test Samples: {len(test_folds[i])}")

        assert(len(set(train_folds[i]).intersection(test_folds[i])) == 0)

    for i in range(len(test_folds)):
        for j in range(len(test_folds)):
            if i == j:
                continue
            assert(len(set(test_folds[i]).intersection(test_folds[j])) == 0)

    # Write fold information to file
    for i in range(len(test_folds)):
        fold_idxs = [X[idx] + '\n' for idx in test_folds[i]]
        with open(os.path.join(config.FOLDS_DIRECTORY, f"fold-{i}.txt"), "w") as f:
            f.writelines(fold_idxs)


if __name__ == "__main__":
    stratified_split()