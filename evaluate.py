import os.path
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictWriter

import config
import numpy as np
import torch
import torchvision
import tqdm
from datasets import load_folds, AudioDeterministicBirdDataset, load_species_ids_species_names_and_targets
from models import build_classification_network, ClassificationNetworkType
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score


def load_model(path: str, device: str, classification_type: ClassificationNetworkType = None):
    kwargs, state_dict = torch.load(path)

    if classification_type is not None:
        kwargs["network_type"] = classification_type

    network = build_classification_network(**kwargs)
    network.load_state_dict(state_dict)

    return network.to(device)


def test_models(
        network,
        base_model_dir,
        augment_p,
        metadata_path,
        species_list,
        data_dir,
        batch_size
):
    test_folds = [0, 1, 2, 3, 4]

    fractions_of_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    species_ids, species_names, file_ids_to_targets = load_species_ids_species_names_and_targets(
        path_to_metadata=metadata_path,
        path_to_species_file=species_list
    )

    num_workers = len(os.sched_getaffinity(0))

    cuda_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'shuffle': False
    }

    seed = 4242

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Running on ", "CUDA" if device == "cuda" else "CPU")

    results = []
    results_per_species = defaultdict(list)


    out_directory = os.path.join(base_model_dir, f"augment-{augment_p}/{network.value}/", "results")

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    out_path = os.path.join(out_directory, "stats.csv")

    if os.path.isfile(out_path):
        return

    for fraction_of_samples in fractions_of_samples:

        precisions = []
        recalls = []
        f1s = []

        class_wise_precisions = defaultdict(list)
        class_wise_recalls = defaultdict(list)
        class_wise_f1s = defaultdict(list)

        for test_fold in test_folds:
            test_filenames = load_folds(
                fold_directory=config.FOLDS_DIRECTORY,
                folds=[test_fold]
            )

            test_dataset = AudioDeterministicBirdDataset(
                data_dir=data_dir,
                file_names=test_filenames,
                species_ids=species_ids,
                species_names=species_names,
                file_ids_to_target_ids=file_ids_to_targets,
                mel_transforms=torch.nn.Sequential(
                    torchvision.transforms.Resize(
                        (128, 384),
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                    ),
                    torchvision.transforms.Normalize(config.FOLD_STATS[test_fold][0],
                                                     config.FOLD_STATS[test_fold][1])
                )
            )

            dataloader = DataLoader(test_dataset, batch_size=batch_size, **cuda_kwargs)

            model_dir = os.path.join(base_model_dir, f"augment-{augment_p}/{network.value}/fold-{test_fold}")

            model_name = f"fraction-{fraction_of_samples}-seed-{seed}"

            model = load_model(
                os.path.join(model_dir, model_name, "latest_model.h5"),
                device,
                classification_type=network
            )

            model.eval()

            precision = Precision(task="multilabel", num_labels=len(species_ids), average="macro").to(device)
            recall = Recall(task="multilabel", num_labels=len(species_ids), average="macro").to(device)
            f1 = F1Score(task="multilabel", num_labels=len(species_ids), average="macro").to(device)

            class_wise_precision = Precision(task="multilabel", num_labels=len(species_ids), average="none").to(device)
            class_wise_recall = Recall(task="multilabel", num_labels=len(species_ids), average="none").to(device)
            class_wise_f1 = F1Score(task="multilabel", num_labels=len(species_ids), average="none").to(device)

            with torch.no_grad():
                for i, (images, targets) in enumerate(
                        tqdm.tqdm(dataloader, desc=f"Evaluating model {network.value} (p={augment_p})")
                ):
                    images, targets = images.to(device), targets.to(device)

                    predicted = model(images)

                    predicted = torch.sigmoid(predicted)

                    precision.update(predicted, targets)
                    recall.update(predicted, targets)
                    f1.update(predicted, targets)

                    class_wise_precision.update(predicted, targets)
                    class_wise_recall.update(predicted, targets)
                    class_wise_f1.update(predicted, targets)


            precisions.append(precision.compute().item())
            recalls.append(recall.compute().item())
            f1s.append(f1.compute().item())

            for index in range(len(species_ids)):
                species_id = species_ids[index]

                class_wise_precisions[species_id].append(class_wise_precision.compute()[index].item())
                class_wise_recalls[species_id].append(class_wise_recall.compute()[index].item())
                class_wise_f1s[species_id].append(class_wise_f1.compute()[index].item())

        p_mean, p_minimum, p_maximum = (
            np.mean(precisions),
            np.min(precisions),
            np.max(precisions)
        )

        r_mean, r_minimum, r_maximum = (
            np.mean(recalls),
            np.min(recalls),
            np.max(recalls)
        )

        f1_mean, f1_minimum, f1_maximum = (
            np.mean(f1s),
            np.min(f1s),
            np.max(f1s)
        )

        results.append({
            "fraction_of_samples": fraction_of_samples,
            "P_mean": p_mean,
            "P_min": p_minimum,
            "P_max": p_maximum,
            "R_mean": r_mean,
            "R_min": r_minimum,
            "R_max": r_maximum,
            "F1_mean": f1_mean,
            "F1_min": f1_minimum,
            "F1_max": f1_maximum
        })

        for index in range(len(species_ids)):
            species_id = species_ids[index]
            ps = class_wise_precisions[species_id]
            rs = class_wise_recalls[species_id]
            fs = class_wise_f1s[species_id]

            p_mean, p_minimum, p_maximum = (
                np.mean(ps),
                np.min(ps),
                np.max(ps)
            )

            r_mean, r_minimum, r_maximum = (
                np.mean(rs),
                np.min(rs),
                np.max(rs)
            )

            f1_mean, f1_minimum, f1_maximum = (
                np.mean(fs),
                np.min(fs),
                np.max(fs)
            )

            results_per_species[species_id].append({
                "fraction_of_samples": fraction_of_samples,
                "P_mean": p_mean,
                "P_min": p_minimum,
                "P_max": p_maximum,
                "R_mean": r_mean,
                "R_min": r_minimum,
                "R_max": r_maximum,
                "F1_mean": f1_mean,
                "F1_min": f1_minimum,
                "F1_max": f1_maximum
            })



    if os.path.isfile(out_path):
        print(f"Already evaluated. Delete file at {out_path} if you want to re-evaluate.")
    else:
        with open(out_path, "w") as f:
            writer = DictWriter(f, fieldnames=["fraction_of_samples", "P_mean", "P_min", "P_max", "R_mean", "R_min", "R_max", "F1_mean", "F1_min", "F1_max"])
            writer.writeheader()
            writer.writerows(results)

    for index in range(len(species_ids)):
        species_id = species_ids[index]

        out_path = os.path.join(out_directory, f"stats_{species_id}.csv")

        with open(out_path, "w") as f:
            writer = DictWriter(f, fieldnames=["fraction_of_samples", "P_mean", "P_min", "P_max", "R_mean", "R_min",
                                               "R_max", "F1_mean", "F1_min", "F1_max"])
            writer.writeheader()
            writer.writerows(results_per_species[species_id])


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--data-dir',
        default=config.FILE_PATHS_DOWNSTREAM
    )
    parser.add_argument(
        '--model-dir',
        default="./models-multilabel"
    )
    parser.add_argument(
        '--networks',
        default=[
            ClassificationNetworkType.RESNET18,
            ClassificationNetworkType.RESNET34,
            ClassificationNetworkType.RESNET50,
            ClassificationNetworkType.BIRD_NET,
            ClassificationNetworkType.WIDE_RESNET50,
            ClassificationNetworkType.EFFICIENT_NET_S
        ],
        type=ClassificationNetworkType,
        nargs='+'
    )
    parser.add_argument(
        '--species-list',
        default=config.SPECIES_LIST_DOWNSTREAM
    )
    parser.add_argument(
        '--metadata-path',
        default=config.METADATA_PATH_DOWNSTREAM
    )
    parser.add_argument(
        '--augment',
        default=0.0,
        type=float
    )
    parser.add_argument(
        '--batch-size',
        default=256,
        type=int
    )
    args = parser.parse_args()

    for network in args.networks:
        test_models(
            network=network,
            base_model_dir=args.model_dir,
            augment_p=args.augment,
            metadata_path=args.metadata_path,
            species_list=args.species_list,
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
