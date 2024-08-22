import os.path
import sys
from argparse import ArgumentParser
from csv import DictWriter
from datetime import datetime

import torch.cuda
import torchaudio
import torchvision
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score
from tqdm import tqdm

import config
from datasets import AudioRandomBirdDataset, \
    RollDimension, RandomApply, RollAndWrap, Noise, NoiseType, ESC50NoiseInjection, \
    RandomExclusiveListApply, load_species_ids_species_names_and_targets, load_folds
from early_stopper import EarlyStopping
from models import build_classification_network, ClassificationNetworkType
from utils import _make_deterministic


def downstream_experiments(
        classification_network_type=None,
        augment_p=0.0,
        data_dir=config.FILE_PATHS_DOWNSTREAM,
        metadata_path=config.METADATA_PATH_DOWNSTREAM,
        species_list_path=config.SPECIES_LIST_DOWNSTREAM,
        folds_directory=config.FOLDS_DIRECTORY,
        esc_dir=config.ESC50_DIR,
        pretrain_path: str = None,
        model_dir: str = "./models-multilabel"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 4242

    num_batches = 400
    batch_size = 64
    epochs = 1000
    lr = 1e-4

    patience = 5

    num_workers = len(os.sched_getaffinity(0))
    cuda_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'shuffle': False
    }

    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    species_ids, species_names, file_ids_to_targets = load_species_ids_species_names_and_targets(
        path_to_metadata=metadata_path,
        path_to_species_file=species_list_path
    )

    for test_fold in range(5):

        _make_deterministic(seed)

        tf_model_dir = os.path.join(model_dir, f"augment-{augment_p}/{classification_network_type.value}/fold-{test_fold}")

        filenames = load_folds(
            fold_directory=folds_directory,
            folds=[fold for fold in range(5) if fold != test_fold]
        )

        model_kwargs = {
            "network_type": classification_network_type,
            "num_classes": len(species_ids),
            "is_multilabel": True
        }

        criterion = BCEWithLogitsLoss()

        precision = Precision(task="multilabel", num_labels=len(species_ids), average="macro").to(device)
        recall = Recall(task="multilabel", num_labels=len(species_ids), average="macro").to(device)
        f1 = F1Score(task="multilabel", num_labels=len(species_ids), average="macro").to(device)

        for fraction in fractions:
            model_name = f"fraction-{fraction}-seed-{seed}"

            current_model_dir = os.path.join(tf_model_dir, model_name)

            model_path = os.path.join(current_model_dir, "latest_model.h5")

            if os.path.exists(model_path):
                print(f"Model at {model_path} already exists. Skipping.")
                continue

            model = build_classification_network(
                network_type=classification_network_type,
                num_classes=len(species_ids),
                is_multilabel=True,
                pretrained_path=pretrain_path
            ).to(device)

            optimizer = AdamW(model.parameters(), lr=lr)

            f1_early_stopping = EarlyStopping(patience=patience, mode="max")

            dataset = AudioRandomBirdDataset(
                data_dir=data_dir,
                file_names=filenames,
                species_ids=species_ids,
                species_names=species_names,
                file_ids_to_target_ids=file_ids_to_targets,
                fraction_per_species=fraction,
                seed=seed,
                num_batches=num_batches,
                batch_size=batch_size,
                audio_transforms=torch.nn.Sequential(
                    RandomApply(
                        module=RandomExclusiveListApply(
                            choice_modules=torch.nn.ModuleList([
                                ESC50NoiseInjection(
                                    dst_sr=44100,
                                    directory=esc_dir
                                ),
                                Noise(scale=NoiseType.WHITE, min_snr_db=0.0, max_snr_db=30.0),
                                Noise(scale=NoiseType.PINK, min_snr_db=0.0, max_snr_db=30.0)
                            ]),
                        ),
                        p=augment_p
                    ),
                ),
                mel_transforms=torch.nn.Sequential(
                    RandomApply(
                        module=torch.nn.Sequential(
                            torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
                            torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
                        ),
                        p=augment_p
                    ),
                    RandomApply(
                        module=RollAndWrap(max_shift=5, min_shift=-5, dim=RollDimension.FREQUENCY),
                        p=augment_p
                    ),
                    RandomApply(
                        module=RollAndWrap(max_shift=int(44100 / (256) * 0.4),
                                           min_shift=int(-44100 / (256) * 0.4),
                                           dim=RollDimension.TIME),
                        p=augment_p
                    ),
                    torchvision.transforms.Resize(
                        (128, 384),
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                    ),
                    torchvision.transforms.Normalize(config.FOLD_STATS[test_fold][0],
                                                     config.FOLD_STATS[test_fold][1])
                )
            )

            dataloader = DataLoader(dataset, batch_size=batch_size, **cuda_kwargs)

            if not os.path.exists(current_model_dir):
                os.makedirs(current_model_dir)

            history_path = os.path.join(current_model_dir, "history.csv")

            with open(history_path, "w") as f:
                writer = DictWriter(f, fieldnames=["epoch", "loss", "precision", "recall", "f1", "duration"])
                writer.writeheader()

            for epoch in range(1, epochs + 1):

                start = datetime.now()

                loss = 0.0
                sample_count = 0

                pbar = tqdm(dataloader, desc=f"Training {epoch}/{epochs}", colour='yellow', file=sys.stdout)

                model.train()
                for i, (images, targets) in enumerate(pbar):
                    sample_count += images.size(0)
                    images, targets = images.to(device), targets.to(device)
                    optimizer.zero_grad()

                    predicted = model(images)

                    loss = criterion(predicted, targets)
                    loss += loss.item() * images.size(0)
                    loss.backward()
                    optimizer.step()

                    precision.update(predicted, targets)
                    recall.update(predicted, targets)
                    f1.update(predicted, targets)

                    pbar.set_postfix({
                        'loss': loss / sample_count,
                        'precision': precision.compute().item(),
                        'recall': recall.compute().item(),
                        'f1': f1.compute().item()
                    })

                duration = datetime.now() - start

                loss /= sample_count

                f1_early_stopping.update(f1.compute().item())

                with open(history_path, "a") as f:
                    csv_writer = DictWriter(f, fieldnames=["epoch", "train_loss", "train_precision",
                                                           "train_recall", "train_f1", "val_loss",
                                                           "val_precision", "val_recall", "val_f1", "duration"])
                    csv_writer.writerow({
                        "epoch": epoch,
                        "loss": loss,
                        "precision": precision.compute().item(),
                        "recall": recall.compute().item(),
                        "f1": f1.compute().item(),
                        "duration": duration.total_seconds()
                    })

                if f1_early_stopping.stop_criterion_reached():
                    print(f"Neither Train nor Test F1 improved for {f1_early_stopping.patience} epochs")
                    break

                precision.reset()
                recall.reset()
                f1.reset()

            torch.save([model_kwargs, model.state_dict()], model_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--network',
        type=ClassificationNetworkType,
        default=ClassificationNetworkType.RESNET18,
        help="Which networks to train"
    )
    parser.add_argument(
        '--augment',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--data-dir',
        default=config.FILE_PATHS_DOWNSTREAM
    )
    parser.add_argument(
        '--metadata-path',
        default=config.METADATA_PATH_DOWNSTREAM,
    )
    parser.add_argument(
        '--species-list',
        default=config.SPECIES_LIST_DOWNSTREAM
    )
    parser.add_argument(
        '--esc-dir',
        default=config.ESC50_DIR
    )
    parser.add_argument(
        '--pretrain-path',
        default=None
    )
    parser.add_argument(
        '--model-dir',
        default="./models-downstream/from_pretrain"
    )

    args = parser.parse_args()

    downstream_experiments(
        classification_network_type=args.network,
        augment_p=args.augment,
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        species_list_path=args.species_list,
        esc_dir=args.esc_dir,
        pretrain_path=args.pretrain_path,
        model_dir=args.model_dir
    )
