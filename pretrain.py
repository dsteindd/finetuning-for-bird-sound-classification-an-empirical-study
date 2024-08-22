import os
import os
import sys
from argparse import ArgumentParser
from csv import DictWriter
from datetime import datetime

import config
import torch
import torchaudio
import torchvision
from datasets import load_species_ids_species_names_and_targets, AudioRandomBirdDataset, RandomApply, \
    RandomExclusiveListApply, ESC50NoiseInjection, NoiseType, Noise, RollAndWrap, RollDimension
from models import ClassificationNetworkType, build_classification_network
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score
from tqdm import tqdm
from utils import _make_deterministic


def pretrain_experiments(
        classification_networks,
        augment_p,
        data_dir=config.METADATA_PATH_PRETRAIN,
        metadata_path=config.METADATA_PATH_PRETRAIN,
        species_list_path=config.SPECIES_LIST_PRETRAIN,
        esc_dir=config.ESC50_DIR,
        lr=1e-4,
        base_models_dir="./models-pretrain",
        optimizer="adam"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 4242

    num_batches = 400
    batch_size = 64
    epochs = 1000

    num_workers = os.cpu_count() - 1
    cuda_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'shuffle': False
    }

    species_ids, species_names, file_ids_to_targets = load_species_ids_species_names_and_targets(
        path_to_metadata=metadata_path,
        path_to_species_file=species_list_path
    )

    train_filenames = [f for f in file_ids_to_targets.keys()]

    dataset = AudioRandomBirdDataset(
        data_dir=data_dir,
        file_names=train_filenames,
        species_ids=species_ids,
        species_names=species_names,
        file_ids_to_target_ids=file_ids_to_targets,
        fraction_per_species=1,
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
            torchvision.transforms.Normalize(config.PRETRAIN_STATS[0],
                                             config.PRETRAIN_STATS[1])
        )
    )

    for classification_network_type in classification_networks:
        _make_deterministic(seed)

        model_dir = os.path.join(base_models_dir, f"augment-{augment_p}/{classification_network_type.value}")

        model = build_classification_network(
            network_type=classification_network_type,
            num_classes=len(species_ids),
            is_multilabel=True
        ).to(device)
        print(model)
        optimizer = AdamW(model.parameters(), lr=lr) if optimizer == "adam" else SGD(model.parameters(), lr=lr, momentum=0.9)

        model_kwargs = {
            "network_type": classification_network_type,
            "num_classes": len(species_ids),
            "is_multilabel": True
        }

        criterion = BCEWithLogitsLoss()

        precision = Precision(task="multilabel", num_labels=len(species_ids), average="macro").to(device)
        recall = Recall(task="multilabel", num_labels=len(species_ids), average="macro").to(device)
        f1 = F1Score(task="multilabel", num_labels=len(species_ids), average="macro").to(device)

        model_name = f"seed-{seed}"

        dataloader = DataLoader(dataset, batch_size=batch_size, **cuda_kwargs)

        current_model_dir = os.path.join(model_dir, model_name)

        model_path = os.path.join(current_model_dir, "latest_model.h5")

        if not os.path.exists(current_model_dir):
            os.makedirs(current_model_dir)

        if any([f.startswith("latest_model") for f in os.listdir(current_model_dir)]):
            print(f"Model at {model_path} already exists. Skipping.")
            continue

        history_path = os.path.join(current_model_dir, "history.csv")

        with open(history_path, "w") as f:
            writer = DictWriter(f, fieldnames=["epoch", "train_loss", "train_precision", "train_recall",
                                               "train_f1", "duration"])
            writer.writeheader()

        for epoch in range(1, epochs + 1):

            start = datetime.now()

            loss = 0.0
            sample_count = 0

            model.train()

            pbar = tqdm(dataloader, desc=f"Training {epoch}/{epochs}", colour='yellow', file=sys.stdout)

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

            with open(history_path, "a") as f:
                csv_writer = DictWriter(f, fieldnames=["epoch", "train_loss", "train_precision",
                                                       "train_recall", "train_f1", "duration"])
                csv_writer.writerow({
                    "epoch": epoch,
                    "train_loss": loss,
                    "train_precision": precision.compute().item(),
                    "train_recall": recall.compute().item(),
                    "train_f1": f1.compute().item(),
                    "duration": duration.total_seconds()
                })

            precision.reset()
            recall.reset()
            f1.reset()

            if epoch % 100 == 0:
                torch.save([model_kwargs, model.state_dict()], model_path.replace(".h5", f"-{epoch}.h5"))

        torch.save([model_kwargs, model.state_dict()], model_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--networks',
        type=ClassificationNetworkType,
        default=[
            ClassificationNetworkType.RESNET18,
            ClassificationNetworkType.RESNET34,
            ClassificationNetworkType.RESNET50,
            ClassificationNetworkType.BIRD_NET,
            ClassificationNetworkType.WIDE_RESNET50,
            ClassificationNetworkType.EFFICIENT_NET_S
        ],
        help="Which networks to train",
        nargs='+'
    )
    parser.add_argument(
        '--augment',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--data-dir',
        default=config.PRETRAIN_FILE_PATHS
    )
    parser.add_argument(
        '--metadata-path',
        default=config.METADATA_PATH_PRETRAIN,
    )
    parser.add_argument(
        '--species-list',
        default=config.SPECIES_LIST_PRETRAIN
    )
    parser.add_argument(
        '--esc-dir',
        default=config.ESC50_DIR
    )
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=float
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models-pretrain'
    )
    parser.add_argument(
        '--optimizer',
        default="adam",
        choices=["adam", "sgd"],
        type=str
    )

    args = parser.parse_args()

    pretrain_experiments(
        classification_networks=args.networks,
        augment_p=args.augment,
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        species_list_path=args.species_list,
        esc_dir=args.esc_dir,
        lr=args.lr,
        base_models_dir=args.models_dir,
        optimizer=args.optimizer
    )
