import csv
import json
import logging
import os
from collections import defaultdict
import random
import numpy as np

import torch
import torchvision
import torchaudio
from typing import Dict, List, Union
from enum import Enum

from torch.utils.data import Dataset

import config


class TargetEncoding(Enum):
    ONE_HOT = 0
    CATEGORICAL = 1


class NoiseType(Enum):
    WHITE = 0
    PINK = 1
    BROWNIAN = 2
    BLUE = -1
    VIOLET = -2


class RollDimension(Enum):
    TIME = 2
    FREQUENCY = 1


ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_AUGMENTS = [
    "rain",
    "water_drops",
    "wind",
    "pouring_water",
    "sea_waves",
    "thunderstorm",
    "crickets",
    "crackling_fire",
    "insects",
    "frog"
]

ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_ALL = [
    "dog", "rain", "crying_baby", "door_wood_knock", "helicopter",
    "rooster", "sea_waves", "sneezing", "mouse_click", "chainsaw",
    "pig", "crackling_fire", "clapping", "keyboard_typing", "siren",
    "cow", "crickets", "breathing", "door_wood_creaks", "car_horn",
    "frog", "coughing", "can_opening", "engine", "cat", "water_drops", "footsteps", "washing_machine", "train",
    "hen", "wind", "laughing", "vacuum_cleaner", "church_bells",
    "insects", "pouring_water", "brushing_teeth", "clock_alarm", "airplane",
    "sheep", "toilet_flush", "snoring", "clock_tick", "fireworks",
    "crow", "thunderstorm", "drinking_sipping", "glass_breaking", "hand_saw"
]


class NoiseType(Enum):
    WHITE = 0
    PINK = 1
    BROWNIAN = 2
    BLUE = -1
    VIOLET = -2


class RandomExclusiveListApply(torch.nn.Module):
    """
    Applies exactly one of the transformations with the given probablities
    """

    def __init__(self, choice_modules: torch.nn.ModuleList, probabilities: np.ndarray = None):
        super(RandomExclusiveListApply, self).__init__()
        self.choice_modules = choice_modules
        if probabilities:
            self.probabilities = torch.tensor(probabilities / np.sum(probabilities))
        else:
            self.probabilities = torch.tensor(np.ones(len(choice_modules)) / len(choice_modules))

    def forward(self, tensor):
        if len(self.choice_modules) == 0:
            return tensor
        module_index = torch.multinomial(self.probabilities, num_samples=1)
        # module_index = np.random.choice(range(len(self.choice_modules)), p=self.probabilities)
        return self.choice_modules[module_index].forward(tensor)


class ESC50NoiseInjection(torch.nn.Module):
    def __init__(self,
                 directory: str,
                 dst_sr: int,
                 classes: List[str] = None,
                 probabilities: np.ndarray = None
                 ):
        super(ESC50NoiseInjection, self).__init__()
        classes = classes or ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_AUGMENTS

        self.classes = classes
        self.dst_sr = dst_sr
        if probabilities is not None:
            self.probabilities = probabilities / np.sum(probabilities)
        else:
            self.probabilities = np.ones(len(classes)) / len(classes)

        meta_path = os.path.join(directory, "meta", "esc50.csv")
        audio_dir = os.path.join(directory, "audio")

        self.grouped_examples = defaultdict(lambda: [])

        with open(meta_path, "r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                category = row["category"]
                if category in classes:
                    self.grouped_examples[category].append(os.path.join(audio_dir, row["filename"]))

    def forward(self, tensor):
        # choose random noise audio for augmentation
        random_class = np.random.choice(self.classes, p=self.probabilities)
        random_index = random.randint(0, len(self.grouped_examples[random_class]) - 1)
        path_to_noise_audio = self.grouped_examples[random_class][random_index]

        # read audio
        audio, sr = torchaudio.load(path_to_noise_audio)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.dst_sr)

        # pick a random start index
        # tensor (input audio) does have channels
        if tensor.shape[1] < audio.shape[1]:
            # input audio is shorter
            random_index = random.randint(0, audio.shape[1] - tensor.shape[1])
            audio = audio[:, random_index:random_index + tensor.shape[1]]
        elif tensor.shape[1] == audio.shape[1]:
            pass
        else:
            num_repetitions = tensor.shape[1] // audio.shape[1]
            rest = tensor.shape[1] - num_repetitions * audio.shape[1]
            return_audio = torch.empty(tensor.shape[1])
            for j in range(num_repetitions):
                return_audio[j * audio.shape[1]:(j + 1) * audio.shape[1]] = audio
            return_audio[num_repetitions * audio.shape[1]:num_repetitions * audio.shape[1] + rest] = audio[:rest]
            audio = return_audio

        snr = random.uniform(self.min_snr, self.max_snr)
        tensor_squared = tensor.square().sum()
        audio_squared = audio.square().sum()

        if tensor_squared == 0:
            return audio
        if audio_squared == 0:
            # this prevents divide by zero
            # for example in the ESC-50 dataset there are some 1s chunks which contain only zeros
            return tensor
        return tensor + torch.sqrt(10 ** (-snr / 10) * tensor_squared / audio_squared) * audio


class Noise(torch.nn.Module):
    def __init__(self, scale: Union[int, NoiseType] = 0, min_snr_db=3.0, max_snr_db=30.0):
        """
        :param scale: Exponent of noise (default: 0)
        :param min_snr_db: Minimum signal-to-noise ratio in absolute values, i.e. not dB scale (SNR_DB = -10*log(snr))
        :param max_snr_db: Maximum signal-to-noise ratio in absolute values, i.e. not dB scale (SNR_DB = -10*log(snr))
        """
        super().__init__()
        if isinstance(scale, int):
            self.exponent = scale
        elif isinstance(scale, NoiseType):
            self.exponent = scale.value
        self.min_snr = min_snr_db
        self.max_snr = max_snr_db

    def psd(self, f):
        return 1 / np.where(f == 0, float('inf'), np.power(f, self.exponent / 2))

    def forward(self, audio):
        snr = random.uniform(self.min_snr, self.max_snr)

        # check if this is fast enough, or if there are functions from torch
        white_noise = np.fft.rfft(np.random.randn(audio.shape[1]))
        spectral_density = self.psd(np.fft.rfftfreq(audio.shape[1]))
        # Normalize S
        spectral_density = spectral_density / np.sqrt(np.mean(spectral_density ** 2))
        colored_noise = white_noise * spectral_density
        colored_noise = np.fft.irfft(colored_noise)
        colored_noise = torch.tensor(colored_noise, dtype=audio.dtype)
        colored_noise = colored_noise.unsqueeze(0)

        signal_squared_sum = audio.square().sum()
        colored_noise_squared_sum = colored_noise.square().sum()

        # for testing noise signals only
        # otherwise the signal strength is zero, so the noise signal also gets scaled down to 0
        if signal_squared_sum == 0:
            return colored_noise
        return audio + torch.sqrt(10 ** (-snr / 10) * signal_squared_sum / colored_noise_squared_sum) * colored_noise


class RollAndWrap(torch.nn.Module):
    def __init__(self,
                 max_shift: int,
                 dim: Union[int, RollDimension],
                 min_shift: int = None
                 ):
        super(RollAndWrap, self).__init__()
        self.max_shift = max_shift
        if min_shift is None:
            self.min_shift = -max_shift
        else:
            self.min_shift = min_shift

        if isinstance(dim, RollDimension):
            self.dim = dim.value
        else:
            self.dim = dim

    def forward(self, tensor):
        # draw random shift parameter
        shift = random.randint(self.min_shift, self.max_shift)
        return torch.roll(tensor, shifts=shift, dims=self.dim)


class RandomApply(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, p: float = 0.5):
        super(RandomApply, self).__init__()
        self.module = module
        self.p = p

    def forward(self, tensor):
        if self.p == 0.0:
            return tensor

        if torch.rand(1) <= self.p:
            return self.module.forward(tensor)

        return tensor


class AudioRandomBirdDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 file_names: List[str],
                 species_ids: List[int],
                 species_names: List[str],
                 file_ids_to_target_ids: Dict[str, List[int]],
                 num_batches: int = None,
                 batch_size: int = None,
                 nfft: int = 1024,
                 n_mels: int = 128,
                 fmin: float = 300.0,
                 fmax: float = 18000.0,
                 dst_sr: int = 44100,
                 audio_transforms: torch.nn.Module = None,
                 stft_transforms: torch.nn.Module = None,
                 mel_transforms: torch.nn.Module = torch.nn.Sequential(
                     torchvision.transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
                                                   interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                 ),
                 target_encoding: TargetEncoding = TargetEncoding.ONE_HOT,
                 seed: int = 4042,
                 fraction_per_species=1
                 ):
        super(AudioRandomBirdDataset, self).__init__()

        if fraction_per_species < 0 or fraction_per_species > 1:
            raise ValueError(f"fraction_per_species must be between 0 and 1, but was {fraction_per_species}")

        self.data_dir = data_dir

        self.file_names = file_names
        self.species_ids = species_ids
        self.species_names = species_names
        self.file_ids_to_target_ids = file_ids_to_target_ids

        self.audio_transforms = audio_transforms
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft, power=2, hop_length=nfft // 4)

        self.stft_transforms = stft_transforms
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.MelScale(
                n_mels=n_mels, sample_rate=dst_sr, n_stft=nfft // 2 + 1, f_min=fmin, f_max=fmax
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
        self.mel_transforms = mel_transforms

        self.dst_sr = dst_sr

        self.number_of_elements = 0
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.target_encoding = target_encoding

        self.grouped_examples = defaultdict(list)

        self.samples_per_species = fraction_per_species

        for file in file_names:

            file_species_ids = file_ids_to_target_ids[file]

            self.number_of_elements += 1

            for file_species_id in file_species_ids:
                self.grouped_examples[file_species_id].append(file)

        if fraction_per_species != 1.0:
            for species_id in self.grouped_examples.keys():
                files = self.grouped_examples[species_id]
                np.random.seed(seed)
                files = np.random.choice(sorted(files), int(fraction_per_species * len(files)), False)
                self.grouped_examples[species_id] = files

        for sp_id in self.species_ids:
            if (sp_id not in self.grouped_examples.keys()) or (len(self.grouped_examples[sp_id]) == 0):
                logging.warning(f"No data for species {species_ids}.")

    def audio_to_spec(self, audio, sr):
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.dst_sr)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        spec = self.stft(audio)
        if self.stft_transforms:
            spec = self.stft_transforms(spec)

        mel = self.mel(spec)
        if self.mel_transforms:
            mel = self.mel_transforms(mel)

        return mel

    def __len__(self):
        if self.batch_size is not None and self.num_batches is not None:
            return self.batch_size * self.num_batches

        return self.number_of_elements

    def one_hot(self, targets):
        one_hot = torch.zeros(len(self.species_ids))
        for (i, species_id) in enumerate(self.species_ids):
            if species_id in targets:
                one_hot[i] = 1

        return one_hot

    def __getitem__(self, index):
        species_index = random.randint(0, len(self.species_ids) - 1)
        species_id = self.species_ids[species_index]

        # redraw if no samples are there for this particular species
        while len(self.grouped_examples[species_id]) == 0:
            species_index = random.randint(0, len(self.species_ids) - 1)
            species_id = self.species_ids[species_index]

        random_index = random.randint(0, len(self.grouped_examples[species_id]) - 1)
        filename = self.grouped_examples[species_id][random_index]

        audio, sr1 = torchaudio.load(os.path.join(self.data_dir, filename))
        spec = self.audio_to_spec(audio, sr1)

        targets = self.file_ids_to_target_ids[filename]

        if self.target_encoding == TargetEncoding.ONE_HOT:
            target = self.one_hot(targets)
        elif self.target_encoding == TargetEncoding.CATEGORICAL:
            target = targets
        else:
            raise ValueError(f"Target encoding {self.target_encoding} not known")

        return spec, target


class AudioDeterministicBirdDataset(Dataset):
    def __init__(self,
                 data_dir,
                 file_names: List[str],
                 species_ids: List[int],
                 species_names: List[str],
                 file_ids_to_target_ids: Dict[str, List[int]],
                 nfft: int = 1024,
                 n_mels: int = 128,
                 fmin: float = 300.0,
                 fmax: float = 18000.0,
                 dst_sr: int = 44100,
                 audio_transforms: torch.nn.Module = None,
                 stft_transforms: torch.nn.Module = None,
                 mel_transforms: torch.nn.Module = torch.nn.Sequential(
                     torchvision.transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
                                                   interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                 ),
                 target_encoding: TargetEncoding = TargetEncoding.ONE_HOT,
                 return_file_ids: bool = False
                 ):
        super(AudioDeterministicBirdDataset, self).__init__()

        self.data_dir = data_dir

        self.return_file_ids = return_file_ids

        self.species_ids = species_ids
        self.species_names = species_names

        self.grouped_examples = {}
        self.audio_transforms = audio_transforms
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft, power=2, hop_length=nfft // 4)

        self.stft_transforms = stft_transforms
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.MelScale(
                n_mels=n_mels, sample_rate=dst_sr, n_stft=nfft // 2 + 1, f_min=fmin, f_max=fmax
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
        self.mel_transforms = mel_transforms

        self.dst_sr = dst_sr

        self.target_encoding = target_encoding

        self.file_ids_to_target_ids = file_ids_to_target_ids

        self.files = file_names

        self.number_of_elements = len(self.files)

    def one_hot(self, targets):
        one_hot = torch.zeros(len(self.species_ids))
        for (i, species_id) in enumerate(self.species_ids):
            if species_id in targets:
                one_hot[i] = 1

        return one_hot

    def audio_to_spec(self, audio, sr):
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.dst_sr)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        spec = self.stft(audio)
        if self.stft_transforms:
            spec = self.stft_transforms(spec)

        mel = self.mel(spec)
        if self.mel_transforms:
            mel = self.mel_transforms(mel)

        return mel

    def __len__(self):
        return self.number_of_elements

    def __getitem__(self, index):
        filename = self.files[index]

        audio, sr1 = torchaudio.load(os.path.join(self.data_dir, filename))
        spec = self.audio_to_spec(audio, sr1)
        if self.target_encoding == TargetEncoding.ONE_HOT:
            target = self.one_hot(self.file_ids_to_target_ids[filename])
        elif self.target_encoding == TargetEncoding.CATEGORICAL:
            target = self.file_ids_to_target_ids[filename]
        else:
            raise ValueError(f"Target Encoding {self.target_encoding} not known")

        if self.return_file_ids:
            return spec, target, filename
        else:
            return spec, target


def _load_txt_file(file):
    with open(file, "r") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip()) != 0]

    return lines


def _load_species_id_species_names(path: str):
    species_ids = []
    species_names = []

    with open(path, "r") as f:
        for line in f.readlines():
            if len(line) == 0:
                continue

            species_id = int(line.split(" ")[0])
            species_name = line.split(" ")[1].replace("_", " ").strip()

            species_ids.append(species_id)
            species_names.append(species_name)

    return species_ids, species_names


def _load_file_ids_to_target_ids(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    return data


def load_species_ids_species_names_and_targets(
        path_to_species_file: str = "./species-list-finetune.txt",
        path_to_metadata: str = "./metadata-eco2scape-data.json"
):
    species_ids, species_names = _load_species_id_species_names(path_to_species_file)

    targets = _load_file_ids_to_target_ids(path_to_metadata)

    return species_ids, species_names, targets


def load_folds(
        folds: List[int],
        fold_directory: str = "./"
):
    file_names = []

    for fold in folds:
        file_names.extend(_load_txt_file(os.path.join(fold_directory, f"fold-{fold}.txt")))

    return file_names
