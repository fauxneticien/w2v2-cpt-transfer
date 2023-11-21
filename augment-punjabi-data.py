# Install augment library from https://github.com/facebookresearch/WavAugment
import augment
import numpy as np
import os
import random
import torchaudio
import torch

from glob import glob
from pathlib import Path
from tqdm import tqdm

class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir):
        self.sample_rate = sample_rate

        r = np.exp(np.random.uniform(10, 15) * np.log(10) / 10)
        self.coeff = r / (1.0 + r)

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(glob(noise_dir + '/noise/*/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ['gain', '-n'], # normalizes to 0 db
            ['remix', '1'], # convert to mono
            ['rate', str(self.sample_rate)], # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length-audio_length)
            noise = noise[..., offset:offset+audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length-noise_length))], dim=-1)

        return self.coeff * x + (1.0 - self.coeff) * noise.view_as(x)

aug_versions = [
    "n", "p", "r",
    "n,p", "n,r", "p,r",
    "n,p,r"
]

punjabi_10h = [ l.split("\t")[0] for l in open('/workspace/data/manifests/pretrain/punjabi_train-10h.tsv', 'r').readlines() if ".wav" in l ]

for filename in tqdm(punjabi_10h):

    original_path = Path('/workspace/data/IndicSUPERB/punjabi/audio') / filename

    x_raw, sr = torchaudio.load(str(original_path))

    # Randomly select 6 transformations for each file to produce extra 60 hours from 10 hours of audio
    transform_combos = [ t.split(",") for t in random.sample(aug_versions, 6) ]

    for transforms in transform_combos:

        x = x_raw

        for transform in transforms:
            if transform == "p":
                random_pitch_shift = lambda: np.random.normal(0, 50)
                random_pitch_shift_effect = augment.EffectChain().pitch(random_pitch_shift).rate(sr)
                x = random_pitch_shift_effect.apply(x, src_info={'rate': sr})
            if transform == "r":
                random_room_size = lambda: np.random.normal(0, 60)
                random_room_size = min(abs(random_room_size()), 100)
                random_reverb = augment.EffectChain().reverb(50, 50, random_room_size).channels(1)
                x = random_reverb.apply(x, src_info={'rate': sr})
            if transform == "n":
                noise_transform = RandomBackgroundNoise(sr, '/workspace/data/musan')
                x = noise_transform(x)

        augmented_path = f"/workspace/data/IndicSUPERB/punjabi/audio-augmented/{original_path.stem}_{''.join(transforms)}.wav"
        torchaudio.save(augmented_path, x, sr)
