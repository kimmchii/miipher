# Implementated from https://github.com/Wataru-Nakata/miipher/blob/main/src/miipher/preprocess/noiseAugmentation.py
from typing import Any
from torch import nn as nn
import torchaudio
import random
import pyroomacoustics as pra
import numpy as np
from numpy.random import default_rng

import torch
from pathlib import Path
from tqdm import tqdm
import random
import scipy
from silero_vad import get_speech_timestamps, load_silero_vad

class AudioAugmentationApplier:
    def __init__(self, cfg) -> None:
        self.format_encoding_pairs = cfg.format_encoding_pairs
        self.reverb_conditions = cfg.reverb_conditions
        self.background_noise = cfg.background_noise
        self.cfg = cfg
        self.rirs = []
        self.prepare_rir(cfg.n_rirs)
        self.noise_audio_paths = []
        self.rng = default_rng(seed=42)

        for noise_dir in self.cfg.background_noise.dirs:
            print("Loading noise files from:", noise_dir)
            self.noise_audio_paths.extend(
                AudioAugmentationApplier._get_files(noise_dir, self.cfg.background_noise.extensions)
            )

    @staticmethod
    def _get_files(directory, extensions: list[str] = ["wav", "flac", "mp3"]):
        all_files = []
        for ext in extensions:
            all_files.extend(Path(directory).rglob(f"*.{ext}"))
        print(f"Found {len(all_files)} files with extensions {extensions} in {directory}")
        return all_files

    @staticmethod
    def _align_waveform(wav1, wav2):
        assert wav2.size(1) >= wav1.size(1), f"size of wav2: {wav2.size(1)} is greater than the size of wav1: {wav1.size(1)}"
        diff = wav2.size(1) - wav1.size(1)
        min_mse = float("inf")
        best_i = -1

        for i in range(diff):
            segment = wav2[:, i : i + wav1.size(1)]
            mse = torch.mean((wav1 - segment) ** 2).item()
            if mse < min_mse:
                min_mse = mse
                best_i = i

        return best_i, wav2[:, best_i : best_i + wav1.size(1)]
    

    def apply_codec(self, waveform, sample_rate):
        if len(self.format_encoding_pairs) == 0:
            return waveform
        param = random.choice(self.format_encoding_pairs)
        augmented = torchaudio.functional.apply_codec(
            waveform=waveform.float(), sample_rate=sample_rate, **param
        )
        # mp3 encoding may increase the length of the waveform by zero-padding
        if waveform.size(1) != augmented.size(1):
            _, augmented = AudioAugmentationApplier._align_waveform(waveform, augmented)
        return augmented.float()

    def apply_reverb(self, waveform):
        if len(self.rirs) == 0:
            raise RuntimeError
        rir = random.choice(self.rirs)
        augmented = torchaudio.functional.fftconvolve(waveform, rir)
        # rir convolution may increase the length of the waveform
        if waveform.size(1) != augmented.size(1):
            augmented = augmented[:, : waveform.size(1)]
        return augmented.float()

    def prepare_rir(self, n_rirs):
        print("Preparing rir: ")
        for _ in tqdm(range(n_rirs)):
            xy_minmax = self.reverb_conditions.room_xy
            z_minmax = self.reverb_conditions.room_z
            x = random.uniform(xy_minmax.min, xy_minmax.max)
            y = random.uniform(xy_minmax.min, xy_minmax.max)
            z = random.uniform(z_minmax.min, z_minmax.max)
            corners = np.array([[0, 0], [0, y], [x, y], [x, 0]]).T
            room = pra.Room.from_corners(corners, **self.reverb_conditions.room_params)
            room.extrude(z)
            room.add_source(self.cfg.reverb_conditions.source_pos)
            room.add_microphone(self.cfg.reverb_conditions.mic_pos)

            room.compute_rir()
            rir = torch.tensor(np.array(room.rir[0]))
            rir = rir / rir.norm(p=2)
            self.rirs.append(rir)

    def apply_noise_with_snr(self, waveform, snr_db):
        """
        waveform: torch.Tensor, shape (channels, samples) or (samples,)
        snr_db: float, desired signal-to-noise ratio in dB
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise

    def cut_off_frequency(self, waveform, sample_rate, cutoff_freq):
        """
        Apply a low-pass filter to the waveform to cut off frequencies above cutoff_freq.
        waveform: torch.Tensor, shape (channels, samples) or (samples,)
        sample_rate: int
        cutoff_freq: float, cutoff frequency in Hz
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(1, normal_cutoff, btype='low', analog=False)
        
        filtered_waveform = scipy.signal.filtfilt(b, a, waveform.numpy(), axis=-1)
        return torch.from_numpy(filtered_waveform.copy()).float()
    
    def apply_burst_static_speech(self, waveform, sample_rate, burst_amplitude):
        """
        waveform: torch.Tensor, shape (channels, samples) or (samples,)
        sample_rate: int
        burst_amplitude: float
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform_aug = waveform.clone()

        speech_timestamps = get_speech_timestamps(
            audio=waveform_aug,
            model=load_silero_vad(),
            sampling_rate=sample_rate,
            return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
        #Apply static noise bursts to speech segments
        for segment in speech_timestamps:
            start, end = int(segment['start'] * sample_rate), int(segment['end'] * sample_rate)
            static_burst = torch.randn(1, end - start) * burst_amplitude

            waveform_aug[:, start:end] += static_burst

        return waveform_aug.clamp(-1.0, 1.0)  # Ensure waveform is within [-1, 1] range
    
    def apply_bg_noise(self, waveform, sample_rate):
        snr_max, snr_min = self.background_noise.snr.max, self.background_noise.snr.min
        snr = random.uniform(snr_min, snr_max)

        noise_path = random.choice(self.noise_audio_paths)
        noise, noise_sr = torchaudio.load(noise_path)
        noise /= noise.norm(p=2)
        if noise.size(0) > 1:
            noise = noise[0].unsqueeze(0)
        noise = torchaudio.functional.resample(noise, noise_sr, sample_rate)
        if noise.size(1) >= waveform.size(1):
            start_idx = random.randint(0, noise.size(1) - waveform.size(1))
            end_idx = start_idx + waveform.size(1)
            noise = noise[:, start_idx:end_idx]
        else:
            noise = noise.repeat(1, waveform.size(1) // noise.size(1) + 1)[
                :, : waveform.size(1)
            ]
        if noise.abs().max() > 0:
            augmented = torchaudio.functional.add_noise(
                waveform=waveform, noise=noise, snr=torch.tensor([snr])
            )
        else:
            augmented = waveform
        return augmented

    def process(self, waveform, sample_rate):
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        orginal_wav_len = waveform.size(1)

        # Randomly apply background noise
        random_apply_bg_noise_score = self.rng.uniform(0, 1)
        if random_apply_bg_noise_score > self.cfg.background_noise.p:
            waveform = self.apply_bg_noise(waveform, sample_rate)
        
        # Randomly apply reverb
        random_apply_reverb_score = self.rng.uniform(0, 1)
        if random_apply_reverb_score > self.cfg.reverb_conditions.p:
            waveform = self.apply_reverb(waveform)

        waveform = self.apply_codec(waveform, sample_rate)

        assert orginal_wav_len == waveform.size(1), f"{orginal_wav_len}, {waveform.size(1)}"

        return waveform.squeeze()

    def __call__(self, waveform: torch.tensor, sample_rate: int) -> torch.tensor:
        return self.process(waveform, sample_rate)
