from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import concatenate_datasets, Dataset, Audio
import pandas as pd
import re
from text2phonemesequence import Text2PhonemeSequence
import torch
import torchaudio
import hydra
import os
from tqdm import tqdm
from .augmentation import AudioAugmentationApplier


class MiipherDataModule(LightningDataModule):
    REQUIRED_COLUMNS = ["audio_path", "text"]

    def __init__(self, cfg) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_ssl_processor = hydra.utils.instantiate(
            cfg.data.speech_ssl_processor.processor
        )
        self.audio_augmentation_applier = AudioAugmentationApplier(cfg.data.augmentation)
        self.speech_ssl_sr = cfg.data.speech_ssl_processor.sr
        self.phoneme_tokenizer = hydra.utils.instantiate(cfg.data.phoneme_tokenizer)
        self.text2phone_convertor = Text2PhonemeSequence(language=cfg.data.text_language.lang_code, is_cuda=self.device)
        self.cfg = cfg


    def clean_text(self, text):
        # Remove text in brackets ()
        text = re.sub(r"\(.*?\)", "", text)
        # Remove some special characters
        text = re.sub(r"[^\w\s]", "", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_dataset_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        # Check if all required columns are present
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        # Clean the text column
        df["text"] = df["text"].astype(str)
        # Remove text in brackets ()
        df["text"] = df["text"].apply(self.clean_text)
        
        # TODO: handle a case where the text contains multiple languages.
        # Convert text to phonemes
        df["phoneme"] = df["text"].apply(
            lambda x: self.text2phone_convertor.infer_sentence(x)
        )

        # Check if audio files exist
        for audio_path in tqdm(df["audio_path"]):
            if not os.path.exists(audio_path):
                print(f"Audio file does not exist: {audio_path}, Skipping...")
        
        texts = df["text"].tolist()
        phonemes = df["phoneme"].tolist()
        audio_paths = df["audio_path"].tolist()

        return Dataset.from_dict(
            {
                "text": texts,
                "phoneme": phonemes,
                "audio_path": audio_paths,
            }
        ).cast_column("audio_path", Audio(sampling_rate=self.cfg.data.sample_rate))


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    @torch.no_grad()
    def collate_fn(self, batch):
        output = dict()
        degraded_wav_16ks = []
        clean_wav_16ks = []

        for sample in batch:
            clean_wav, sr = sample["speech.wav"]
            clean_wav_16ks.append(
                torchaudio.functional.resample(clean_wav, sr, new_freq=16000).squeeze()[:16000*20]
            )
            degraded_wav, sr = sample["degraded_speech.wav"]
            degraded_wav_16ks.append(
                torchaudio.functional.resample(
                    degraded_wav, sr, new_freq=16000
                ).squeeze()[:16000*20]
            )
        output["degraded_wav_16k"] = pad_sequence(degraded_wav_16ks, batch_first=True)
        output["degraded_wav_16k_lengths"] = torch.tensor(
            [degraded_wav_16k.size(0) for degraded_wav_16k in degraded_wav_16ks]
        )
        output["clean_ssl_input"] = self.speech_ssl_processor(
            [x.numpy() for x in clean_wav_16ks],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        )
        output["degraded_ssl_input"] = self.speech_ssl_processor(
            [x.numpy() for x in degraded_wav_16ks],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        )
        output["phoneme_input_ids"] = self.phoneme_tokenizer(
            [b["phoneme.txt"] for b in batch], return_tensors="pt", padding=True
        )
        return output
