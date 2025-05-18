from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import concatenate_datasets, Dataset, Audio
import pandas as pd
import re
from text2phonemesequence import Text2PhonemeSequence
import torch
from pythainlp.util import num_to_thaiword
import hydra
import os
from tqdm import tqdm
from .augmentation import AudioAugmentationApplier
from typing import Literal
from num2words import num2words
from pythainlp.util import normalize
import string

class MiipherDataModule(LightningDataModule):
    REQUIRED_COLUMNS = ["audio_path", "text"]
    TH_CHAR_RANGE = r"\u0E00-\u0E7F"  # Thai block
    ENG_CHAR_RANGE = r"a-zA-Z"
    DIGIT_PATTERN = re.compile(r"\d+")
    TH_EN_NUMBER_PATTERN = r"[A-Za-z'-]+(?:\s+[A-Za-z'-]+)*|[\u0E00-\u0E7F]+(?:\s+[\u0E00-\u0E7F]+)*|\d+"
    PHONEME_SPACE_CHAR = " ▁ "

    def __init__(self, cfg) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_ssl_processor = hydra.utils.instantiate(
            cfg.data.speech_ssl_processor.processor
        )
        self.audio_augmentation_applier = AudioAugmentationApplier(cfg.data.augmentation)
        self.speech_ssl_sr = cfg.data.speech_ssl_processor.sr
        self.phoneme_tokenizer = hydra.utils.instantiate(cfg.data.phoneme_tokenizer)
        if cfg.data.text_language.lang_code != "tha":
            raise ValueError(
                f"Unsupported language code: {cfg.data.text_language.lang_code}. Only 'tha' is supported."
            )
        
        self.text2phone_convertor = Text2PhonemeSequence(language=cfg.data.text_language.lang_code, is_cuda=self.device)
        self.cfg = cfg

    def replace_digits_with_thaiword(self, text):
        return self.DIGIT_PATTERN.sub(lambda m: num_to_thaiword(int(m.group())), text)

    def clean_text(self, text, remove_punctuation: bool=True):
        # Remove text in brackets ()
        text = re.sub(r"\(.*?\)", "", text)
        
        # Normalize the text
        text = normalize(text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        if remove_punctuation:
            text = "".join(
                [char for char in text if char not in string.punctuation]
            )
        return text
    
    def split_th_eng_numbers(self, text):
        # Split the text into Thai, English, and numbers
        parts = re.findall(self.TH_EN_NUMBER_PATTERN, text)
        return parts

    def detect_language(self, text) -> Literal['thai', 'english', 'thai+english', 'number']:

        # Count Thai and English characters
        thai_chars = re.findall(f'[{self.TH_CHAR_RANGE}]', text)
        english_chars = re.findall(f'[{self.ENG_CHAR_RANGE}]', text)

        thai_count = len(thai_chars)
        english_count = len(english_chars)

        if thai_count > 0 and english_count == 0:
            return "thai"
        elif english_count > 0 and thai_count == 0:
            return "english"
        elif thai_count > 0 and english_count > 0:
            return "thai+english"
        else:
            return "number"

    # TODO: Dont hardcode the language code - THAI
    def get_phoneme(self, text):
        splitted_text = self.split_th_eng_numbers(text)
        splitted_text_indices = [
            {"text": part, "lang": self.detect_language(part)} for part in splitted_text
        ]

        total_thai_len = len(
            [part for part in splitted_text_indices if part["lang"] == "thai"]
        )

        for i, part in enumerate(splitted_text_indices):
            if part["lang"] == "thai" or (part["lang"] == "number" and total_thai_len > 0):
                # Convert Thai numbers to words
                text = self.replace_digits_with_thaiword(
                    splitted_text_indices[i]["text"]
                )
                splitted_text_indices[i]["phoneme"] = self.custom_lang_text2phone_convertor.infer_sentence(text)
            elif part["lang"] == "english":
                text = splitted_text_indices[i]["text"]
                splitted_text_indices[i]["phoneme"] = self.en_lang_text2phone_convertor.infer_sentence(text)
            elif part["lang"] == "number":
                # Convert English numbers to words
                text = num2words(
                    splitted_text_indices[i]["text"], lang="en", to="currency"
                )
                splitted_text_indices[i]["phoneme"] = self.en_lang_text2phone_convertor.infer_sentence(text)

        phoneme = self.PHONEME_SPACE_CHAR.join(
            [part["phoneme"] for part in splitted_text_indices]
        )
        return phoneme


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
        # Replace digits with Thai words
        df["text"] = df["text"].apply(self.replace_digits_with_thaiword)
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


    def setup(self, stage: str=None):
        self.train_dataset = concatenate_datasets([
            self.load_dataset_from_csv(dataset_csv_path) for dataset_csv_path in self.cfg.data.train_dataset_csv_path
        ]).shuffle(seed=42)

        if self.cfg.data.val_dataset_csv_path is not None:
            self.val_dataset = concatenate_datasets([
                self.load_dataset_from_csv(dataset_csv_path) for dataset_csv_path in self.cfg.data.val_dataset_csv_path
            ])
        else:
            #  Split the train dataset into train and validation sets with 80% for training and 20% for validation
            self.train_dataset = self.train_dataset.train_test_split(
                test_size=0.2, shuffle=True, seed=42
            )
            self.val_dataset = self.train_dataset["test"]
            self.train_dataset = self.train_dataset["train"]


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
            clean_wav = sample["audio_path"]["array"]
            tensor_clean_wav = torch.tensor(clean_wav)

            #  Clip the audio to 20 seconds.
            tensor_clean_wav = tensor_clean_wav.squeeze()[:self.cfg.data.sample_rate * 20]
            clean_wav_16ks.append(
                tensor_clean_wav
            )

            degraded_wav = self.audio_augmentation_applier(tensor_clean_wav, self.cfg.data.sample_rate)
            degraded_wav = degraded_wav.squeeze()[:self.cfg.data.sample_rate * 20]
            degraded_wav_16ks.append(
                degraded_wav
            )
        output["degraded_wav_16k"] = pad_sequence(degraded_wav_16ks, batch_first=True)
        output["degraded_wav_16k_lengths"] = torch.tensor(
            [degraded_wav_16k.size(0) for degraded_wav_16k in degraded_wav_16ks]
        )
        output["clean_ssl_input"] = self.speech_ssl_processor(
            [clean_wav.numpy() for clean_wav in clean_wav_16ks],
            return_tensors="pt",
            sampling_rate=self.cfg.data.sample_rate,
            padding=True,
        )
        output["degraded_ssl_input"] = self.speech_ssl_processor(
            [degraded_wav.numpy() for degraded_wav in degraded_wav_16ks],
            return_tensors="pt",
            sampling_rate=self.cfg.data.sample_rate,
            padding=True,
        )
        output["phoneme_input_ids"] = self.phoneme_tokenizer(
            [sample["phoneme"] for sample in batch], return_tensors="pt", padding=True
        )
        return output
