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
from num2words import num2words
from pythainlp.util import normalize
import string
import attacut

from .augmentation import AudioAugmentationApplier
class MiipherDataModule(LightningDataModule):
    REQUIRED_COLUMNS = ["audio_path", "text"]
    PHONEME_SPACE_CHAR = " ▁ "

    def __init__(self, cfg) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_ssl_processor = hydra.utils.instantiate(
            cfg.data.speech_ssl_processor.processor
        )

        # Load the T5 model and tokenizer for graphene to phoneme processing
        self.g2p_processor = hydra.utils.instantiate(cfg.data.g2p_processor.model)
        self.g2p_tokenizer = hydra.utils.instantiate(
            cfg.data.g2p_processor.tokenizer, use_fast=True
        )
        self.supported_char_ranges = {lang_range["lang_code"]: lang_range["char_range_regex"] for lang_range in cfg.data.g2p_processor.langs}

        self.custom_lang_range_splitter_pattern = r"|".join([lang_range for lang_range in self.supported_char_ranges.values()])
        # Add number to the pattern
        self.custom_lang_range_splitter_pattern = f"{self.custom_lang_range_splitter_pattern}|\d+"
        self.custom_lang_range_splitter = re.compile(self.custom_lang_range_splitter_pattern)

        self.audio_augmentation_applier = AudioAugmentationApplier(cfg.data.augmentation)
        self.speech_ssl_sr = cfg.data.speech_ssl_processor.sr
        self.phoneme_tokenizer = hydra.utils.instantiate(cfg.data.phoneme_tokenizer)
        
        self.custom_lang_text2phone_convertor = Text2PhonemeSequence(language=cfg.data.text_language.lang_code, is_cuda=self.device)
        self.en_lang_text2phone_convertor = Text2PhonemeSequence(language="eng-us", is_cuda=self.device)

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
    
    def _split_th_eng_numbers(self, text):
        # Split the text into Thai, English, and numbers
        # parts = re.findall(self.TH_EN_NUMBER_PATTERN, text)
        parts = re.findall(self.custom_lang_range_splitter, text)
        return parts

    def detect_language(self, text) -> str:
        # Count Thai and English characters
        num_lang_count = {}
        for lang_code, char_range in self.supported_char_ranges.items():
            lang_chars = re.findall(char_range, text)
            num_lang_count[lang_code] = len(lang_chars)

        # Check if the text contains only numbers
        if not any(char.isalpha() for char in text):
            return "number"
        # Return the language based on the character counts,
        # If the num_lang_count is empty, return "unknown" otherwise return the language based on the counts.
        if not num_lang_count:
            return "unknown"
        # Get the language code that has len(lang_chars) > 0
        found_langs = [lang_code for lang_code, count in num_lang_count.items() if count > 0]
        return "+".join(found_langs) if found_langs else "unknown"


    # TODO: Dont hardcode the language code - THAI
    def get_phoneme(self, text):
        splitted_text = self._split_th_eng_numbers(text)

        # Ignore the empty parts and parts that only contain "ๆ".
        splitted_text_indices = [
            {"text": part, "lang": self.detect_language(part)} for part in splitted_text if part.strip() != ""
        ]

        # TODO: Could handle only Thai and English numbers.
        total_thai_len = len(
            [part for part in splitted_text_indices if part["lang"] == "tha"]
        )

        # Format the numbers in the text.
        normalized_text = []
        for i, part in enumerate(splitted_text_indices):
            if part["lang"] == "number":
                # Assuming the number is pronunced in Thai, so we convert it to Thai words. Otherwise, we convert it to English words.
                if total_thai_len > 0:
                    # Convert Thai numbers to words
                    converted_text = self.replace_digits_with_thaiword(splitted_text_indices[i]["text"])
                    lang_tag = "tha"
                else:
                    # Normalize the text
                    converted_text = num2words(
                        splitted_text_indices[i]["text"], lang="en", to="currency"
                    )
                    lang_tag = "eng-us"

                normalized_text.append(
                    {"text": converted_text, "lang": lang_tag}
                )

            elif part["lang"] == "eng-us":
                # Normalize the text
                normalized_text.append(
                    {"text": splitted_text_indices[i]["text"], "lang": "eng-us"}
                )
            # TODO: In Thai language, we should handle a case where the text contains more than one Thai word.
            elif part["lang"] == "tha":
                tokenized_texts = attacut.tokenize(splitted_text_indices[i]["text"])
                tokenized_texts = [token for token in tokenized_texts if token.strip() != ""]
                #  Handle the case where the text is "ๆ", we will define the value of "ๆ" as the previous word.
                for token_idx, token in enumerate(tokenized_texts):
                    if token.strip() == "ๆ":
                        if token_idx > 0 and tokenized_texts[token_idx-1]["lang"] == "tha":
                            tokenized_texts[token_idx] = tokenized_texts[token_idx - 1]
                        elif token_idx == 0 and normalized_text[-1]["lang"] == "tha":
                            tokenized_texts[token_idx] = normalized_text[-1]["text"]
                        else:
                            # Drop the "ๆ" token if it is the first token or if it is not preceded by a Thai word.
                            tokenized_texts[token_idx] = ""
                # Remove the empty tokens
                tokenized_texts = [token for token in tokenized_texts if token.strip() != ""]
                normalized_text.extend(
                    [{"text": token, "lang": "tha"} for token in tokenized_texts]
                )

        # Convert the text to phonemes using the G2P model
        pre_tagged_texts = [
            f"<{part['lang']}>: {part['text']}" for part in normalized_text
        ]
        tokenized_taged_texts = self.g2p_tokenizer(
            pre_tagged_texts, padding=True, add_special_tokens=False,return_tensors='pt'
        )
        predicted_phones = self.g2p_processor.generate(
            **tokenized_taged_texts, num_beams=1,max_length=50
        )
        phonemized_texts = self.g2p_tokenizer.batch_decode(
            predicted_phones.tolist(), skip_special_tokens=True
        )
        return self.PHONEME_SPACE_CHAR.join(phonemized_texts)


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
        if "phoneme" in df.columns:
            print("Phoneme column found in the CSV file. Using it directly.")
            df["phoneme"] = df["phoneme"].astype(str)
        else:
            print("Phoneme column not found. Processing texts for phonemes...")
            # Convert text to phonemes
            df["phoneme"] = df["text"].apply(
                lambda x: self.get_phoneme(x)
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
