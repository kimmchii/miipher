import pandas as pd
from text2phonemesequence import Text2PhonemeSequence
from pythainlp.util import num_to_thaiword
from num2words import num2words
import re
import string
from typing import Literal
from pythainlp.util import normalize
from tqdm import tqdm
from datasets import Dataset
import torch

torch.set_num_threads(1)

tqdm.pandas()

REQUIRED_COLUMNS = ["audio_path", "text"]
TH_CHAR_RANGE = r"\u0E00-\u0E7F"  # Thai block
ENG_CHAR_RANGE = r"a-zA-Z"
DIGIT_PATTERN = re.compile(r"\d+")
TH_EN_NUMBER_PATTERN = r"[A-Za-z'-]+(?:\s+[A-Za-z'-]+)*|[\u0E00-\u0E7F]+(?:\s+[\u0E00-\u0E7F]+)*|\d+"
PHONEME_SPACE_CHAR = " ▁ "

def replace_digits_with_thaiword(text):
    return DIGIT_PATTERN.sub(lambda m: num_to_thaiword(int(m.group())), text)

def clean_text(text, remove_punctuation: bool=True):
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

def split_th_eng_numbers(text):
    # Split the text into Thai, English, and numbers
    parts = re.findall(TH_EN_NUMBER_PATTERN, text)
    return parts

def detect_language(text) -> Literal['thai', 'english', 'thai+english', 'number']:

    # Count Thai and English characters
    thai_chars = re.findall(f'[{TH_CHAR_RANGE}]', text)
    english_chars = re.findall(f'[{ENG_CHAR_RANGE}]', text)

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
def get_phoneme(text):
    splitted_text = split_th_eng_numbers(text)
    splitted_text_indices = [
        {"text": part, "lang": detect_language(part)} for part in splitted_text
    ]

    total_thai_len = len(
        [part for part in splitted_text_indices if part["lang"] == "thai"]
    )

    for i, part in enumerate(splitted_text_indices):
        if part["lang"] == "thai" or (part["lang"] == "number" and total_thai_len > 0):
            # Convert Thai numbers to words
            replaced_part = replace_digits_with_thaiword(
                part["text"]
            )
            splitted_text_indices[i]["phoneme"] = th_text2phone_convertor.infer_sentence(replaced_part)
        elif part["lang"] == "english":
            splitted_text_indices[i]["phoneme"] = en_text2phone_convertor.infer_sentence(part["text"])
        elif part["lang"] == "number":
            # Convert English numbers to words
            replaced_part = num2words(
                part["text"], lang="en", to="currency"
            )
            splitted_text_indices[i]["phoneme"] = en_text2phone_convertor.infer_sentence(replaced_part)

    phoneme = PHONEME_SPACE_CHAR.join(
        [part["phoneme"] for part in splitted_text_indices]
    )
    # print(f"Text: {text} >>> Phoneme: {phoneme}")
    return phoneme



if __name__ == "__main__":
    th_text2phone_convertor = Text2PhonemeSequence(language="tha", is_cuda=True)
    en_text2phone_convertor = Text2PhonemeSequence(language="eng-us", is_cuda=True)


    csv_paths = [
        "data/csv/train_data.csv",
    ]

    for csv_path in tqdm(csv_paths):
        print(f"Processing {csv_path}...")
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Check if the required columns are present
        if not all(col in df.columns for col in REQUIRED_COLUMNS):
            print(f"Missing required columns in {csv_path}.")
            continue

        # Clean the text column
        df["text"] = df["text"].apply(lambda x: clean_text(x, remove_punctuation=True))

        df["phoneme"] = df["text"].progress_apply(lambda x: get_phoneme(x))

        # Save the DataFrame to a new CSV file
        output_csv_path = csv_path.replace(".csv", "_phoneme.csv")
        df.to_csv(output_csv_path, index=False)