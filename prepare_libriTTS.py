import os
import pandas as pd
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

libriTTS_root = "/speech/Database/LibriTTS/LibriTTS"
data_save_dir = "data/LibriTTS/"

splits = {
    "train": ["train-clean-100", "train-clean-360", "train-other-500"],
    "dev_other": ["dev-other"],
    "dev_clean": ["dev-clean"],
    "test_clean": ["test-clean"],
    "test_other": ["test-other"]
}

splits = {k: [Path(libriTTS_root) / v for v in item] for k, item in splits.items()}
data_save_dir = Path(data_save_dir)
data_save_dir.mkdir(parents=True, exist_ok=True)

def process_split(split_name, split_data_list):
    data = {}
    print(f"Processing {split_name} ...")

    for split_data in split_data_list:
        all_wavs = split_data.rglob("*.wav")

        for wav_file in all_wavs:
            id_ = wav_file.stem
            dur = sf.info(wav_file).duration
            text_file = wav_file.with_suffix(".normalized.txt")
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()
            data[id_] = {"path": wav_file, "duration":dur, "text":text}

    data_list = [{"id_": id_, "duration":id_data["duration"], "path": str(id_data["path"]), "text": id_data.get("text", "")} for id_, id_data in data.items()]
    df = pd.DataFrame(data_list)
    parquet_file = data_save_dir / f"{split_name}.parquet"
    df.to_parquet(parquet_file, engine="pyarrow")
    print(f"finished processing {split_name}, saved to {parquet_file}")
    
    

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_split, split_name, split_data_list) for split_name, split_data_list in splits.items()]
        for future in futures:
            future.result()
