import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

librispeech_root = "/speech/arjun/data/LibriSpeech"
data_save_dir = "/path/to/save/processed/data"

splits = {
    "train": ["train-clean-100", "train-clean-360", "train-other-500"],
    "dev_other": ["dev-other"],
    "dev_clean": ["dev-clean"],
    "test_clean": ["test-clean"],
    "test_other": ["test-other"]
}

splits = {k: [Path(librispeech_root) / v for v in item] for k, item in splits.items()}
data_save_dir = Path(data_save_dir)
data_save_dir.mkdir(parents=True, exist_ok=True)

def process_split(split_name, split_data_list):
    data = {}
    print(f"Processing {split_name}...")

    for split_data in split_data_list:
        all_flacs = split_data.rglob("*.flac")
        all_text = split_data.rglob("*.txt")

        for flac_file in all_flacs:
            id_ = flac_file.stem
            data[id_] = {"path": flac_file}

        for text_file in all_text:
            with open(text_file, "r", encoding="utf-8") as f:
                text_lines = f.read().splitlines()
                for line in text_lines:
                    id_, content = line.strip().split(maxsplit=1)
                    if id_ in data.keys() and content.strip() != "":
                        data[id_]["text"] = content

    data_list = [{"id_": id_, "path": str(id_data["path"]), "text": id_data.get("text", "")} for id_, id_data in data.items()]
    df = pd.DataFrame(data_list)
    parquet_file = data_save_dir / f"{split_name}.parquet"
    df.to_parquet(parquet_file, engine="pyarrow")
    print(f"finished processing {split_name}, saved to {parquet_file}")

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_split, split_name, split_data_list) for split_name, split_data_list in splits.items()]
        for future in futures:
            future.result()
