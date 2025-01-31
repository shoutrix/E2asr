import os
import pandas as pd
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sentencepiece as spm

libriTTS_root = "/speech/Database/LibriTTS/LibriTTS"

splits = {
    "train": ["train-clean-100", "train-clean-360", "train-other-500"],
    "dev_other": ["dev-other"],
    "dev_clean": ["dev-clean"],
    "test_clean": ["test-clean"],
    "test_other": ["test-other"]
}

splits = {k: [Path(libriTTS_root) / v for v in item] for k, item in splits.items()}


SENTENCEPIECE_MODEL = "/speech/shoutrik/torch_exp/E2asr/exp/LibriTTS_trial03/dump/spm.model"

def process_split(split_name, split_data_list):
    data = {}
    print(f"Processing {split_name} ...")
    
    # Initialize SentencePieceProcessor in subprocess
    sp = spm.SentencePieceProcessor()
    sp.load(SENTENCEPIECE_MODEL)

    for split_data in split_data_list:
        all_wavs = split_data.rglob("*.wav")

        for wav_file in all_wavs:
            text_file = wav_file.with_suffix(".normalized.txt")
            if not text_file.exists():
                print(f"Warning: Text file not found for {wav_file}, skipping.")
                continue

            id_ = wav_file.stem
            try:
                dur = sf.info(wav_file).duration
                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                data[id_] = {"path": wav_file, "duration": dur, "text": text}
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                continue

    data_list = [{"id_": id_, "duration": id_data["duration"], "path": str(id_data["path"]), "text": id_data.get("text", "")} for id_, id_data in data.items()]
    
    df = pd.DataFrame(data_list)

    # Compute text lengths using SentencePiece
    len_fn = lambda x: len(sp.EncodeAsPieces(x))
    df["text_lengths"] = df["text"].apply(len_fn)

    # Compute frame lengths
    def frame_len(duration):
        frames = duration * 16000  # Sample rate
        frames = 1 + frames // 160
        frames = (((((frames - 1) // 2) - 1) // 2) - 1) // 2
        return frames

    df["frames_lengths"] = df["duration"].apply(frame_len)

    # Filter out entries where frame lengths are smaller than text lengths
    filtered_df = df[df["frames_lengths"] < df["text_lengths"]]
    print(f"{len(filtered_df)} samples out of {len(df)} from {split_name} have frame_length less than text_lengths.")

    # parquet_file = data_save_dir / f"{split_name}.parquet"
    # filtered_df.to_parquet(parquet_file, engine="pyarrow")
    # print(f"Finished processing {split_name}, saved to {parquet_file}")

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_split, split_name, split_data_list) for split_name, split_data_list in splits.items()]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing: {e}")
