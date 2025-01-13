import os
import shutil
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import random
from tqdm import tqdm

# Hyperparameters
NJ = 16
TARGET_SAMPLE_RATE = 16000

class ASRdataset(Dataset):
    def __init__(self, wav_paths, texts, stoi, sp):
        self.keys = list(wav_paths.keys())
        self.stoi = stoi
        self.sp = sp
        
        common_ids = list(set(wav_paths.keys()) & set(texts.keys()))
        
        self.data = {}
        
        for i, id_ in tqdm(enumerate(common_ids), total = len(common_ids), desc="Preparing dataset"):
            arr, sr = torchaudio.load(wav_paths[id_])
            if sr != TARGET_SAMPLE_RATE:
                resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
                arr = resampler(arr)
            if arr.shape[0]>1:
                c = random.randint(0, arr.shape[0]-1)
                arr = arr[c].unsqueeze(0)
            tokens = [self.stoi.get(t, self.stoi["<unk>"]) for t in self.sp.EncodeAsPieces(texts[id_]) + ["<eos>"]]
            self.data[i] = {"id":id_ ,"speech":arr, "length":arr.shape[0], "tokens":torch.LongTensor(tokens)}
                
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(data_path):
    assert os.path.exists(os.path.join(data_path, "wav.scp"))
    assert os.path.exists(os.path.join(data_path, "text"))

    with open(os.path.join(data_path, "wav.scp"), "r", encoding="utf-8") as f1, \
         open(os.path.join(data_path, "text"), "r", encoding="utf-8") as f2:
        wav_lines = f1.read().splitlines()
        text_lines = f2.read().splitlines()

        wavs = {line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1] for line in wav_lines}
        texts = {line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1] for line in text_lines}

    common_utts = wavs.keys() & texts.keys()
    wavs = {k: v for k, v in wavs.items() if k in common_utts}
    texts = {k: v for k, v in texts.items() if k in common_utts}
    return wavs, texts


def prepare_text_vocab(train_texts, expdir):
    data_dir = os.path.join(expdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    dump_text_path = os.path.join(data_dir, "dump_text")
    with open(dump_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts.values()))
    
    spm.SentencePieceTrainer.train(
        input=dump_text_path,
        model_prefix=os.path.join(data_dir, "spm"),
        model_type="char",
        character_coverage=1.0,
        user_defined_symbols=["<sos>", "<eos>"]
    )
    
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(data_dir, "spm.model"))
    
    with open(os.path.join(data_dir, "spm.vocab"), "r", encoding="utf-8") as f:
        vocab = f.read().splitlines()
    
    stoi = {line.split()[0]: i for i, line in enumerate(vocab)}
    itos = {v: k for k, v in stoi.items()}
    return sp, stoi, itos


def prepare_datasets(train_data_path, valid_data_path, expdir):
    if os.path.exists(expdir):
        shutil.rmtree(expdir)
    os.makedirs(expdir)

    train_wavs, train_texts = load_data(train_data_path)
    valid_wavs, valid_texts = load_data(valid_data_path)
    sp, stoi, itos = prepare_text_vocab(train_texts, expdir)

    train_dataset = ASRdataset(train_wavs, train_texts, stoi, sp)
    valid_dataset = ASRdataset(valid_wavs, valid_texts, stoi, sp)

    return train_dataset, valid_dataset, stoi, itos, sp


class SortedSampler:
    def __init__(self, data_source, max_frames, seed, stft_center, win_length, hop_length):
        self.data_source = data_source
        self.seed = seed
        
        def get_frame_length(ilen):
            if stft_center:
                olen = 1 + ilen // hop_length
            else:
                olen = 1 + (ilen - win_length) // hop_length
            return olen
        
        indices = {k:get_frame_length(v["length"]) for k, v in self.data_source.data.items()} 
        indices = dict(sorted(indices.items(), key=lambda x : x[1]), reverse=True)
        batches = []
        batch = []
        batch_length = 0
        for i, len_ in indices.items():
            if batch_length + len_ <= max_frames:
                batch.append(i)
                batch_length += len_
            else:
                batches.append(batch)
                batch = [i]
                batch_length = len_
        
        self.batches = batches
        if seed:
            random.seed(seed)
            random.shuffle(self.batches)
    
    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
            
    
        


def collate_fn(batch):
    ids_ = [b["id"] for b in batch]
    speech = [b["speech"].squeeze(0) for b in batch]
    tokens = [b["tokens"] for b in batch]
    lengths = [b["length"] for b in batch]
    
    speech = torch.nn.utils.rnn.pad_sequence(speech, batch_first=True, padding_value=0.0)
    text = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=-1)
    lengths = torch.tensor(lengths)
    
    # print(ids_)
    
    return {
        "speech": speech,
        "tokens": text,
        "lengths": lengths
    }
    

