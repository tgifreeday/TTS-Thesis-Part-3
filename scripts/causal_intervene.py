#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import torch
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from dataset import TTSDataset
from train import TTSTrainer


def main():
    parser = argparse.ArgumentParser(description="Causal intervention: flip a feature and observe outputs")
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--feature', default='primary_stress_pos', choices=['primary_stress_pos','stress_level'])
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = Config.from_yaml(args.config)
    ds = TTSDataset(config.data.dataset_path, file_list_path=config.data.val_files,
                    base_path="data/processed", input_features=config.data.input_features)
    # pick first sample
    batch = [ds[0]]
    # build minimal batch dict with padding via collate_fn from train
    from train import collate_fn
    batch = collate_fn(batch)

    # clone batch and flip feature
    intervened = {k: (v.clone() if torch.is_tensor(v) else v) for k,v in batch.items()}
    # naive flip: if feature exists in linguistic_features tensor by known column index
    # For simplicity, toggle stress_level column (assume index 1) or primary_stress_pos (index 2)
    if args.feature == 'stress_level' and intervened['linguistic_features'].shape[2] > 1:
        intervened['linguistic_features'][:, :, 1] = 1 - intervened['linguistic_features'][:, :, 1]
    elif args.feature == 'primary_stress_pos' and intervened['linguistic_features'].shape[2] > 2:
        intervened['linguistic_features'][:, :, 2] = (intervened['linguistic_features'][:, :, 2] + 1) % 3

    model = TTSTrainer(config)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    with torch.no_grad():
        out_base = model.forward(batch, use_teacher_forcing=False)
        out_int = model.forward(intervened, use_teacher_forcing=False)

    # save mel and f0 comparisons
    np.save(Path(args.output_dir)/'mel_base.npy', out_base['mel_spectrogram'].squeeze(0).cpu().numpy())
    np.save(Path(args.output_dir)/'mel_intervened.npy', out_int['mel_spectrogram'].squeeze(0).cpu().numpy())
    if 'f0' in out_base['prosody_predictions']:
        np.save(Path(args.output_dir)/'f0_base.npy', out_base['prosody_predictions']['f0'].squeeze(0).cpu().numpy())
        np.save(Path(args.output_dir)/'f0_intervened.npy', out_int['prosody_predictions']['f0'].squeeze(0).cpu().numpy())

    print(f"Saved causal intervention outputs to {args.output_dir}")

if __name__ == '__main__':
    main() 