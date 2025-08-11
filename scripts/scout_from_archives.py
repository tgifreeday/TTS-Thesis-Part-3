#!/usr/bin/env python3
import argparse
import os
import tarfile
import shutil
import subprocess
import sys
import json
from pathlib import Path


def find_first(base: Path, name: str):
    matches = list(base.rglob(name))
    return matches[0] if matches else None


def run(cmd: list, cwd: Path):
    print('>',' '.join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def main():
    p = argparse.ArgumentParser(description='Scout run from code and dataset archives')
    p.add_argument('--code-archive', required=True, help='Path to code tar.gz')
    p.add_argument('--dataset-archive', required=True, help='Path to dataset tar.gz')
    p.add_argument('--output-root', default='scout_workspace', help='Working directory')
    p.add_argument('--max-epochs', type=int, default=2)
    p.add_argument('--limit-train-batches', type=int, default=50)
    p.add_argument('--limit-val-batches', type=int, default=10)
    p.add_argument('--preview', action='store_true', help='Synthesize a preview wav after training')
    args = p.parse_args()

    work = Path(args.output_root).absolute()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # 1) Extract code
    with tarfile.open(args.code_archive, 'r:gz') as tar:
        tar.extractall(work)
    # if the archive contains a top-level folder, use it
    # pick directory with src/train.py
    candidate = None
    for d in work.iterdir():
        if d.is_dir() and (d/'src'/'train.py').exists():
            candidate = d
            break
    code_dir = candidate if candidate else work

    # 2) Extract dataset into code_dir/data/processed
    data_processed = code_dir/'data'/'processed'
    data_processed.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.dataset_archive, 'r:gz') as tar:
        tar.extractall(data_processed)

    # 3) Standardize expected dataset paths
    lf = find_first(data_processed, 'linguistic_features_with_audio_paths.json')
    tf = find_first(data_processed, 'train_files.txt')
    vf = find_first(data_processed, 'val_files.txt')
    if not (lf and tf and vf):
        raise FileNotFoundError('Could not locate required dataset files after extraction')
    shutil.copy(lf, data_processed/'linguistic_features_with_audio_paths.json')
    shutil.copy(tf, data_processed/'train_files.txt')
    shutil.copy(vf, data_processed/'val_files.txt')

    # 4) Patch scout config
    import yaml
    cfg_path = code_dir/'configs'/'scout_mlp.yaml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['data']['dataset_path'] = str(data_processed/'linguistic_features_with_audio_paths.json')
    cfg['data']['train_files'] = str(data_processed/'train_files.txt')
    cfg['data']['val_files'] = str(data_processed/'val_files.txt')
    cfg['training']['max_epochs'] = args.max_epochs
    cfg['debug'] = cfg.get('debug', {})
    cfg['debug']['limit_train_batches'] = args.limit_train_batches
    cfg['debug']['limit_val_batches'] = args.limit_val_batches
    patched_cfg = code_dir/'configs'/'scout_from_archives.yaml'
    with open(patched_cfg, 'w') as f:
        yaml.safe_dump(cfg, f)

    # 5) Install requirements
    run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt', 'librosa', 'soundfile'], cwd=code_dir)

    # 6) Run training briefly
    out_dir = code_dir/'experiments'/'scout_from_archives'
    out_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, 'src/train.py', '--config', str(patched_cfg), '--output_dir', str(out_dir), '--debug'], cwd=code_dir)

    # 7) Optional preview synthesis using Griffin-Lim
    if args.preview:
        # Minimal inference to create a wav
        sys.path.insert(0, str(code_dir/'src'))
        from utils.config import Config
        from train import TTSTrainer, create_data_loaders
        import torch
        import numpy as np
        import librosa
        import soundfile as sf

        config = Config.from_yaml(str(patched_cfg))
        train_loader, val_loader = create_data_loaders(config)
        model = TTSTrainer(config)
        model.eval()
        batch = next(iter(val_loader))
        with torch.no_grad():
            out = model.forward(batch, use_teacher_forcing=False)
            mel = out['mel_spectrogram'][0].cpu().numpy().T  # [n_mels, T]
        # Griffin-Lim reconstruction
        S = librosa.db_to_amplitude(mel)
        audio = librosa.feature.inverse.mel_to_audio(S, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_iter=32)
        sf.write(out_dir/'scout_preview.wav', audio, 22050)
        print('Preview written to', out_dir/'scout_preview.wav')

    print('Scout run completed. Outputs in', out_dir)


if __name__ == '__main__':
    main() 