#!/bin/bash
#SBATCH --job-name=GSC3                  # Job name
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=salaj.au@gmail.com  # Where to send mail
#SBATCH --output=slurm_out_3lstm_%j.log       # Standard output and error log
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=IGIcrunchers

conda activate venv2
# python3 train.py --model_architecture=lsnn --n_hidden=2048 --window_stride_ms=0.5 --avg_spikes=True --comment=LSNN_stride0.5_win30_AvgT
python3 train.py --model_architecture=lstm --n_hidden=512 --window_stride_ms=0.5 --comment=LSTM_stride0.5_win30
