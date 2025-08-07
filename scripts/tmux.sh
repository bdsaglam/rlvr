#!/bin/bash

SESSION="verifiers"

# Start new session, but don't attach
tmux new-session -d -s $SESSION

# Window 0: Services needed for training
tmux rename-window -t $SESSION:0 'services'
tmux send-keys -t $SESSION:0 'conda activate verifiers' C-m
tmux send-keys -t $SESSION:0 'export CUDA_VISIBLE_DEVICES=0' C-m
tmux send-keys -t $SESSION:0 'docker-compose down --remove-orphans && docker-compose up --build' C-m

sleep 1

# Window 1: Training shell
tmux new-window -t $SESSION:1 -n 'training'
tmux send-keys -t $SESSION:1 'conda activate verifiers' C-m
tmux send-keys -t $SESSION:1 'export LD_LIBRARY_PATH=/home/baris/miniconda3/envs/verifiers/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH' C-m

# Make sure we're on the services window and attach
tmux select-window -t $SESSION:0
tmux attach-session -t $SESSION