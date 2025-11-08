#!/bin/bash

SESSION="rlvr"

# Start new session, but don't attach
tmux new-session -d -s $SESSION

# Window 0: Services
tmux rename-window -t $SESSION:0 'services'
tmux send-keys -t $SESSION:services 'source .venv/bin/activate' C-m
tmux send-keys -t $SESSION:services 'just start-services' C-m

sleep 1

# Window 1: vLLM
tmux new-window -t $SESSION -n 'vllm'
tmux send-keys -t $SESSION:vllm 'source .venv/bin/activate' C-m

sleep 1

# Window 2: Training
tmux new-window -t $SESSION -n 'train'
tmux send-keys -t $SESSION:train 'source .venv/bin/activate' C-m

# Make sure we're on the services window and attach
tmux select-window -t $SESSION:services
tmux attach-session -t $SESSION