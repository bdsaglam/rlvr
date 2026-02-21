#! /bin/bash

REPOS_DIR=./tmp/repos

mkdir -p $REPOS_DIR

# Clone or fetch verifiers
if [ ! -d $REPOS_DIR/verifiers ]; then
    git clone https://github.com/PrimeIntellect-ai/verifiers.git $REPOS_DIR/verifiers
else
    git -C $REPOS_DIR/verifiers pull
fi

# Clone or fetch prime-rl
if [ ! -d $REPOS_DIR/prime-rl ]; then
    git clone https://github.com/PrimeIntellect-ai/prime-rl.git $REPOS_DIR/prime-rl
else
    git -C $REPOS_DIR/prime-rl pull
fi

# Clone or fetch trl
if [ ! -d $REPOS_DIR/trl ]; then
    git clone https://github.com/huggingface/trl.git $REPOS_DIR/trl
else
    git -C $REPOS_DIR/trl pull
fi
