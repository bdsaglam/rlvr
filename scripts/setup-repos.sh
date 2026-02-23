#! /bin/bash

REPOS_DIR=./tmp/repos

clone_or_pull() {
    local name=$1
    local url=$2
    local dest=$REPOS_DIR/$name
    echo "Cloning or pulling $name from $url to $dest"
    if [ ! -d "$dest" ]; then
        git clone "$url" "$dest"
    else
        git -C "$dest" pull
    fi
}

mkdir -p $REPOS_DIR

clone_or_pull dspy              git@github.com:stanfordnlp/dspy.git
clone_or_pull verifiers         git@github.com:PrimeIntellect-ai/verifiers.git
clone_or_pull prime-rl          git@github.com:PrimeIntellect-ai/prime-rl.git
clone_or_pull OpenTinker        git@github.com:open-tinker/OpenTinker.git
clone_or_pull trl               git@github.com:huggingface/trl.git
clone_or_pull sdpo              git@github.com:lasgroup/SDPO.git
clone_or_pull verl              git@github.com:verl-project/verl.git
clone_or_pull verl-recipe       git@github.com:verl-project/verl-recipe.git
