#!/bin/sh

dvc exp run --queue \
    -S agent='ragent' \
    -S model.path='/home/ubuntu/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250603_205328-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='1.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='1' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

dvc exp run --queue \
    -S agent='ragent' \
    -S model.path='/home/ubuntu/repos/verifiers/outputs/Llama-3.1-8B-Instruct-ragent-20250603_205328-merged' \
    -S model.temperature='0.5' \
    -S model.top_p='0.95' \
    -S model.few_shot_prob='0.0' \
    -S retriever.name='hybrid-tei' \
    -S retriever.top_k='1' \
    -S repeat='1' \
    -S run='1' \
    -S dataset.path='bdsaglam/musique' \
    -S dataset.name='answerable' \
    -S dataset.split='validation' \
    -S devices='"0"'

