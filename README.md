# claim2fact

_(Fool Me Once: Supervised Clustering for Fact-Checked Claim Retrieval)_

---

## Overview
This project addresses the task of verified claim retrieval - an important step in the information verification pipeline, which enables real-time rumour and fake news detection, thus preventing false claims from becoming "viral".

## Setting up

- Install conda (we recommend 
[miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Create an environment and install dependencies 
    ```bash
    conda create -n claim2fact -y python=3.7 && conda activate claim2fact && pip install -r requirements.txt && conda install cython pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    ```
- Build cluster-linking `special_partition` function (from Cython)
    ```bash
    cd blink/biencoder/special_partition; python setup.py build_ext --inplace
    ```
- Our setup assumes GPU availability
  - Experiments in our work were run using a single 16GB Tesla P100 GPU.

## Datasets

- [Learning from Fact Checkers](https://github.com/nguyenvo09/LearningFromFactCheckers): The FC-Tweets dataset, created by Vo and Lee (2019), contains claim tweets that each have a corresponding fact-checking URL pointing to articles from two fact-checking databases - Snopes and PolitiFact. PolitiFact rates the authenticity of claims made by politicians and other prominent personalities, while Snopes fact-checks myths and rumors. There are 73,203 claim tweets-to-fact URL mappings in the dataset.

## Data Pre-processing

**TODO: modify to call correct scripts**
```bash
# Create the entity dictionary
python blink/preprocess/medmentions_dictionary.py
# Pre-process the query mentions
python blink/preprocess/medmentions_preprocess.py
```

## Baseline Scripts (**TODO**)

## Training Scripts

### Full-Text Dataset

- In-Batch
  ```bash
  python blink/biencoder/train_biencoder.py --num_train_epochs=5 --data_path=data/learnffc/processed --output_path=models/trained/learnffc/in_batch --learning_rate=1e-05 --train_batch_size=128 --gradient_accumulation_steps=8 --eval_batch_size=16 --eval_interval=2000 --lowercase --max_seq_length=512 --max_cand_length=384 --data_parallel
  ```
- k-NN Negatives
  ```bash
  python blink/biencoder/train_biencoder_mult.py --data_path=data/learnffc/processed --pickle_src_path=models/trained/learnffc --output_path=models/trained/learnffc/knn --num_train_epochs=2 --learning_rate=1e-05 --train_batch_size=2 --eval_batch_size=16 --force_exact_search --pos_neg_loss --eval_interval=2000 --lowercase --data_parallel --max_seq_length=512 --max_cand_length=384 --knn=64
  ```
- MST-based
  ```bash
  python blink/biencoder/train_biencoder_mst.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/arbo --pickle_src_path=models/trained/learnffc --num_train_epochs=5 --learning_rate=1e-05 --train_batch_size=128 --gradient_accumulation_steps=8 --eval_batch_size=8 --force_exact_search --eval_interval=75 --max_seq_length=512 --max_cand_length=384 --lowercase --use_rand_negs --data_parallel
  ```
- 1-NN Positive
  ```bash
  python blink/biencoder/train_biencoder_mst.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/1nn --pickle_src_path=models/trained/learnffc --num_train_epochs=5 --learning_rate=1e-05 --train_batch_size=128 --gradient_accumulation_steps=8 --eval_batch_size=8 --force_exact_search --eval_interval=75 --max_seq_length=512 --max_cand_length=384 --lowercase --use_rand_negs --gold_arbo_knn=1 --data_parallel
  ```

### Summary Dataset

- In-Batch
  ```bash
  python blink/biencoder/train_biencoder.py --num_train_epochs=5 --data_path=data/learnffc/processed --output_path=models/trained/learnffc/summary/in_batch --learning_rate=1e-05 --train_batch_size=128 --gradient_accumulation_steps=16 --eval_batch_size=8 --eval_interval=2000 --lowercase --max_seq_length=512 --max_cand_length=384 --use_desc_summaries
  ```
- k-NN Negatives
  ```bash
  python blink/biencoder/train_biencoder_mult.py --data_path=data/learnffc/processed --pickle_src_path=models/trained/learnffc/summary --output_path=models/trained/learnffc/summary/knn --num_train_epochs=2 --learning_rate=1e-05 --train_batch_size=2 --eval_batch_size=16 --force_exact_search --pos_neg_loss --eval_interval=2000 --lowercase --data_parallel --max_seq_length=512 --max_cand_length=384 --knn=64 --use_desc_summaries
  ```
- MST-based
  ```bash
  python blink/biencoder/train_biencoder_mst.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/summary/arbo --pickle_src_path=models/trained/learnffc/summary --num_train_epochs=5 --learning_rate=1e-05 --train_batch_size=128 --gradient_accumulation_steps=8 --eval_batch_size=8 --force_exact_search --eval_interval=75 --max_seq_length=512 --max_cand_length=384 --lowercase --use_rand_negs --data_parallel --use_desc_summaries --save_interval=0
  ```
- 1-NN Positive
  ```bash
  python blink/biencoder/train_biencoder_mst.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/summary/1nn --pickle_src_path=models/trained/learnffc/summary --num_train_epochs=5 --learning_rate=1e-05 --train_batch_size=128 --gradient_accumulation_steps=8 --eval_batch_size=8 --force_exact_search --eval_interval=75 --max_seq_length=512 --max_cand_length=384 --lowercase --use_rand_negs --gold_arbo_knn=1 --data_parallel --use_desc_summaries --save_interval=0
  ```

## Inference Scripts

### Full-text Dataset

- In-Batch
  ```bash
  python blink/biencoder/eval_cluster_linking.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/in_batch/eval --pickle_src_path=models/trained/learnffc --path_to_model=models/trained/learnffc/in_batch/pytorch_model.bin --lowercase --recall_k=64 --data_parallel --force_exact_search --max_seq_length=512 --max_cand_length=384 --embed_batch_size=256 --data_parallel
  ```
- k-NN Negatives
  ```bash
  python blink/biencoder/eval_cluster_linking.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/knn/eval --pickle_src_path=models/trained/learnffc --path_to_model=models/trained/learnffc/knn/pytorch_model.bin --lowercase --recall_k=64 --force_exact_search --data_parallel --max_seq_length=512 --max_cand_length=384 --embed_batch_size=256
  ```
- MST-based
  ```bash
  python blink/biencoder/eval_cluster_linking.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/arbo/eval --pickle_src_path=models/trained/learnffc --path_to_model=models/trained/learnffc/arbo/pytorch_model.bin --lowercase --recall_k=64 --max_seq_length=512 --max_cand_length=384 --embed_batch_size=256 --force_exact_search --data_parallel
  ```
- 1-NN Positive
  ```bash
  python blink/biencoder/eval_cluster_linking.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/1nn/eval --pickle_src_path=models/trained/learnffc --path_to_model=models/trained/learnffc/1nn/pytorch_model.bin --lowercase --recall_k=64 --max_seq_length=512 --max_cand_length=384 --embed_batch_size=256 --force_exact_search --data_parallel
  ```
### Summary Dataset

- In-Batch
  ```bash
  python blink/biencoder/eval_cluster_linking.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/summary/in_batch/eval --pickle_src_path=models/trained/learnffc/summary --path_to_model=models/trained/learnffc/summary/in_batch/pytorch_model.bin --lowercase --recall_k=64 --data_parallel --force_exact_search --max_seq_length=512 --max_cand_length=384 --embed_batch_size=256 --use_desc_summaries
  ```
- k-NN Negatives
  ```bash
  python blink/biencoder/eval_cluster_linking.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/summary/knn/eval --pickle_src_path=models/trained/learnffc/summary --path_to_model=models/trained/learnffc/summary/knn/pytorch_model.bin --lowercase --recall_k=64 --force_exact_search --data_parallel --max_seq_length=512 --max_cand_length=384 --embed_batch_size=256 --use_desc_summaries
  ```
- MST-based
  ```bash
  python blink/biencoder/eval_cluster_linking.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/summary/arbo/eval --pickle_src_path=models/trained/learnffc/summary --path_to_model=models/trained/learnffc/summary/arbo/pytorch_model.bin --lowercase --recall_k=64 --max_seq_length=512 --max_cand_length=384 --embed_batch_size=256 --force_exact_search --data_parallel --use_desc_summaries
  ```
- 1-NN Positive
  ```bash
  python blink/biencoder/eval_cluster_linking.py --data_path=data/learnffc/processed --output_path=models/trained/learnffc/summary/1nn/eval --pickle_src_path=models/trained/learnffc/summary --path_to_model=models/trained/learnffc/summary/1nn/pytorch_model.bin --lowercase --recall_k=64 --max_seq_length=512 --max_cand_length=384 --embed_batch_size=256 --force_exact_search --data_parallel --use_desc_summaries
  ```
## Acknowledgments

Thanks to [arboEL](https://github.com/dhdhagar/arboEL) and [BLINK](https://github.com/facebookresearch/BLINK) for the infrastructure of this project!
