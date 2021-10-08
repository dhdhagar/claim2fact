# claim2fact

_(Fool Me Once: Supervised Clustering for Fact-Checked Claim Retrieval)_

---

Thanks to [arboEL](https://github.com/dhdhagar/arboEL) and [BLINK](https://github.com/facebookresearch/BLINK) for the 
infrastructure of this project!

## Overview

## Setting up

- Install conda (we recommend 
[miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Create an environment and install dependencies 
    ```bash
    conda create -n blink37 -y python=3.7 && conda activate blink37 && pip install -r requirements.txt && conda install cython pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    ```
- Build cluster-linking special_partition function (from Cython)
    ```bash
    cd blink/biencoder/special_partition; python setup.py build_ext --inplace
    ```
- Our setup assumes GPU availability
  - The code for our paper was run using 2 NVIDIA Quadro RTX 8000

## Datasets

- [MedMentions](https://github.com/chanzuckerberg/MedMentions) (full): The MedMentions corpus
consists of 4,392 papers (Titles and Abstracts) randomly selected from among papers 
released on PubMed in 2016, that were in the biomedical field, published in the 
English language, and had both a Title and an Abstract.
- [ZeShEL](https://github.com/lajanugen/zeshel): The Zero Shot Entity Linking dataset 
was constructed using multiple sub-domains in Wikia from FANDOM with automatically 
extracted labeled mentions using hyper-links.

## Pre-processing

- For MedMentions
  ```bash
  # Create the entity dictionary
  python blink/preprocess/medmentions_dictionary.py
  # Pre-process the query mentions
  python blink/preprocess/medmentions_preprocess.py
  ```
- For ZeShEL
  ```bash
  # Create the entity dictionary
  python blink/preprocess/zeshel_dictionary.py
  # Pre-process the query mentions
  python blink/preprocess/zeshel_preprocess.py
  ```
  
## Bi-encoder Training

### MST
Example command for MedMentions
```bash
python blink/biencoder/train_biencoder_mst.py --bert_model=models/biobert-base-cased-v1.1 --data_path=data/medmentions/processed --output_path=models/trained/medmentions_mst/pos_neg_loss/no_type --pickle_src_path=models/trained/medmentions --num_train_epochs=5 --train_batch_size=128 --gradient_accumulation_steps=4 --eval_interval=10000 --pos_neg_loss --force_exact_search --embed_batch_size=3500 --data_parallel
```

### k-NN negatives
Example command for MedMentions
```bash
python blink/biencoder/train_biencoder_mult.py --bert_model=models/biobert-base-cased-v1.1 --data_path=data/medmentions/processed --output_path=models/trained/medmentions/pos_neg_loss/no_type --pickle_src_path=models/trained/medmentions --num_train_epochs=5 --train_batch_size=128 --gradient_accumulation_steps=4 --eval_interval=10000 --pos_neg_loss --force_exact_search --embed_batch_size=3500 --data_parallel
```

### In-batch negatives
Example command for MedMentions
```bash
python blink/biencoder/train_biencoder.py --bert_model=models/biobert-base-cased-v1.1 --num_train_epochs=5 --data_path=data/medmentions/processed --output_path=models/trained/medmentions_blink --data_parallel --train_batch_size=128 --eval_batch_size=128 --eval_interval=10000
```

## Bi-encoder Inference

### Linking
Example command for MedMentions
```bash
python blink/biencoder/eval_cluster_linking.py --bert_model=models/biobert-base-cased-v1.1 --data_path=data/medmentions/processed --output_path=models/trained/medmentions_mst/eval/pos_neg_loss/no_type/wo_type --pickle_src_path=models/trained/medmentions/eval --path_to_model=models/trained/medmentions_mst/pos_neg_loss/no_type/epoch_best_5th/pytorch_model.bin --recall_k=64 --embed_batch_size=3500 --force_exact_search --data_parallel
```

### Discovery
Example command for MedMentions
```bash
python blink/biencoder/eval_entity_discovery.py --bert_model=models/biobert-base-cased-v1.1 --data_path=data/medmentions/processed --output_path=models/trained/medmentions_mst/eval/pos_neg_loss/directed --pickle_src_path=models/trained/medmentions/eval --embed_data_path=models/trained/medmentions_mst/eval/pos_neg_loss --use_types --force_exact_search --graph_mode=directed --exact_threshold=127.87733985396665 --exact_knn=8 --data_parallel
```
