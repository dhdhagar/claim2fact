# claim2fact

_(Fool Me Once: Supervised Clustering for Fact-Checked Claim Retrieval)_

---

## Overview
**TODO: Add project description**

## Setting up

- Install conda (we recommend 
[miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Create an environment and install dependencies 
    ```bash
    conda create -n claim2fact -y python=3.7 && conda activate claim2fact && pip install -r requirements.txt && conda install cython pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    ```
- Build cluster-linking special_partition function (from Cython)
    ```bash
    cd blink/biencoder/special_partition; python setup.py build_ext --inplace
    ```
- Our setup assumes GPU availability
  - The code for our paper was run using **TODO: Add details**

## Datasets

- [Learning from Fact Checkers](https://github.com/nguyenvo09/LearningFromFactCheckers): **TODO: Add description.**

## Pre-processing

**TODO: modify to call correct scripts**
```bash
# Create the entity dictionary
python blink/preprocess/medmentions_dictionary.py
# Pre-process the query mentions
python blink/preprocess/medmentions_preprocess.py
```
  
## Bi-encoder Training
- CPU debugging:
  ```bash
  python blink/biencoder/train_biencoder_mst.py --data_path=data/learnffc/processed --output_path=models/learnffc/arbo --pickle_src_path=models/learnffc --eval_interval=-1 --gold_arbo_knn=1 --rand_gold_arbo --lowercase --force_exact_search --embed_batch_size=1
  ```


## Acknowledgments

Thanks to [arboEL](https://github.com/dhdhagar/arboEL) and [BLINK](https://github.com/facebookresearch/BLINK) for the 
infrastructure of this project!
