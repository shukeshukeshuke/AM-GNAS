# Adaptive multi-scale Graph Neural Architecture Search framework

## Getting Started

### 0. Prerequisites

+ Linux
+ NVIDIA GPU + CUDA CuDNN 

### 1. Setup Python Environment
```python
# clone Github repo
conda install git
git clone https://github.com/shukeshukeshuke/AM-GNAS.git
cd AM-GNAS

# Install python environment
conda env create -f environment.yml
conda activate amgnas
```
### 2. Download datasets

The datasets are provided by project [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns), you can click [here](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/02_download_datasets.md) to download all the required datasets. 

### 3. Search Architectures
```python
sh scripts/search_molecules_zinc.sh [gpu_id]
```
### 4. Train & Test
```python
sh scripts/train_molecules_zinc.sh [gpu_id] '[path_to_genotypes]/example.yaml'
```
## Reference
```
@article{yang2024adaptive,
  title={Adaptive multi-scale Graph Neural Architecture Search framework},
  author={Yang, Lintao and Liò, Pietro and Shen, Xu and Zhang, Yuyang and Peng, Chengbin},
  journal={Neurocomputing},
  pages={128094},
  year={2024},
  publisher={Elsevier}
}

```

