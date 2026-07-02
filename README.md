# CLAS

Official Python implementation of the manuscript:

**A Gravitational-Potential-Field-Inspired Density-Based Clustering Algorithm**  
Yongxu Chen, Mei Chen, Lipeng Wei, and Chi Zhang

CLAS (**CL**ustering **A**lgorithm **S**upported by gravitational potential field theory) is a density-based clustering method designed to identify clusters with arbitrary shapes, varying densities, and different sizes. The method combines an adaptive density measure with mutual nearest neighbors to construct cluster backbones and then assigns the remaining points according to nearby higher-density points.

## Repository structure

```text
CLAS/
├── Main.py                 # CLAS implementation and an example on Pendigits
├── CLAS_text.py            # Experiment on the Reuters-21578 text dataset
├── datasets/               # Benchmark datasets that can be redistributed
│   ├── two_dim/
│   ├── three_dim/
│   └── multi_dim/
├── results/                # Selected experimental results
├── util/                   # Data-loading and visualization utilities
├── requirements.txt
└── LICENSE
```

Datasets whose redistribution conditions are unclear are not included in this repository.

## Requirements

- Python 3.10 is recommended.

Install the required packages with:

```bash
pip install -r requirements.txt
```

The code was tested with Python 3.10. Package versions may vary depending on the local environment.

## Quick start

Run the example from the repository root:

```bash
python Main.py
```

The default example loads the Pendigits dataset and runs CLAS with:

```python
k = 10
l = 19
```

The program reports the running time, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and Purity.

A typical call is:

```python
ARI, NMI, result, purity = cluster_with_parameters(
    dataset,
    labels_true,
    k=10,
    l=19,
)
```

Here, `k` is the number of nearest neighbors used for adaptive density estimation, and `l` is the number of nearest neighbors used to construct mutual nearest-neighbor relationships.

## Input format

The example datasets used by `Main.py` are tab-separated text files. The first column contains the ground-truth label, and the remaining columns contain feature values.

```python
from numpy import genfromtxt

file_name = "datasets/multi_dim/pendigits.in"
data = genfromtxt(file_name, delimiter="\t")
labels_true = data[:, 0]
dataset = data[:, 1:]
```

## Reuters-21578 experiment

`CLAS_text.py` downloads the ModApte version of Reuters-21578 through the Hugging Face `datasets` package, retains single-label documents, and constructs TF-IDF features using unigrams and bigrams.

Run it with:

```bash
python CLAS_text.py
```

This experiment downloads data from the internet and can require substantial memory and running time. Results are written to `results/reuters21578_results.txt`.

## Reproducibility notes

- Run the scripts from the repository root so that the relative dataset and result paths are resolved correctly.
- Euclidean distance is used in `Main.py` for conventional numerical datasets.
- Cosine distance is used in `CLAS_text.py` for sparse text features.
- The parameter settings reported in the manuscript are dataset-specific and are summarized in the manuscript.

## Citation

The manuscript is currently under review. Please cite it as:

```bibtex
@article{chen_clas,
  title   = {A Gravitational-Potential-Field-Inspired Density-Based Clustering Algorithm},
  author  = {Chen, Yongxu and Chen, Mei and Wei, Lipeng and Zhang, Chi},
  journal = {Pattern Recognition},
  note    = {Manuscript under review}
}
```

The citation information will be updated after publication.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Mei Chen (corresponding author)  
School of Electronic and Information Engineering, Lanzhou Jiaotong University  
Email: mei.chen.lzjtu@hotmail.com
