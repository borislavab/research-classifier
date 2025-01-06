# How to run locally

1. Create and activate a conda environment:

```bash
conda create -n .conda python=3.11
conda activate .conda
```

2. Install dependencies into the conda environment:

```bash
conda install --file requirements.txt
pip install -e .
```

# Dataset analysis and decision records

Refer to the [analysis.ipynb](research_classifier/analysis/analysis.ipynb) notebook for dataset analysis conducted, helpful visualizations, alternative approaches considered and decision chosen.

# Improvements proposed

- [ ] Implement more sophisticated undersampling for multi-label classification - https://www.din.uem.br/yandre/Neurocomputing_MLTL.pdf
