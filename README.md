# The inevitability and superfluousness of cell types in spatial cognition

### Structure (subject to change)
```
    .
    ├── location_n_rotation_prediction.py # Downstream task training and evaluation.
    ├── border_dist_prediction.py # Downstream task training and evaluation.
    ├── lesion.py               # Lesion model units for downstream tasks.
    ├── inspect_model_units.py  # Main script for visualizing model units by known cell types in brain and by downstream tasks relevance.
    ├── data.py                 # Loading experiments related data e.g., datasets, precomputed model reprepresentations.
    ├── models.py               # Different models considered in all experiments.
    ├── unit_metric_computers.py # Various computing routines for different metrics of known spatial cells according to neuroscience literature. 
    ├── utils.py                # Utils including configuration loading etc.
    ├── configs/                # Variants of experiment settings across models, layers, moving trajectories, etc.
    ├── results/                # Intermediate experiment results/diagnostics (grouped by configs)
    ├── figs/                   # Final resulsts plotted (grouped by configs)
```

### Attribution
```
@article {Luo2024.01.10.575026,
	author = {Luo, Xiaoliang and Mok, Robert M. and Love, Bradley C.},
	title = {The inevitability and superfluousness of cell types in spatial cognition},
	year = {2024},
	doi = {10.1101/2024.01.10.575026},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/01/12/2024.01.10.575026},
	journal = {bioRxiv}
}
```
