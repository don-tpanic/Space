# Space. (WIP)

### Warning: this code base is part of an active research project which is constantly evolving and subject to refactoring.

### Description
This project delves into the realm of perception-based statistical learning to unravel the mystery of how animals can effectively self-locate and navigate in space. Drawing inspiration from the current discourse surrounding the importance of path integration in navigation skills, as well as the tenuous nature of the suggested computational mechanisms underlying space-sensitive cells like grid and place cells, this project presents evidence that even deep convolutional neural networks pre-trained on object recognition inherently encode spatial information. Hopefully this project sheds light on the potential role of perception in spatial cognition and offer insights into the mechanisms driving animal navigation.

### Current state
Mapping individual DNN units to known spatial cell types in relevant brain regions and evaluating relevance of various unit types to downstream tasks (e.g., localisation and border prediction).

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
