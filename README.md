# Machine Learning & Deep Learning mini-projects

A collection of small, complete projects that I build while I learn machine learning and deep learning with **PyTorch**.

Each project is in its own folder and has:

- a **notebook** with commented code,
- a **README** with the goal, the method, the results and what I learned,
- **figures** of the results (learning curves, confusion matrix, ...).

## Projects

| # | Project | Task | Model | Main result |
|---|---|---|---|---|
| 01 | [Fashion-MNIST classification](01-fashion-mnist-mlp/) | Image classification (10 classes) | MLP with BatchNorm and Dropout | 90.60% accuracy |

More projects will be added.

## Repository structure

```
ml-dl-mini-projects/
├── README.md            <- this file
├── requirements.txt     <- Python packages for all projects
├── .gitignore           <- datasets and temporary files are not stored
└── 01-fashion-mnist-mlp/
    ├── README.md
    ├── photo_classification.ipynb
    └── images/
```

## Setup

```bash
git clone https://github.com/kouroshov/ml-dl-mini-projects.git
cd ml-dl-mini-projects
pip install -r requirements.txt
```

Then open the notebook of a project with Jupyter or VS Code. Datasets are downloaded automatically when you run the notebook.

## Tools

Python, PyTorch, torchvision, NumPy, Matplotlib, seaborn, scikit-learn, Jupyter
