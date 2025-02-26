# Patient Trajectory Prediction with Clinical Notes Integration

This repository contains the code and resources for our research on patient trajectory prediction, integrating both structured (CCS codes) and unstructured (clinical notes) data using advanced deep learning techniques.

## Project Structure

- `cross_val_train_ddp.py`: Script for cross-validation training of the Transformer encoder-decoder model with integrated clinical notes, using distributed data parallelism.
- `model.py`: Contains the main model architecture.
- `prepare_notes.py`: Script for preprocessing clinical notes.
- `pretrain_bert.py`: Script for pretraining Clinical Mosaic, our custom BERT model.
- `train.py`: Vanilla Transformer-based training without clinical notes or distributed data parallelism.
- `train_ddp.py`: Training script for the Transformer model with injected notes using distributed data parallelism.
- `train_with_notes.py`: Training script for the Transformer model with injected notes without distributed data parallelism.

### Subdirectories

- `literature_models/`: Contains notebooks and scripts to reproduce results from various models in the literature using our cross-validation folds.
- `notebooks/`: Jupyter notebooks for data preparation, model training, and evaluation.
  - `MedNLI.ipynb`: Notebook for reporting MedNLI results of Clinical Mosaic.
  - `prepare_data.ipynb`: Notebook to prepare the data (after preparing the notes)
- `stats/`: Utilities for visualization and statistical analysis.
- `tests/`: Test scripts to ensure correct preprocessing of notes, especially when using multiple processes/threads.
- `utils/`: Various utility functions and modules for data processing, model training, and evaluation.

## Setup

1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```

2. Set up the environment:
   ```
   conda env create -f environment.yml
   conda activate [env-name]
   ```

   Alternatively, if you prefer using pip:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the clincal notes:
   ```
   python prepare_notes.py
   ```

2. Pretrain Clinical Mosaic:
   ```
   python pretrain_bert.py
   ```
3. Prepare the data:
   - Run the notebook located at `notebooks/prepare_data.ipynb`.
4. Train the model:
   - For vanilla training:
     ```
     python train.py
     ```
   - For training with notes:
     ```
     python train_with_notes.py
     ```
   - For distributed training with notes:
     ```
     python train_ddp.py
     ```

5. Evaluate the model:
   - Use the notebooks in the `notebooks/` directory for detailed evaluation and analysis.

## Reproducing Literature Results

To reproduce results from other models in the literature:

1. Navigate to the `literature_models/` directory.
2. Run the corresponding notebook or script for the model you wish to reproduce.

## Visualization

Various visualization scripts and utilities are available in the `stats/` directory. Use these to generate plots and analyze the results.

## Acknowledgments

The project leading to this publication has received funding from the Excellence Initiative of Aix Marseille Université - A*Midex, a French “Investissements d’Avenir programme” AMX-21-IET-017.

We would like to thank **LIS** | Laboratoire d'Informatique et Systèmes, Aix-Marseille University for providing the GPU resources necessary for pretraining and conducting extensive experiments. Additionally, we acknowledge **CEDRE** | CEntre de formation et de soutien aux Données de la REcherche, Programme 2 du projet France 2030 IDeAL for supporting early-stage experiments and hosting part of the computational infrastructure.

## Citation

**BibTeX:**

```bibtex
@misc{klioui2025patienttrajectorypredictionintegrating,
      title={Patient Trajectory Prediction: Integrating Clinical Notes with Transformers}, 
      author={Sifal Klioui and Sana Sellami and Youssef Trardi},
      year={2025},
      eprint={2502.18009},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.18009}, 
}

@article{RNTI/papers/1002990,
  author    = {Sifal Klioui and Sana Sellami and Youssef Trardi},
  title     = {Prédiction de la trajectoire du patient : Intégration des notes cliniques aux transformers},
  journal = {Revue des Nouvelles Technologies de l'Information},
  volume = {Extraction et Gestion des Connaissances, RNTI-E-41},
  year      = {2025},
  pages     = {135-146}
}
```

## More Information

For further details, please refer to the model’s repository and supplementary documentation.
