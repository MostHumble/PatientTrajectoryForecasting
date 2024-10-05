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
