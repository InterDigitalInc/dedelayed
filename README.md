# Disclaimer

This code is provided solely as supplementary material for the anonymized review of CVPR 2025 submission REDACTED

It is intended exclusively for the purpose of verifying the reproducibility of the results reported in the submission during the double-blind peer review process.

Distribution, reproduction, modification, or any other use beyond the CVPR 2025 review process is strictly prohibited without prior written permission from the copyright holder.

All intellectual property rights are reserved by the copyright owner. No license is granted. Provided "as is" without warranty of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, # or non-infringement.

For the anonymized review process only.

# Dedelayed

This directory contains Python code files provided as supplementary material to verify the reproducibility of the results reported in the submission. These files implement the models and training/evaluation procedures described in the paper.

## Files Overview
- **local.py**: Defines the on-device (local) model components, including convolutional blocks, attention mechanisms, and the main MSTransformer2D architecture for local inference.
- **remote.py**: Defines the remote model components, including the RemoteModel class that handles delayed remote features using EfficientViT and 3D ViT blocks.
- **fused.py**: Defines the fused model (FusedModel) that combines local and remote components for on-device correction of delayed remote inferences.
- **bdd100k_mixed_res.ipynb**: Notebook for training and evaluating the fused model on the BDD100K dataset, including data loading, collation, optimization, and mIoU computation. This is the primary entry point for reproducing experiments.
- **bdd100k_mixed_res.py**: Identical to bdd100k_mixed_res.ipynb, but readable as text without opening in jupyter and without outputs. 

All files are provided "as is" for review purposes only, as per the license notice in each file.

## Requirements
- Python 3.10+ (tested on 3.12)
- CUDA-compatible GPU for training/evaluation (e.g., NVIDIA GPU with CUDA 12+)
- Access to the BDD100K dataset.

Install dependencies using:

`pip install -r requirements.txt`

## Dataset Preparation
This code assumes access to processed BDD100K datasets in Hugging Face Datasets format.

The public release will include a link to the processed dataset on hugging face, but this is not included with the supplemental materials due to to size and anonymity constraints.

If not available locally, download and preprocess BDD100K as follows (adapt paths as needed):
1. Download BDD100K videos from the official website.
2. Preprocess into Hugging Face format with columns like 'original_{frame}', 'near_lossless_{frame}', 'label_{frame}', etc., for frames 0-14 of each video.
4. Save to disk using `datasets.Dataset.save_to_disk()`.

The validation set uses frame 14 for evaluation by default (configurable via `config.idx_eval_frame`).

## Pre-trained Checkpoints
The training script loads pre-trained checkpoints:
- `local.r480_15e.pth`: Pre-trained local model (resolution 480).
- `bdd100k_fully_remote_0_5.pth`: Pre-trained remote model (delays 0-5).

These should be placed in the working directory. If reproducing from scratch, train the individual local/remote models first (not included here due to size limit of supplemental material, but described in the paper and will be included in the public release).

## Training and Evaluation notebook (bdd100k_mixed_res.ipynb)

- Trains the fused model for 10 epochs.
- Outputs include training losses, validation mIoU, and a checkpoint (`checkpoint_mixed_res.pth`).
- Evaluation computes mIoU for delays 0-5 on the validation set.
- Plots training losses and validation miou corresponding to delay of 5 frames.

For questions during review, contact via the submission system.