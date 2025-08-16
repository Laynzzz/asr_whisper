# ASR Whisper Fine-Tuning Project

## Project Overview
This project fine-tunes the OpenAI Whisper-base model on a children's speech dataset to improve Word Error Rate (WER) performance. The project implements and compares two fine-tuning strategies: Parameter-Efficient Fine-Tuning with LoRA and selective layer fine-tuning.

## Setup Instructions

### Prerequisites
- Python 3.10
- Conda package manager
- Access to a cloud GPU instance for training

### Environment Setup
1. Clone the repository
2. Create and activate a Conda environment:
   ```bash
   conda create -n asr_env python=3.10
   conda activate asr_env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
asr_whisper/
├── data/                 # Raw data as provided (train/ and test/ subdirectories)
├── processed_data/       # Segmented audio clips and manifest files
│   ├── train/
│   └── test/
├── src/                  # All Python source code
│   ├── 01_preprocess.py  # Script for data segmentation and preparation
│   ├── 02_train.py       # Main script for model training and evaluation
│   └── utils.py          # Helper functions, data collators, etc.
├── models/               # Saved model checkpoints and LoRA adapters
├── results/              # Evaluation metrics (e.g., WER scores), logs, learning curves
├── scripts/              # Optional: Bash scripts for automating experiment runs
├── README.md             # Comprehensive project documentation
└── requirements.txt      # Python package dependencies
```

## Workflow

### 1. Preprocess Data Locally
Run the segmentation script on your local machine:
```bash
python src/01_preprocess.py
```

### 2. Configure AWS CLI (Section 4 Setup)
Configure AWS CLI with your credentials:
```bash
# Check current AWS configuration status
./scripts/setup_aws.sh

# Configure AWS CLI (you'll be prompted for credentials)
aws configure
```

### 3. Upload Data to AWS S3 (Section 4)
Upload the processed data to the specified S3 bucket:
```bash
# Upload processed data to S3 (as specified in plan)
./scripts/upload_to_s3.sh
```

This will execute: `aws s3 sync processed_data/ s3://asr-finetuning-data-2025/processed_data/`

### 4. Train Model on Cloud GPU (Sections 5-8)
On your cloud VM, download the data and run the training script:

**Download data from S3:**
```bash
aws s3 sync s3://asr-finetuning-data-2025/processed_data/ ./processed_data/
```

**For LoRA fine-tuning:**
```bash
python src/02_train.py --strategy lora
```

**For selective layer fine-tuning:**
```bash
python src/02_train.py --strategy selective_tuning
```

## Reproducing Evaluation
To reproduce the Word Error Rate (WER) reported in the technical report, run the training script with the `--evaluate_only` flag and provide the path or Hugging Face Hub ID to the fine-tuned model:

```bash
python src/02_train.py --strategy lora --evaluate_only --model_checkpoint <model_path_or_hub_id>
```

## Dependencies
See `requirements.txt` for the complete list of dependencies and their versions.

## Fine-Tuning Strategies

### Strategy A: LoRA (Low-Rank Adaptation)
- Parameter-efficient fine-tuning approach
- Freezes pre-trained weights and adds trainable low-rank matrices
- Targets query and value projection matrices in attention layers
- Significantly reduces trainable parameters (~0.5-1% of total)

### Strategy B: Selective Layer Fine-Tuning
- Unfreezes specific layers for full fine-tuning
- Targets top encoder layers and all decoder layers
- More computationally intensive but allows expressive updates
- Approximately 30-40% of parameters trainable

## Results
Results and comparative analysis will be documented in the technical report after training completion.
