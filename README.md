# ASR Whisper Fine-Tuning Project

## Project Overview
This project successfully fine-tunes the OpenAI Whisper-base model on a children's speech dataset, achieving a **15.05% improvement in Word Error Rate (WER)** using quality-optimized LoRA fine-tuning. The project demonstrates parameter-efficient fine-tuning with only 1.08% of total parameters trainable.

**Key Results:**
- **Baseline WER:** 124.00% → **Fine-tuned WER:** 105.34%
- **Parameter Efficiency:** 784,466 trainable parameters (1.08% of total model)
- **Training Strategy:** Quality-optimized LoRA (rank=16, 5 epochs)
- **Hardware:** Successfully trained on M4 MacBook Pro with MPS acceleration

## Setup Instructions

### Prerequisites
- Python 3.10
- Conda package manager

### Environment Setup
1. Clone the repository
2. **Download the dataset:**
   - Download the children's speech dataset from: [https://drive.google.com/file/d/1rbnQ2RFLgKBNXqzCp-EHEAh8Z5eKn_r7/view?usp=sharing](https://drive.google.com/file/d/1rbnQ2RFLgKBNXqzCp-EHEAh8Z5eKn_r7/view?usp=sharing)
   - Extract the downloaded file
   - Place the extracted `data/` folder in the project root directory
   - The structure should be: `asr_whisper/data/train/` and `asr_whisper/data/test/`
3. Create and activate a Conda environment:
   ```bash
   conda create -n asr_env python=3.10
   conda activate asr_env
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
asr_whisper/
├── data/                        # Raw dataset (download from Google Drive link above)
│   ├── train/                   # Training audio files and transcripts
│   └── test/                    # Test audio files and transcripts
├── processed_data/              # Generated during preprocessing (segmented audio clips)
│   ├── train/
│   └── test/
├── src/                         # Essential Python scripts
│   ├── preprocess.py           # Data preprocessing and segmentation
│   ├── configure_pipeline.py   # Model configuration setup
│   ├── train_simple.py         # Main training script (quality-optimized LoRA)
│   ├── finetuning_strategies.py # LoRA implementation
│   ├── baseline_evaluation.py  # Model evaluation script
│   ├── comparative_analysis.py # Results analysis
│   └── utils.py                # Helper functions and data collators
├── whisper_quality_training/    # Generated model checkpoints and LoRA adapters
├── evaluation_results.json     # Evaluation metrics and WER scores
├── comparative_analysis.json   # Comparative analysis results
├── README.md                   # This documentation
└── requirements.txt            # Python dependencies
```


## Quick Start - Reproduce Results

### Full Training Pipeline

#### 1. Data Preprocessing
```bash
# Preprocess the children's speech dataset (download data first - see setup instructions)
python src/preprocess.py
```

#### 2. Quality-Optimized Training (Local)
```bash
# Run quality-optimized LoRA fine-tuning (~15 hours on M4 MacBook Pro)
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_simple.py
```

#### 3. Evaluation and Analysis
```bash
# Run baseline and fine-tuned model evaluation
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/baseline_evaluation.py

# Generate comparative analysis
python src/comparative_analysis.py
```

## Dependencies
See `requirements.txt` for the complete list of dependencies and their versions.

## Fine-Tuning Strategies

### LoRA (Low-Rank Adaptation)



## Results

### Final Performance Metrics

| Model Configuration | Trainable Parameters | Trainable % | Test WER % |
|---------------------|---------------------|-------------|------------|
| **Whisper-Base (Baseline)** | 73,773,568 | 100.00% | **124.00%** |
| **Quality-Optimized LoRA** | 1,179,648 | 1.60% | **105.34%** |

### Key Achievements
- **15.05% WER improvement** over baseline


### Technical Details
- **LoRA Configuration:** rank=16, alpha=32, dropout=0.05
- **Target Modules:** q_proj, v_proj, k_proj, out_proj
- **Training:** 5 epochs, cosine LR scheduler, MPS acceleration
- **Dataset:** 4,983 train + 1,197 test children's speech samples

