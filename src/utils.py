#!/usr/bin/env python3
"""
Utility functions for Whisper fine-tuning on children's speech.
Implements the data collator and preprocessing functions
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Audio
import logging

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for speech-to-text tasks with padding.
    
    This collator handles the different padding requirements for input features 
    (spectrograms) and labels (token IDs)
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need
        # different padding methods.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut it here as it's appended later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def prepare_dataset(batch, processor):
    """
    Prepare dataset function for mapping over the dataset.
    
    This function applies the feature extractor to audio and tokenizer to text
    
    Args:
        batch: A batch from the dataset
        processor: WhisperProcessor instance
        
    Returns:
        Processed batch with input_features and labels
    """
    # Load and resample audio data
    audio = batch["audio"]
    
    # Compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]  # Remove batch dimension
    
    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    
    return batch

def apply_children_speech_preprocessing(dataset, target_sampling_rate=16000):
    """
    Apply specialized preprocessing for children's speech.
    
    This includes:
    1. Resampling to 16kHz
    2. Audio normalization 
    3. Duration filtering
    4. Optional speed perturbation for data augmentation
    
    Args:
        dataset: HuggingFace dataset
        target_sampling_rate: Target sampling rate (default: 16000 for Whisper)
        
    Returns:
        Preprocessed dataset
    """
    logging.info("Applying children's speech preprocessing...")
    
    # 1. Resampling to 16kHz using datasets library
    logging.info(f"Resampling audio to {target_sampling_rate}Hz")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
    
    # 2. Audio normalization and duration filtering
    def normalize_and_filter(batch):
        """Normalize audio and filter by duration."""
        audio = batch["audio"]
        audio_array = np.array(audio["array"])
        
        # Normalize to zero mean and unit variance
        if len(audio_array) > 0:
            audio_array = (audio_array - np.mean(audio_array)) / (np.std(audio_array) + 1e-8)
            
        # Calculate duration in seconds
        duration = len(audio_array) / audio["sampling_rate"]
        
        # Update the audio array with normalized version
        batch["audio"]["array"] = audio_array.tolist()
        batch["duration"] = duration
        
        return batch
    
    # Apply normalization
    logging.info("Normalizing audio and calculating durations")
    dataset = dataset.map(normalize_and_filter)
    
    # 3. Duration filtering (remove too short < 1s or too long > 30s)
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: 1.0 <= x["duration"] <= 30.0)
    filtered_size = len(dataset)
    
    logging.info(f"Duration filtering: {original_size} -> {filtered_size} samples "
                f"({original_size - filtered_size} removed)")
    
    return dataset

def create_speed_perturbation_dataset(dataset, speed_factors=[0.9, 1.1]):
    """
    Create speed-perturbed versions of the dataset for data augmentation.
    
    As recommended this creates additional copies of training data
    with slightly different speeds to improve model robustness and simulate
    variations in children's speaking rates.
    
    Args:
        dataset: Original dataset
        speed_factors: List of speed multiplication factors
        
    Returns:
        Augmented dataset with speed perturbations
    """
    import librosa
    
    logging.info(f"Creating speed perturbation with factors: {speed_factors}")
    
    augmented_samples = []
    
    def speed_perturb_sample(sample, speed_factor):
        """Apply speed perturbation to a single sample."""
        audio_array = np.array(sample["audio"]["array"])
        sr = sample["audio"]["sampling_rate"]
        
        # Apply speed perturbation using librosa
        perturbed_audio = librosa.effects.time_stretch(audio_array, rate=speed_factor)
        
        # Create new sample
        new_sample = sample.copy()
        new_sample["audio"] = {
            "array": perturbed_audio.tolist(),
            "sampling_rate": sr
        }
        # Update duration
        new_sample["duration"] = len(perturbed_audio) / sr
        
        return new_sample
    
    # Create perturbed versions
    for speed_factor in speed_factors:
        logging.info(f"Creating {speed_factor}x speed variants...")
        for sample in dataset:
            try:
                perturbed_sample = speed_perturb_sample(sample, speed_factor)
                augmented_samples.append(perturbed_sample)
            except Exception as e:
                logging.warning(f"Failed to create speed perturbation for sample: {e}")
    
    # Combine original and augmented data
    from datasets import Dataset
    if augmented_samples:
        augmented_dataset = Dataset.from_list(augmented_samples)
        # Concatenate with original dataset
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets([dataset, augmented_dataset])
        logging.info(f"Dataset augmented: {len(dataset)} -> {len(combined_dataset)} samples")
        return combined_dataset
    else:
        logging.warning("No augmented samples created, returning original dataset")
        return dataset

def compute_metrics(pred, processor, wer_metric):
    """
    Compute evaluation metrics (WER)
    
    Args:
        pred: Predictions from the model
        processor: WhisperProcessor instance
        wer_metric: WER metric from evaluate library
        
    Returns:
        Dictionary with computed metrics
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def log_dataset_statistics(dataset, split_name="dataset"):
    """
    Log useful statistics about the dataset.
    
    Args:
        dataset: HuggingFace dataset
        split_name: Name of the dataset split for logging
    """
    logging.info(f"\n=== {split_name.upper()} STATISTICS ===")
    logging.info(f"Total samples: {len(dataset)}")
    
    if "duration" in dataset.column_names:
        durations = dataset["duration"]
        logging.info(f"Duration statistics:")
        logging.info(f"  Mean: {np.mean(durations):.2f}s")
        logging.info(f"  Median: {np.median(durations):.2f}s")
        logging.info(f"  Min: {np.min(durations):.2f}s")
        logging.info(f"  Max: {np.max(durations):.2f}s")
        logging.info(f"  Total: {np.sum(durations)/3600:.2f} hours")
    
    if "transcription" in dataset.column_names:
        transcriptions = dataset["transcription"]
        word_counts = [len(text.split()) for text in transcriptions]
        logging.info(f"Transcription statistics:")
        logging.info(f"  Mean words: {np.mean(word_counts):.1f}")
        logging.info(f"  Median words: {np.median(word_counts):.1f}")
        logging.info(f"  Min words: {np.min(word_counts)}")
        logging.info(f"  Max words: {np.max(word_counts)}")
    
    logging.info("=" * 40)
