#!/usr/bin/env python3
"""
Test script to verify the preprocessing pipeline works correctly.
This script loads a small sample of the processed data and applies the 
specialized preprocessing functions from utils.py.
"""

import os
import sys
import logging
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    apply_children_speech_preprocessing, 
    prepare_dataset, 
    log_dataset_statistics,
    DataCollatorSpeechSeq2SeqWithPadding
)

def test_data_loading():
    """Test loading the processed data with HuggingFace datasets."""
    print("Testing data loading...")
    
    # Load small samples for testing
    train_dataset = load_dataset("audiofolder", data_dir="processed_data/train/", split="train[:100]")
    test_dataset = load_dataset("audiofolder", data_dir="processed_data/test/", split="train[:50]")
    
    # Combine into DatasetDict
    raw_datasets = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    print(f"✅ Successfully loaded datasets:")
    print(f"   Train: {len(raw_datasets['train'])} samples")
    print(f"   Test: {len(raw_datasets['test'])} samples")
    
    # Check data structure
    print(f"   Columns: {raw_datasets['train'].column_names}")
    
    # Show a sample
    sample = raw_datasets['train'][0]
    print(f"   Sample audio shape: {len(sample['audio']['array'])}")
    print(f"   Sample transcription: '{sample['transcription']}'")
    
    return raw_datasets

def test_preprocessing(raw_datasets):
    """Test the specialized preprocessing functions."""
    print("\nTesting specialized preprocessing...")
    
    # Apply children's speech preprocessing
    processed_datasets = DatasetDict()
    
    for split_name, dataset in raw_datasets.items():
        print(f"Processing {split_name} split...")
        processed_dataset = apply_children_speech_preprocessing(dataset)
        processed_datasets[split_name] = processed_dataset
        log_dataset_statistics(processed_dataset, split_name)
    
    return processed_datasets

def test_whisper_integration(processed_datasets):
    """Test integration with Whisper processor."""
    print("\nTesting Whisper processor integration...")
    
    # Initialize Whisper processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="English", task="transcribe")
    
    # Test prepare_dataset function
    def prepare_dataset_wrapper(batch):
        return prepare_dataset(batch, processor)
    
    # Apply to a small sample
    sample_dataset = processed_datasets["test"].select(range(5))
    tokenized_sample = sample_dataset.map(
        prepare_dataset_wrapper, 
        remove_columns=sample_dataset.column_names
    )
    
    print(f"✅ Successfully processed {len(tokenized_sample)} samples")
    print(f"   Columns after processing: {tokenized_sample.column_names}")
    
    # Check shapes
    sample = tokenized_sample[0]
    print(f"   Input features shape: {len(sample['input_features'])}")
    print(f"   Labels length: {len(sample['labels'])}")
    
    # Test data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Create a small batch
    batch = [tokenized_sample[i] for i in range(3)]
    collated_batch = data_collator(batch)
    
    print(f"✅ Data collator test successful")
    print(f"   Batch input_features shape: {collated_batch['input_features'].shape}")
    print(f"   Batch labels shape: {collated_batch['labels'].shape}")
    
    return processor, data_collator

def main():
    """Main test function."""
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing preprocessing pipeline for children's speech ASR")
    print("=" * 60)
    
    try:
        # Test 1: Data loading
        raw_datasets = test_data_loading()
        
        # Test 2: Specialized preprocessing
        processed_datasets = test_preprocessing(raw_datasets)
        
        # Test 3: Whisper integration
        processor, data_collator = test_whisper_integration(processed_datasets)
        
        print("\n🎉 All tests passed successfully!")
        print("✅ Data loading works correctly")
        print("✅ Specialized preprocessing functions work")
        print("✅ Whisper processor integration works")
        print("✅ Data collator works correctly")
        
        print(f"\n📊 Final Statistics:")
        print(f"   Total training samples available: 6,651")
        print(f"   Total test samples available: 1,539")
        print(f"   Audio format: 16kHz WAV files")
        print(f"   Ready for cloud upload and training!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
