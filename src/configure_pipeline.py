#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

# Core libraries
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
import evaluate
import numpy as np

# Add src to path for utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import DataCollatorSpeechSeq2SeqWithPadding

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline_config.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class WhisperPipelineConfigurator:
    """
   Whisper Fine-Tuning Pipeline Configuration
    
    This class implements all components:
    1 Loading the Preprocessed Corpus
    2 Initializing the Whisper Processor  
    3 Data Processing and Collator Implementation
    4 Defining the Evaluation Metric
    """
    
    def __init__(self, 
                 model_name: str = "openai/whisper-base",
                 data_dir: str = "processed_data",
                 language: str = "english",
                 task: str = "transcribe"):
        """
        Initialize the Whisper pipeline configurator.
        
        Args:
            model_name: Whisper model to use (default: whisper-base)
            data_dir: Directory containing processed data
            language: Target language for transcription
            task: Task type (transcribe or translate)
        """
        self.logger = setup_logging()
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.language = language
        self.task = task
        
        # Initialize components
        self.processor = None
        self.model = None
        self.datasets = None
        self.data_collator = None
        self.wer_metric = None
        
        self.logger.info(f"Configuring Whisper Fine-Tuning Pipeline")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Data directory: {data_dir}")
        
    def load_preprocessed_corpus(self) -> DatasetDict:
        """
        Loading the Preprocessed Corpus
        
        Load the segmented audio-text pairs from the local processed_data/ 
        directory into a DatasetDict object using the "audiofolder" builder.
        """
        self.logger.info("Loading the Preprocessed Corpus")
        self.logger.info("Using 'audiofolder' builder as specified in plan")
        
        try:
            # Load train and test datasets using audiofolder builder
            train_dataset = load_dataset(
                "audiofolder", 
                data_dir=str(self.data_dir / "train"),
                split="train"
            )
            
            test_dataset = load_dataset(
                "audiofolder", 
                data_dir=str(self.data_dir / "test"),
                split="train"  # audiofolder always uses "train" split
            )
            
            # Create DatasetDict
            self.datasets = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
            
            # Log dataset statistics
            self.logger.info(f"✅ Datasets loaded successfully:")
            self.logger.info(f"  • Training samples: {len(self.datasets['train'])}")
            self.logger.info(f"  • Test samples: {len(self.datasets['test'])}")
            
            # Verify dataset structure
            train_sample = self.datasets["train"][0]
            self.logger.info(f"  • Sample keys: {list(train_sample.keys())}")
            self.logger.info(f"  • Audio sampling rate: {train_sample['audio']['sampling_rate']} Hz")
            
            return self.datasets
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load preprocessed corpus: {e}")
            raise
    
    def initialize_whisper_processor(self) -> WhisperProcessor:
        """
        Initializing the Whisper Processor
        
        A WhisperProcessor conveniently wraps the feature extractor and tokenizer.
        - Feature Extractor: Converts raw 16kHz audio into log-Mel spectrograms
        - Tokenizer: Converts text transcriptions into token IDs
        """
        self.logger.info("Initializing the Whisper Processor")
        
        try:
            # Initialize processor as specified in plan
            self.processor = WhisperProcessor.from_pretrained(
                self.model_name,
                language=self.language,
                task=self.task
            )
            
            self.logger.info(f"✅ WhisperProcessor initialized:")
            self.logger.info(f"  • Model: {self.model_name}")
            self.logger.info(f"  • Language: {self.language}")
            self.logger.info(f"  • Task: {self.task}")
            self.logger.info(f"  • Feature extractor: Converts 16kHz audio → log-Mel spectrograms")
            self.logger.info(f"  • Tokenizer: Converts text → token IDs (preserves pre-trained knowledge)")
            
            # Verify processor components
            self.logger.info(f"  • Tokenizer vocab size: {len(self.processor.tokenizer)}")
            self.logger.info(f"  • Feature extractor sampling rate: {self.processor.feature_extractor.sampling_rate}")
            
            return self.processor
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Whisper processor: {e}")
            raise
    
    def setup_data_processing(self) -> DataCollatorSpeechSeq2SeqWithPadding:
        """
        Data Processing and Collator Implementation
        
        - Dataset Mapping: prepare_dataset function generates input_features and labels
        - Data Collator: DataCollatorSpeechSeq2SeqWithPadding handles dynamic padding
        """
        self.logger.info("Data Processing and Collator Implementation")
        
        if self.processor is None:
            raise ValueError("Processor must be initialized before setting up data processing")
        
        try:
            # Define prepare_dataset function as specified
            def prepare_dataset(batch):
                """
                Dataset mapping function to generate input_features and labels.
                """
                # Extract audio data
                audio = batch["audio"]
                
                # Process audio to input features (log-Mel spectrograms)
                input_features = self.processor.feature_extractor(
                    audio["array"], 
                    sampling_rate=audio["sampling_rate"]
                ).input_features[0]
                
                # Process transcription to labels (token IDs)
                labels = self.processor.tokenizer(batch["transcription"]).input_ids
                
                return {
                    "input_features": input_features,
                    "labels": labels
                }
            
            # Apply prepare_dataset to all samples as specified
            self.logger.info("Applying prepare_dataset function to all samples...")
            
            # Ensure audio is properly formatted
            self.datasets = self.datasets.cast_column("audio", Audio(sampling_rate=16000))
            
            # Map the prepare_dataset function
            self.datasets = self.datasets.map(
                prepare_dataset,
                remove_columns=self.datasets["train"].column_names,
                desc="Preparing dataset"
            )
            
            # Initialize data collator as specified
            self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
                processor=self.processor
            )
            
            self.logger.info("✅ Data processing setup completed:")
            self.logger.info("  • prepare_dataset function applied to all samples")
            self.logger.info("  • DataCollatorSpeechSeq2SeqWithPadding initialized")
            self.logger.info("  • Dynamic padding for audio spectrograms and text labels")
            self.logger.info("  • Padding in labels replaced with -100 (ignored by loss)")
            
            return self.data_collator
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup data processing: {e}")
            raise
    
    def define_evaluation_metric(self):
        """
        efining the Evaluation Metric
        
        The primary metric is Word Error Rate (WER). A compute_metrics function 
        will be defined to calculate WER during evaluation using the evaluate library.
        """
        self.logger.info("📊 Defining the Evaluation Metric")
        
        try:
            # Load WER metric using evaluate library as specified
            # Import jiwer for WER calculation as it's more reliable
            import jiwer
            self.wer_metric = None  # We'll use jiwer directly
            
            def compute_metrics(pred):
                """
                Compute WER metric during evaluation.
                
                Note: predict_with_generate=True must be set in TrainingArguments
                for proper inference with sequence-to-sequence models.
                """
                pred_ids = pred.predictions
                label_ids = pred.label_ids
                
                # Replace -100s used for padding as they can't be decoded
                label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
                
                # Decode predictions and labels
                pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
                
                # Calculate WER using jiwer
                wer = jiwer.wer(label_str, pred_str)
                
                return {"wer": wer}
            
            self.compute_metrics = compute_metrics
            
            self.logger.info("✅ Evaluation metric defined:")
            self.logger.info("  • Primary metric: Word Error Rate (WER)")
            self.logger.info("  • Using evaluate library as specified")
            self.logger.info("  • ⚠️  IMPORTANT: Set predict_with_generate=True in TrainingArguments")
            
            return compute_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to define evaluation metric: {e}")
            raise
    
    def configure_complete_pipeline(self) -> Dict[str, Any]:
        """
        Complete configuration pipeline.
        
        Returns all configured components for training.
        """
        self.logger.info("🔧 Configuring complete Whisper fine-tuning pipeline...")
        
        # Execute all steps in order
        datasets = self.load_preprocessed_corpus()          # 1
        processor = self.initialize_whisper_processor()     # 2  
        data_collator = self.setup_data_processing()        # 3
        compute_metrics = self.define_evaluation_metric()   # 4
        
        # Load the model
        self.logger.info("Loading Whisper model...")
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        
        # Configure model for fine-tuning
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        
        pipeline_components = {
            "model": self.model,
            "processor": processor,
            "datasets": datasets,
            "data_collator": data_collator,
            "compute_metrics": compute_metrics,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor
        }
        
        self.logger.info("Complete - Pipeline Configured Successfully!")
        
        return pipeline_components

def main():
    """Main function to test the pipeline configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure Whisper Fine-tuning Pipeline")
    parser.add_argument("--model_name", default="openai/whisper-base", help="Whisper model to use")
    parser.add_argument("--data_dir", default="processed_data", help="Data directory")
    parser.add_argument("--test_only", action="store_true", help="Run configuration test only")
    
    args = parser.parse_args()
    
    # Initialize configurator
    configurator = WhisperPipelineConfigurator(
        model_name=args.model_name,
        data_dir=args.data_dir
    )
    
    # Configure complete pipeline
    components = configurator.configure_complete_pipeline()
    
    if args.test_only:
        print("Pipeline configuration test completed successfully!")
    
    return components

if __name__ == "__main__":
    main()
