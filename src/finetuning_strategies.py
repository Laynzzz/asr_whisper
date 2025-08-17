#!/usr/bin/env python3
"""
Fine-Tuning Strategies: LoRA Implementation
This script implements Parameter-Efficient Fine-Tuning with LoRA for Whisper models.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
import torch
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('finetuning_strategies.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class WhisperFineTuningStrategies:
    """
    LoRA Fine-Tuning Strategy for Whisper Models
    
    This class implements Parameter-Efficient Fine-Tuning with LoRA.
    """
    
    def __init__(self, model: WhisperForConditionalGeneration):
        """
        Initialize the LoRA fine-tuning strategy.
        
        Args:
            model: Pre-trained Whisper model
        """
        self.logger = setup_logging()
        self.model = model
        
        self.logger.info("🚀 LoRA Fine-Tuning Strategy for Whisper Models")
        self.logger.info("Parameter-Efficient Fine-Tuning with LoRA")
    
    def apply_lora_strategy(self, 
                           r: int = 16,
                           lora_alpha: int = 32,
                           lora_dropout: float = 0.05,
                           target_modules: list = None) -> WhisperForConditionalGeneration:
        """
        Apply LoRA (Low-Rank Adaptation) to the Whisper model.
        
        LoRA freezes the pre-trained model and injects small, trainable low-rank 
        matrices, drastically reducing the number of trainable parameters.
        
        Args:
            r: Rank of the low-rank matrices (default: 16 for quality optimization)
            lora_alpha: LoRA scaling parameter (default: 32)
            lora_dropout: Dropout probability for LoRA layers (default: 0.05)
            target_modules: List of modules to target (default: quality-optimized modules)
            
        Returns:
            Model with LoRA adapters applied
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]  # Quality-optimized targets
            
        self.logger.info("🔧 Applying LoRA Fine-Tuning Strategy")
        self.logger.info("LoRA freezes pre-trained model and injects trainable low-rank matrices")
        
        try:
            # Create LoraConfig with quality-optimized parameters
            self.logger.info(f"Creating LoraConfig targeting modules: {target_modules}")
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,  # For Whisper sequence-to-sequence models
                inference_mode=False,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            self.logger.info(f"✅ LoraConfig created with quality-optimized specifications:")
            self.logger.info(f"  • Target modules: {target_modules}")
            self.logger.info(f"  • Rank (r): {r}")
            self.logger.info(f"  • LoRA alpha: {lora_alpha}")
            self.logger.info(f"  • LoRA dropout: {lora_dropout}")
            
            # Apply LoRA to the model
            self.logger.info("Applying LoRA adapters to the model...")
            lora_model = get_peft_model(self.model, lora_config)
            
            # Log trainable parameters
            trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in lora_model.parameters())
            
            self.logger.info(f"✅ LoRA Strategy Applied Successfully:")
            self.logger.info(f"  • Total parameters: {total_params:,}")
            self.logger.info(f"  • Trainable parameters: {trainable_params:,}")
            self.logger.info(f"  • Trainable percentage: {100 * trainable_params / total_params:.2f}%")
            self.logger.info(f"  • Parameter reduction: {100 * (1 - trainable_params / total_params):.2f}%")
            
            # Print trainable modules for verification
            self.logger.info("Trainable modules:")
            for name, param in lora_model.named_parameters():
                if param.requires_grad:
                    self.logger.info(f"  • {name}: {param.numel():,} parameters")
            
            return lora_model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply LoRA strategy: {e}")
            raise
    

    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the LoRA model.
        
        Returns:
            Dictionary with LoRA model statistics
        """
        lora_model = self.apply_lora_strategy()
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        
        stats = {
            "model": lora_model,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }
        
        self.logger.info(f"📊 LoRA Model Statistics:")
        self.logger.info(f"  • Total parameters: {total_params:,}")
        self.logger.info(f"  • Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  • Trainable percentage: {stats['trainable_percentage']:.2f}%")
        
        return stats

def main():
    """Main function to test the LoRA fine-tuning strategy."""
    import argparse
    from transformers import WhisperForConditionalGeneration
    
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Strategy")
    parser.add_argument("--model_name", default="openai/whisper-base", help="Whisper model to use")
    parser.add_argument("--test_only", action="store_true", help="Run strategy test only")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.model_name}...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Initialize LoRA strategy
    strategies = WhisperFineTuningStrategies(model)
    
    # Apply LoRA and get statistics
    stats = strategies.get_model_stats()
    
    if args.test_only:
        print("✅ LoRA fine-tuning strategy test completed successfully!")
        print(f"✅ Model ready with {stats['trainable_params']:,} trainable parameters ({stats['trainable_percentage']:.2f}%)")
        return stats
    
    return strategies

if __name__ == "__main__":
    main()
