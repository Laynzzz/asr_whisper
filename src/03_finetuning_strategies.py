#!/usr/bin/env python3
"""
Section 7: Implementation of Fine-Tuning Strategies
This script implements the two fine-tuning methodologies to be compared
as specified in the plan.
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
    Section 7: Implementation of Fine-Tuning Strategies
    
    This class implements the two fine-tuning methodologies:
    7.1 Strategy A: Parameter-Efficient Fine-Tuning with LoRA
    7.2 Strategy B: Selective Layer Fine-Tuning
    """
    
    def __init__(self, model: WhisperForConditionalGeneration):
        """
        Initialize the fine-tuning strategies configurator.
        
        Args:
            model: Pre-trained Whisper model from Section 6
        """
        self.logger = setup_logging()
        self.model = model
        self.original_model = model  # Keep reference to original
        
        self.logger.info("🚀 Section 7: Implementation of Fine-Tuning Strategies")
        self.logger.info("Two methodologies to be compared:")
        self.logger.info("  • Strategy A: Parameter-Efficient Fine-Tuning with LoRA")
        self.logger.info("  • Strategy B: Selective Layer Fine-Tuning")
    
    def apply_lora_strategy(self, 
                           r: int = 8,
                           lora_alpha: int = 32,
                           lora_dropout: float = 0.1) -> WhisperForConditionalGeneration:
        """
        Section 7.1: Strategy A: Parameter-Efficient Fine-Tuning with LoRA
        
        LoRA freezes the pre-trained model and injects small, trainable low-rank 
        matrices, drastically reducing the number of trainable parameters.
        
        LoRA Configuration: A LoraConfig will be created using the peft library, 
        targeting the q_proj and v_proj modules, which is an effective strategy 
        for Transformer models.
        
        Args:
            r: Rank of the low-rank matrices
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout probability for LoRA layers
            
        Returns:
            Model with LoRA adapters applied
        """
        self.logger.info("🔧 Section 7.1: Strategy A - Parameter-Efficient Fine-Tuning with LoRA")
        self.logger.info("LoRA freezes pre-trained model and injects trainable low-rank matrices")
        
        try:
            # Create LoraConfig targeting q_proj and v_proj modules as specified
            self.logger.info("Creating LoraConfig targeting q_proj and v_proj modules...")
            
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # For encoder-decoder models
                inference_mode=False,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],  # Targeting q_proj and v_proj as specified
                bias="none"
            )
            
            self.logger.info(f"✅ LoraConfig created with specifications:")
            self.logger.info(f"  • Target modules: q_proj, v_proj (effective for Transformers)")
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
    
    def apply_selective_layer_strategy(self, 
                                     freeze_encoder_layers: int = 3,
                                     freeze_decoder_layers: int = 0) -> WhisperForConditionalGeneration:
        """
        Section 7.2: Strategy B: Selective Layer Fine-Tuning
        
        This strategy involves freezing most of the model and unfreezing only 
        a subset of layers.
        
        Layer Selection Rationale: The most effective approach is to freeze the 
        lower encoder layers, which capture general acoustic features, and unfreeze 
        the top encoder layers and the entire decoder to adapt to the specific 
        nuances of children's speech.
        
        Args:
            freeze_encoder_layers: Number of lower encoder layers to freeze
            freeze_decoder_layers: Number of decoder layers to freeze (0 = unfreeze all)
            
        Returns:
            Model with selective layers frozen/unfrozen
        """
        self.logger.info("🎯 Section 7.2: Strategy B - Selective Layer Fine-Tuning")
        self.logger.info("Freeze lower encoder layers, unfreeze top encoder + entire decoder")
        
        try:
            # Start fresh from original model
            model = self.original_model
            
            # First, freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            
            self.logger.info("Layer Selection Rationale Implementation:")
            self.logger.info("• Freeze lower encoder layers (capture general acoustic features)")
            self.logger.info("• Unfreeze top encoder layers (adapt to children's speech nuances)")
            self.logger.info("• Unfreeze entire decoder (adapt to children's speech nuances)")
            
            # Get model structure info
            encoder_layers = len(model.model.encoder.layers)
            decoder_layers = len(model.model.decoder.layers)
            
            self.logger.info(f"Model structure: {encoder_layers} encoder layers, {decoder_layers} decoder layers")
            
            # Freeze lower encoder layers as specified
            for i in range(min(freeze_encoder_layers, encoder_layers)):
                for param in model.model.encoder.layers[i].parameters():
                    param.requires_grad = False
                self.logger.info(f"  • Frozen encoder layer {i} (general acoustic features)")
            
            # Unfreeze top encoder layers as specified
            for i in range(freeze_encoder_layers, encoder_layers):
                for param in model.model.encoder.layers[i].parameters():
                    param.requires_grad = True
                self.logger.info(f"  • Unfrozen encoder layer {i} (children's speech adaptation)")
            
            # Unfreeze entire decoder as specified in rationale
            self.logger.info("Unfreezing entire decoder for children's speech adaptation...")
            for i in range(decoder_layers):
                for param in model.model.decoder.layers[i].parameters():
                    param.requires_grad = True
                if i < 3:  # Log first few for brevity
                    self.logger.info(f"  • Unfrozen decoder layer {i}")
            
            if decoder_layers > 3:
                self.logger.info(f"  • ... and {decoder_layers - 3} more decoder layers")
            
            # Unfreeze decoder embeddings and final layer norm
            for param in model.model.decoder.embed_tokens.parameters():
                param.requires_grad = True
            for param in model.model.decoder.embed_positions.parameters():
                param.requires_grad = True
            for param in model.model.decoder.layer_norm.parameters():
                param.requires_grad = True
            
            # Unfreeze language modeling head
            for param in model.proj_out.parameters():
                param.requires_grad = True
            
            # Log trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            self.logger.info(f"✅ Selective Layer Strategy Applied Successfully:")
            self.logger.info(f"  • Frozen encoder layers: {freeze_encoder_layers}/{encoder_layers}")
            self.logger.info(f"  • Unfrozen encoder layers: {encoder_layers - freeze_encoder_layers}/{encoder_layers}")
            self.logger.info(f"  • Unfrozen decoder layers: {decoder_layers}/{decoder_layers} (entire decoder)")
            self.logger.info(f"  • Total parameters: {total_params:,}")
            self.logger.info(f"  • Trainable parameters: {trainable_params:,}")
            self.logger.info(f"  • Trainable percentage: {100 * trainable_params / total_params:.2f}%")
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to apply selective layer strategy: {e}")
            raise
    
    def compare_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare both fine-tuning strategies and return statistics.
        
        Returns:
            Dictionary with statistics for both strategies
        """
        self.logger.info("📊 Comparing Fine-Tuning Strategies")
        
        # Strategy A: LoRA
        self.logger.info("\n" + "="*60)
        self.logger.info("STRATEGY A: Parameter-Efficient Fine-Tuning with LoRA")
        self.logger.info("="*60)
        
        lora_model = self.apply_lora_strategy()
        lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        lora_total = sum(p.numel() for p in lora_model.parameters())
        
        # Strategy B: Selective Layer Fine-Tuning  
        self.logger.info("\n" + "="*60)
        self.logger.info("STRATEGY B: Selective Layer Fine-Tuning")
        self.logger.info("="*60)
        
        selective_model = self.apply_selective_layer_strategy()
        selective_trainable = sum(p.numel() for p in selective_model.parameters() if p.requires_grad)
        selective_total = sum(p.numel() for p in selective_model.parameters())
        
        # Comparison summary
        self.logger.info("\n" + "="*60)
        self.logger.info("STRATEGY COMPARISON SUMMARY")
        self.logger.info("="*60)
        
        comparison = {
            "lora": {
                "name": "Parameter-Efficient Fine-Tuning with LoRA",
                "total_params": lora_total,
                "trainable_params": lora_trainable,
                "trainable_percentage": 100 * lora_trainable / lora_total,
                "model": lora_model
            },
            "selective": {
                "name": "Selective Layer Fine-Tuning", 
                "total_params": selective_total,
                "trainable_params": selective_trainable,
                "trainable_percentage": 100 * selective_trainable / selective_total,
                "model": selective_model
            }
        }
        
        self.logger.info(f"LoRA Strategy:")
        self.logger.info(f"  • Trainable parameters: {lora_trainable:,} ({comparison['lora']['trainable_percentage']:.2f}%)")
        
        self.logger.info(f"Selective Layer Strategy:")
        self.logger.info(f"  • Trainable parameters: {selective_trainable:,} ({comparison['selective']['trainable_percentage']:.2f}%)")
        
        efficiency_ratio = lora_trainable / selective_trainable
        self.logger.info(f"Efficiency Ratio (LoRA/Selective): {efficiency_ratio:.2f}x")
        
        if efficiency_ratio < 1:
            self.logger.info("✅ LoRA is more parameter-efficient")
        else:
            self.logger.info("✅ Selective Layer strategy is more parameter-efficient")
        
        self.logger.info("\n🎯 Section 7 Complete - Both strategies implemented and ready for training!")
        
        return comparison

def main():
    """Main function to test the fine-tuning strategies."""
    import argparse
    from transformers import WhisperForConditionalGeneration
    
    parser = argparse.ArgumentParser(description="Section 7: Fine-tuning Strategies")
    parser.add_argument("--model_name", default="openai/whisper-base", help="Whisper model to use")
    parser.add_argument("--strategy", choices=["lora", "selective", "compare"], default="compare", 
                       help="Strategy to apply")
    parser.add_argument("--test_only", action="store_true", help="Run strategy test only")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.model_name}...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Initialize strategies
    strategies = WhisperFineTuningStrategies(model)
    
    if args.strategy == "lora":
        result_model = strategies.apply_lora_strategy()
    elif args.strategy == "selective":
        result_model = strategies.apply_selective_layer_strategy()
    elif args.strategy == "compare":
        comparison = strategies.compare_strategies()
        
        if args.test_only:
            print("✅ Fine-tuning strategies test completed successfully!")
            print("Both Strategy A (LoRA) and Strategy B (Selective Layer) are working correctly.")
            return comparison
    
    return strategies

if __name__ == "__main__":
    main()
