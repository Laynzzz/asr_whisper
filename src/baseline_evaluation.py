#!/usr/bin/env python3
"""
Section 8.3: Baseline and Final Evaluation
Implements the three-way comparison as specified in the plan:
1. Baseline (Zero-Shot) Evaluation of original Whisper model
2. Fine-Tuned Model Evaluation comparison
3. Comparative Analysis table generation
"""

import torch
import sys
import os
import logging
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import jiwer
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineEvaluation:
    """
    Section 8.3: Baseline and Final Evaluation
    
    Implements the three-way comparison as specified in the plan:
    - Baseline (Zero-Shot) Evaluation
    - Fine-Tuned Model Evaluation
    - Comparative Analysis
    """
    
    def __init__(self, model_name="openai/whisper-base", data_dir="processed_data", 
                 finetuned_model_dir="whisper_quality_training"):
        self.model_name = model_name
        self.data_dir = data_dir
        self.finetuned_model_dir = finetuned_model_dir
        self.logger = logger
        
        # Check device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")
    
    def load_test_dataset(self):
        """Load the test dataset for evaluation"""
        self.logger.info("📊 Loading test dataset for evaluation...")
        
        test_dataset = load_dataset("audiofolder", data_dir=f"{self.data_dir}/test/", split="train")
        test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        self.logger.info(f"✅ Test dataset loaded: {len(test_dataset)} samples")
        return test_dataset
    
    def evaluate_baseline_model(self, test_dataset):
        """
        Baseline (Zero-Shot) Evaluation: 
        Evaluate original openai/whisper-base model on test set
        """
        self.logger.info("🔍 Section 8.3.1: Baseline (Zero-Shot) Evaluation")
        self.logger.info("Evaluating original openai/whisper-base model")
        self.logger.info("=" * 60)
        
        # Load original model and processor
        processor = WhisperProcessor.from_pretrained(self.model_name, language="english", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        model = model.to(self.device)
        
        # Evaluate on test set
        predictions = []
        references = []
        
        self.logger.info(f"Processing {len(test_dataset)} test samples...")
        
        for i, sample in enumerate(test_dataset):
            if i % 100 == 0:
                self.logger.info(f"Progress: {i}/{len(test_dataset)} samples")
            
            # Process audio
            input_features = processor.feature_extractor(
                sample["audio"]["array"], 
                sampling_rate=sample["audio"]["sampling_rate"],
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                # Use environment variable for MPS fallback
                predicted_ids = model.generate(input_features, max_length=225)
            
            # Decode transcription
            transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            predictions.append(transcription)
            references.append(sample["transcription"])
        
        # Calculate WER
        baseline_wer = jiwer.wer(references, predictions)
        
        self.logger.info("✅ Baseline evaluation completed")
        self.logger.info(f"📊 Baseline Results:")
        self.logger.info(f"   • Model: {self.model_name}")
        self.logger.info(f"   • Test samples: {len(test_dataset)}")
        self.logger.info(f"   • Baseline WER: {baseline_wer:.4f} ({baseline_wer*100:.2f}%)")
        
        return {
            "model_name": self.model_name,
            "wer": baseline_wer,
            "test_samples": len(test_dataset),
            "predictions": predictions,
            "references": references
        }
    
    def evaluate_finetuned_model(self, test_dataset):
        """
        Fine-Tuned Model Evaluation:
        Evaluate the best-performing checkpoint on test set
        """
        self.logger.info("🎯 Section 8.3.2: Fine-Tuned Model Evaluation")
        self.logger.info("Evaluating quality-optimized LoRA fine-tuned model")
        self.logger.info("=" * 60)
        
        # Load processor
        processor = WhisperProcessor.from_pretrained(self.model_name, language="english", task="transcribe")
        
        # Load base model
        base_model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        
        # Load LoRA adapters
        lora_adapters_path = f"{self.finetuned_model_dir}/lora_adapters"
        if os.path.exists(lora_adapters_path):
            self.logger.info(f"Loading LoRA adapters from {lora_adapters_path}")
            model = PeftModel.from_pretrained(base_model, lora_adapters_path)
            # Use the working approach from training
            working_model = model.base_model.model.to(self.device)
        else:
            # Fallback: load the saved model directly
            self.logger.info(f"Loading fine-tuned model from {self.finetuned_model_dir}")
            model = WhisperForConditionalGeneration.from_pretrained(self.finetuned_model_dir)
            working_model = model.to(self.device)
        
        # Evaluate on test set
        predictions = []
        references = []
        
        self.logger.info(f"Processing {len(test_dataset)} test samples...")
        
        for i, sample in enumerate(test_dataset):
            if i % 100 == 0:
                self.logger.info(f"Progress: {i}/{len(test_dataset)} samples")
            
            # Process audio
            input_features = processor.feature_extractor(
                sample["audio"]["array"], 
                sampling_rate=sample["audio"]["sampling_rate"],
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = working_model.generate(input_features, max_length=225)
            
            # Decode transcription
            transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            predictions.append(transcription)
            references.append(sample["transcription"])
        
        # Calculate WER
        finetuned_wer = jiwer.wer(references, predictions)
        
        self.logger.info("✅ Fine-tuned model evaluation completed")
        self.logger.info(f"📊 Fine-Tuned Results:")
        self.logger.info(f"   • Model: Quality-Optimized LoRA Fine-Tuned")
        self.logger.info(f"   • Test samples: {len(test_dataset)}")
        self.logger.info(f"   • Fine-tuned WER: {finetuned_wer:.4f} ({finetuned_wer*100:.2f}%)")
        
        return {
            "model_name": "Quality-Optimized LoRA Fine-Tuned",
            "wer": finetuned_wer,
            "test_samples": len(test_dataset),
            "predictions": predictions,
            "references": references
        }
    
    def generate_comparative_analysis(self, baseline_results, finetuned_results):
        """
        Section 8.3.3: Generate Comparative Analysis Table
        As specified in plan: Table 1: Comparative Performance Analysis
        """
        self.logger.info("📋 Section 8.3.3: Comparative Analysis")
        self.logger.info("Generating comparison table as specified in plan")
        self.logger.info("=" * 60)
        
        # Calculate improvements
        wer_improvement = baseline_results["wer"] - finetuned_results["wer"]
        wer_improvement_percent = (wer_improvement / baseline_results["wer"]) * 100
        
        # Model configuration details
        baseline_trainable_params = 72888832  # All parameters of whisper-base
        finetuned_trainable_params = 784466  # From training logs (quality-optimized LoRA)
        
        baseline_trainable_percent = 100.0
        finetuned_trainable_percent = (finetuned_trainable_params / baseline_trainable_params) * 100
        
        # Create comparison table
        comparison_data = {
            "baseline": {
                "model_configuration": "Whisper-Base (Zero-Shot)",
                "trainable_parameters_count": baseline_trainable_params,
                "trainable_parameters_percent": baseline_trainable_percent,
                "final_test_wer_raw": baseline_results["wer"],
                "final_test_wer_percent": baseline_results["wer"] * 100
            },
            "finetuned": {
                "model_configuration": "Quality-Optimized LoRA Fine-Tuned",
                "trainable_parameters_count": finetuned_trainable_params,
                "trainable_parameters_percent": finetuned_trainable_percent,
                "final_test_wer_raw": finetuned_results["wer"],
                "final_test_wer_percent": finetuned_results["wer"] * 100
            },
            "improvement": {
                "wer_reduction": wer_improvement,
                "wer_improvement_percent": wer_improvement_percent,
                "parameter_efficiency": baseline_trainable_params / finetuned_trainable_params
            }
        }
        
        # Display comparison table
        print("\n" + "=" * 80)
        print("📊 TABLE 1: COMPARATIVE PERFORMANCE ANALYSIS")
        print("=" * 80)
        print(f"{'Model Configuration':<35} {'Trainable Params':<15} {'Trainable %':<12} {'Test WER %':<10}")
        print("-" * 80)
        print(f"{'Whisper-Base (Zero-Shot)':<35} {baseline_trainable_params:,<15} {baseline_trainable_percent:<12.2f} {baseline_results['wer']*100:<10.2f}")
        print(f"{'Quality-Optimized LoRA':<35} {finetuned_trainable_params:,<15} {finetuned_trainable_percent:<12.2f} {finetuned_results['wer']*100:<10.2f}")
        print("-" * 80)
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   • WER Improvement: {wer_improvement:.4f} ({wer_improvement_percent:+.2f}%)")
        print(f"   • Parameter Efficiency: {comparison_data['improvement']['parameter_efficiency']:.1f}x fewer parameters")
        print(f"   • Training Strategy: Quality-Optimized LoRA (rank=16, 5 epochs)")
        print(f"   • Training Time: ~15 hours (M4 MacBook Pro with MPS)")
        
        if wer_improvement > 0:
            print(f"   • ✅ Fine-tuning IMPROVED performance")
        else:
            print(f"   • ⚠️  Fine-tuning did not improve performance")
        
        # Save results
        results_file = "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        self.logger.info(f"✅ Comparative analysis completed")
        self.logger.info(f"📄 Results saved to {results_file}")
        
        return comparison_data
    
    def run_complete_evaluation(self):
        """
        Run the complete Section 8.3 evaluation process
        """
        self.logger.info("🚀 Section 8.3: Baseline and Final Evaluation")
        self.logger.info("Three-way comparison as specified in plan")
        self.logger.info("=" * 70)
        
        try:
            # Load test dataset
            test_dataset = self.load_test_dataset()
            
            # 1. Baseline (Zero-Shot) Evaluation
            baseline_results = self.evaluate_baseline_model(test_dataset)
            
            # 2. Fine-Tuned Model Evaluation
            finetuned_results = self.evaluate_finetuned_model(test_dataset)
            
            # 3. Comparative Analysis
            comparison_data = self.generate_comparative_analysis(baseline_results, finetuned_results)
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info("🎉 SECTION 8.3 EVALUATION COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 70)
            self.logger.info("✅ All evaluations completed as specified in plan")
            self.logger.info("📊 Comparative analysis table generated")
            self.logger.info("📄 Results saved for Section 8.4 analysis")
            
            return {
                "baseline": baseline_results,
                "finetuned": finetuned_results,
                "comparison": comparison_data
            }
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run baseline and final evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Section 8.3: Baseline and Final Evaluation")
    parser.add_argument("--data-dir", default="processed_data",
                      help="Directory containing processed data")
    parser.add_argument("--finetuned-model-dir", default="whisper_quality_training",
                      help="Directory containing fine-tuned model")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BaselineEvaluation(
        data_dir=args.data_dir,
        finetuned_model_dir=args.finetuned_model_dir
    )
    
    # Run complete evaluation
    results = evaluator.run_complete_evaluation()
    
    if results:
        print(f"\n🎉 Section 8.3 completed successfully!")
        print(f"📊 Ready for Section 8.4: Comparative Analysis")
        return 0
    else:
        print(f"\n❌ Section 8.3 failed!")
        return 1

if __name__ == "__main__":
    # Set MPS fallback for evaluation
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    exit(main())
