#!/usr/bin/env python3
"""
 Simplified Full Training Execution
Fixed version without recursion issues, using the successful smoke test approach.
"""

import torch
import time
import sys
import os
import logging
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType
import jiwer

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import DataCollatorSpeechSeq2SeqWithPadding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quality_training():
    """Run quality-optimized training based on successful smoke test approach"""
    
    print("Quality-Optimized Full Training")
    print("=" * 55)
    print("Using successful smoke test approach with quality hyperparameters")
    print()
    
    try:
        # Check device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"✅ Using Metal Performance Shaders: {device}")
        else:
            device = torch.device("cpu")
            print(f"⚠️  Using CPU: {device}")
        
        # Load datasets
        print("📊 Loading full datasets...")
        train_dataset = load_dataset("audiofolder", data_dir="processed_data/train/", split="train")
        eval_dataset = load_dataset("audiofolder", data_dir="processed_data/test/", split="train")
        
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
        eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        print(f"✅ Loaded {len(train_dataset)} train + {len(eval_dataset)} eval samples")
        
        # Initialize model and processor
        print("🤖 Initializing model and processor...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="english", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        
        # Apply LoRA (using successful approach from smoke test)
        print("🔧 Applying quality-optimized LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,  # Quality-optimized: increased from 8
            lora_alpha=32,
            lora_dropout=0.05,  # Quality-optimized: reduced from 0.1
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # Quality-optimized: more modules
            bias="none"
        )
        lora_model = get_peft_model(model, lora_config)
        
        # Use the working approach from smoke test
        working_model = lora_model.base_model.model
        working_model = working_model.to(device)
        
        print(f"✅ Model configured and moved to {device}")
        
        # Prepare datasets
        print("📋 Preparing datasets...")
        def prepare_dataset(batch):
            audio = batch["audio"]
            input_features = processor.feature_extractor(
                audio["array"], 
                sampling_rate=audio["sampling_rate"]
            ).input_features[0]
            
            labels = processor.tokenizer(batch["transcription"]).input_ids
            
            return {
                "input_features": input_features,
                "labels": labels
            }
        
        train_dataset = train_dataset.map(
            prepare_dataset,
            remove_columns=train_dataset.column_names,
            desc="Preparing train dataset"
        )
        
        eval_dataset = eval_dataset.map(
            prepare_dataset,
            remove_columns=eval_dataset.column_names,
            desc="Preparing eval dataset"
        )
        
        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        
        # Define compute_metrics
        def compute_metrics(eval_pred):
            pred_ids, label_ids = eval_pred
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            
            pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            
            wer = jiwer.wer(label_str, pred_str)
            return {"wer": wer}
        
        # Quality-optimized training arguments
        print("⚙️  Configuring quality-optimized training arguments...")
        training_args = Seq2SeqTrainingArguments(
            output_dir="./whisper_quality_training",
            per_device_train_batch_size=8,  # Quality-optimized
            gradient_accumulation_steps=2,  # Quality-optimized
            per_device_eval_batch_size=4,   # Quality-optimized
            learning_rate=5e-6,             # Quality-optimized: lower for stability
            num_train_epochs=5,             # Quality-optimized: more epochs
            warmup_ratio=0.15,              # Quality-optimized: longer warmup
            lr_scheduler_type="cosine",     # Quality-optimized: better scheduler
            eval_strategy="steps",
            eval_steps=155,                 # Quality-optimized: twice per epoch
            save_steps=310,                 # Quality-optimized: multiple of eval_steps (155 * 2)
            logging_steps=25,               # Quality-optimized: more frequent logging
            predict_with_generate=True,
            generation_max_length=225,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_wer",  # Quality-optimized: optimize for WER
            greater_is_better=False,
            report_to=None,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            use_mps_device=device.type == "mps"
        )
        
        print("✅ Quality-optimized training configured:")
        print(f"   • Batch size: 8 (gradient accumulation: 2)")
        print(f"   • Learning rate: 5e-6")
        print(f"   • Epochs: 5")
        print(f"   • LR scheduler: cosine")
        print(f"   • LoRA rank: 16")
        print(f"   • Target modules: q_proj, v_proj, k_proj, out_proj")
        print(f"   • Evaluation: every 155 steps")
        print(f"   • Best model metric: eval_wer")
        
        # Initialize trainer
        print("🏃 Initializing trainer...")
        trainer = Seq2SeqTrainer(
            model=working_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=processor.tokenizer,
            compute_metrics=compute_metrics
        )
        
        # Calculate expected time
        steps_per_epoch = len(train_dataset) // (8 * 2)  # batch_size * gradient_accumulation
        total_steps = steps_per_epoch * 5
        estimated_time_minutes = total_steps * 1.76 / 60  # Based on smoke test results
        
        print(f"📊 Training details:")
        print(f"   • Steps per epoch: {steps_per_epoch}")
        print(f"   • Total steps: {total_steps}")
        print()
        
        # Start training
        print("🔥 Starting quality-optimized training...")
        start_time = time.time()
        
        train_result = trainer.train()
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        # Save model
        print("💾 Saving the best model...")
        trainer.save_model()
        
        # Save LoRA adapters
        lora_output_dir = "./whisper_quality_training/lora_adapters"
        lora_model.save_pretrained(lora_output_dir)
        print(f"✅ LoRA adapters saved to {lora_output_dir}")
        
        # Final evaluation
        print("📊 Final evaluation...")
        eval_result = trainer.evaluate()
        
        # Results summary
        print()
        print("=" * 55)
        print("🎉 QUALITY TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 55)
        print(f"📊 Results:")
        print(f"   • Training time: {training_duration/60:.1f} minutes")
        print(f"   • Final train loss: {train_result.training_loss:.4f}")
        print(f"   • Final eval loss: {eval_result['eval_loss']:.4f}")
        print(f"   • Final eval WER: {eval_result['eval_wer']:.4f}")
        print(f"   • Model saved: ./whisper_quality_training")
        print()
        
        # Performance analysis
        print(f"📈 Performance Analysis:")
        print(f"   • Estimated: {estimated_time_minutes:.1f} minutes")
        print(f"   • Actual: {training_duration/60:.1f} minutes")
        print(f"   • Difference: {training_duration/60 - estimated_time_minutes:+.1f} minutes")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quality_training()
    if success:
        print("🚀 Ready for Baseline and Final Evaluation!")
    else:
        print("❌ Training failed - check logs for details")
    exit(0 if success else 1)
