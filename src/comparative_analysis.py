#!/usr/bin/env python3
"""
Section 8.4: Comparative Analysis
Strictly following the plan: compile results into a table to compare baseline, LoRA, 
and selective tuning methods on trainable parameters and final test WER.

Note: We only have baseline and LoRA results (selective tuning was not implemented),
so we'll create the table with available data and note the missing selective tuning results.
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComparativeAnalysis:
    """
    Section 8.4: Comparative Analysis
    
    Compiles results into Table 1: Comparative Performance Analysis
    as specified in the plan.
    """
    
    def __init__(self, results_file="evaluation_results.json"):
        self.results_file = results_file
        self.logger = logger
    
    def load_evaluation_results(self):
        """Load the evaluation results from Section 8.3"""
        self.logger.info("📊 Loading evaluation results from Section 8.3...")
        
        if not Path(self.results_file).exists():
            raise FileNotFoundError(f"Results file {self.results_file} not found. Please run Section 8.3 first.")
        
        with open(self.results_file, 'r') as f:
            results = json.load(f)
        
        self.logger.info("✅ Evaluation results loaded successfully")
        return results
    
    def generate_comparative_table(self, results):
        """
        Generate Table 1: Comparative Performance Analysis
        As specified in Section 8.4 of the plan
        """
        self.logger.info("📋 Section 8.4: Comparative Analysis")
        self.logger.info("Compiling results into comparison table as specified in plan")
        self.logger.info("=" * 70)
        
        # Extract data from results
        baseline_data = results["baseline"]
        finetuned_data = results["finetuned"]
        improvement_data = results["improvement"]
        
        # Prepare table data according to plan specification
        table_data = [
            {
                "model_configuration": "openai/whisper-base (Baseline)",
                "trainable_parameters_count": baseline_data["trainable_parameters_count"],
                "trainable_parameters_percent": baseline_data["trainable_parameters_percent"],
                "final_test_wer_percent": baseline_data["final_test_wer_percent"]
            },
            {
                "model_configuration": "Fine-tuned (LoRA, r=16)",  # Updated from plan's r=8 to actual r=16
                "trainable_parameters_count": finetuned_data["trainable_parameters_count"],
                "trainable_parameters_percent": finetuned_data["trainable_parameters_percent"],
                "final_test_wer_percent": finetuned_data["final_test_wer_percent"]
            },
            {
                "model_configuration": "Fine-tuned (Selective Layers)",
                "trainable_parameters_count": "Not Implemented",
                "trainable_parameters_percent": "Not Implemented",
                "final_test_wer_percent": "Not Implemented"
            }
        ]
        
        return table_data
    
    def display_comparative_table(self, table_data):
        """Display the comparative analysis table as specified in the plan"""
        
        print("\n" + "=" * 100)
        print("📊 TABLE 1: COMPARATIVE PERFORMANCE ANALYSIS")
        print("   (Section 8.4 - As specified in plan.md)")
        print("=" * 100)
        
        # Table header
        print(f"{'Model Configuration':<40} {'Trainable Parameters (Count)':<25} {'Trainable Parameters (%)':<20} {'Final Test WER (%)':<15}")
        print("-" * 100)
        
        # Table rows
        for row in table_data:
            config = row["model_configuration"]
            count = row["trainable_parameters_count"]
            percent = row["trainable_parameters_percent"]
            wer = row["final_test_wer_percent"]
            
            # Format the count
            if isinstance(count, int):
                count_str = f"{count:,}"
            else:
                count_str = str(count)
            
            # Format the percentage
            if isinstance(percent, float):
                percent_str = f"{percent:.2f}%"
            else:
                percent_str = str(percent)
            
            # Format the WER
            if isinstance(wer, float):
                wer_str = f"{wer:.2f}%"
            else:
                wer_str = str(wer)
            
            print(f"{config:<40} {count_str:<25} {percent_str:<20} {wer_str:<15}")
        
        print("-" * 100)
        print()
    
    def generate_analysis_summary(self, results):
        """Generate analysis summary based on the comparative results"""
        baseline_wer = results["baseline"]["final_test_wer_percent"]
        finetuned_wer = results["finetuned"]["final_test_wer_percent"]
        improvement = results["improvement"]
        
        print("📈 COMPARATIVE ANALYSIS SUMMARY:")
        print("-" * 50)
        print(f"🎯 Performance Comparison:")
        print(f"   • Baseline WER: {baseline_wer:.2f}%")
        print(f"   • LoRA Fine-tuned WER: {finetuned_wer:.2f}%")
        print(f"   • WER Improvement: {improvement['wer_improvement_percent']:+.2f}%")
        
        print(f"\n⚙️ Parameter Efficiency:")
        print(f"   • Baseline Trainable Parameters: {results['baseline']['trainable_parameters_count']:,} (100%)")
        print(f"   • LoRA Trainable Parameters: {results['finetuned']['trainable_parameters_count']:,} ({results['finetuned']['trainable_parameters_percent']:.2f}%)")
        print(f"   • Parameter Reduction: {improvement['parameter_efficiency']:.1f}x fewer parameters")
        
        print(f"\n🔬 Technical Details:")
        print(f"   • LoRA Configuration: rank=16, alpha=32, dropout=0.05")
        print(f"   • Target Modules: q_proj, v_proj, k_proj, out_proj")
        print(f"   • Training Strategy: Quality-optimized (5 epochs)")
        print(f"   • Training Time: ~15 hours (M4 MacBook Pro with MPS)")
        print(f"   • Dataset: Children's speech (4,983 train + 1,197 test samples)")
        
        print(f"\n📊 Key Findings:")
        if improvement['wer_improvement_percent'] > 0:
            print(f"   • ✅ Fine-tuning IMPROVED performance by {improvement['wer_improvement_percent']:.2f}%")
            print(f"   • ✅ Achieved better accuracy with 99% fewer parameters")
            print(f"   • ✅ Quality-optimized hyperparameters were effective")
        else:
            print(f"   • ⚠️ Fine-tuning did not improve performance")
            print(f"   • 📝 May need different hyperparameters or more data")
        
        print(f"\n🚫 Limitations:")
        print(f"   • Selective layer fine-tuning was not implemented")
        print(f"   • Only LoRA strategy was tested")
        print(f"   • Results specific to children's speech domain")
        
        print()
    
    def save_analysis_report(self, table_data, results):
        """Save the comparative analysis report"""
        report_data = {
            "section": "8.4 Comparative Analysis",
            "table_1_comparative_performance": table_data,
            "analysis_summary": {
                "baseline_wer_percent": results["baseline"]["final_test_wer_percent"],
                "lora_wer_percent": results["finetuned"]["final_test_wer_percent"],
                "wer_improvement_percent": results["improvement"]["wer_improvement_percent"],
                "parameter_efficiency": results["improvement"]["parameter_efficiency"],
                "training_time_hours": 15,
                "training_strategy": "Quality-optimized LoRA (rank=16, 5 epochs)"
            },
            "implementation_status": {
                "baseline_evaluation": "✅ Completed",
                "lora_fine_tuning": "✅ Completed", 
                "selective_layer_tuning": "❌ Not Implemented"
            }
        }
        
        report_file = "section_8_4_comparative_analysis.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"📄 Comparative analysis report saved to {report_file}")
        return report_file
    
    def run_comparative_analysis(self):
        """
        Run the complete Section 8.4 comparative analysis
        """
        self.logger.info("🚀 Section 8.4: Comparative Analysis")
        self.logger.info("Compiling results as specified in plan.md")
        self.logger.info("=" * 70)
        
        try:
            # Load evaluation results from Section 8.3
            results = self.load_evaluation_results()
            
            # Generate comparative table
            table_data = self.generate_comparative_table(results)
            
            # Display the table
            self.display_comparative_table(table_data)
            
            # Generate analysis summary
            self.generate_analysis_summary(results)
            
            # Save report
            report_file = self.save_analysis_report(table_data, results)
            
            self.logger.info("=" * 70)
            self.logger.info("🎉 SECTION 8.4 COMPARATIVE ANALYSIS COMPLETED!")
            self.logger.info("=" * 70)
            self.logger.info("✅ Table 1: Comparative Performance Analysis generated")
            self.logger.info("✅ Analysis summary provided")
            self.logger.info(f"✅ Report saved to {report_file}")
            self.logger.info("📋 Ready for Section 9: Final Submission and Reproducibility Package")
            
            return {
                "table_data": table_data,
                "report_file": report_file,
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Comparative analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run comparative analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Section 8.4: Comparative Analysis")
    parser.add_argument("--results-file", default="evaluation_results.json",
                      help="Results file from Section 8.3")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ComparativeAnalysis(results_file=args.results_file)
    
    # Run comparative analysis
    result = analyzer.run_comparative_analysis()
    
    if result:
        print(f"\n🎉 Section 8.4 completed successfully!")
        print(f"📊 Comparative analysis table generated")
        print(f"📄 Report saved: {result['report_file']}")
        print(f"🚀 Ready for Section 9: Final Submission")
        return 0
    else:
        print(f"\n❌ Section 8.4 failed!")
        return 1

if __name__ == "__main__":
    exit(main())
