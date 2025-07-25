#!/usr/bin/env python3
"""
Main runner for the One-Night Meditation AI Alignment Sprint.
Orchestrates all stages of the alignment pipeline.
"""

import os
import sys
import time
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import DATA_DIR, RESULTS_DIR, OPENAI_API_KEY
from generate_preferences import generate_all_preferences
from alignment_loop import AlignmentPipeline
from create_dashboard import create_alignment_dashboard

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(title.center(60))
    print("="*60 + "\n")

def run_stage(stage_name, stage_func, required=True):
    """Run a pipeline stage with error handling."""
    print_header(f"STAGE: {stage_name}")
    start_time = time.time()
    
    try:
        result = stage_func()
        elapsed = time.time() - start_time
        print(f"\n {stage_name} completed in {elapsed:.1f} seconds")
        return result
    except Exception as e:
        print(f"\n Error in {stage_name}: {e}")
        if required:
            print("This is a required stage. Exiting...")
            sys.exit(1)
        else:
            print("This is an optional stage. Continuing...")
            return None

def check_prerequisites():
    """Check that all prerequisites are met."""
    print("Checking prerequisites...")
    
    # Check API key
    if not OPENAI_API_KEY:
        print("L Error: OPENAI_API_KEY not set")
        print("Please run: python src/setup_config.py")
        return False
    
    # Check directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check required files
    required_files = [
        "data/meditation_constitution.json",
        "data/meditation_test_cases.json",
        "prompts/meditation_prompt.txt"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"L Error: Required file missing: {file}")
            return False
    
    print("All prerequisites met")
    return True

def run_alignment_sprint(skip_rft=True, quick_mode=False):
    """Run the complete alignment sprint."""
    
    header_title = "ONE-NIGHT MEDITATION AI ALIGNMENT SPRINT"
    if quick_mode:
        header_title += " (QUICK MODE)"
    print_header(header_title)
    print("Starting at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if quick_mode:
        print("Quick mode enabled - generating only 1 preference pair per topic")
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Track overall results
    sprint_results = {
        "start_time": datetime.now().isoformat(),
        "stages": {}
    }
    
    
    # Stage 1: Run Alignment Pipeline
    def stage1():
        pipeline = AlignmentPipeline(quick_mode=quick_mode)
        results = pipeline.run_full_evaluation()
        sprint_results["stages"]["alignment"] = results["overall_summary"]
        return results
    
    run_stage("Core Alignment Evaluation", stage1)
    
    # Stage 2: Create Dashboard
    def stage2():
        dashboard_path = create_alignment_dashboard()
        sprint_results["stages"]["dashboard"] = {
            "dashboard_path": str(dashboard_path)
        }
        return dashboard_path
    
    dashboard_path = run_stage("Create Dashboard", stage2)
    
    # Stage 3: Advanced Red-Teaming (Optional)
    try:
        print("\n" + "-"*60)
        response = input("Run advanced red-teaming with Promptfoo? (y/n): ")
        if response.lower() == 'y':
            def stage3():
                from run_promptfoo_eval import run_promptfoo_evaluation
                results = run_promptfoo_evaluation()
                sprint_results["stages"]["promptfoo"] = results
                return results
            
            run_stage("Promptfoo Red-Teaming", stage3, required=False)
    except:
        print("Promptfoo not available, skipping...")
    
    # Stage 4: GPT-4o-Mini RFT (Optional, expensive)
    if not skip_rft:
        print("\n" + "-"*60)
        print("o4-mini Reinforcement Fine-Tuning")
        
        # Check if preferences exist
        preference_path = os.path.join(DATA_DIR, "preferences_synthetic.jsonl")
        if not os.path.exists(preference_path):
            print("\n WARNING: Preference Generation Required")
            print("This stage requires synthetic preference data for training.")
            print("\nPreference generation will create:")
            print("  - Synthetic unsafe meditation content")
            print("  - Examples with medical misinformation")
            print("  - Crisis handling failures")
            print("  - Potentially harmful advice")
            print("\nThis is necessary to train the model to avoid these patterns.")
            
            response = input("\nGenerate synthetic preferences? (y/n): ")
            if response.lower() == 'y':
                print("\nGenerating preferences...")
                preferences = generate_all_preferences(test_mode=quick_mode)
                sprint_results["stages"]["preference_generation"] = {
                    "total_pairs": len(preferences),
                    "topics": len(set(p["topic"] for p in preferences)),
                    "test_mode": quick_mode
                }
            else:
                print("Cannot proceed with RFT without preference data. Skipping...")
        
        # Only proceed if preferences exist or were just generated
        if os.path.exists(preference_path):
            print("\nRFT will:")
            print("  - Grade preference pairs using GPT-4o")
            print("  - Prepare data for fine-tuning")
            print("  - Start a fine-tuning job (takes 1-2 hours)")
            print("  - Cost approximately $25-50")
            
            response = input("\nProceed with RFT? (y/n): ")
            if response.lower() == 'y':
                def stage4():
                    from rft_training import RFTTrainingPipeline
                    pipeline = RFTTrainingPipeline()
                    results = pipeline.run_full_pipeline()
                    sprint_results["stages"]["rft"] = results
                    return results
                
                run_stage("GPT-4o-Mini Reinforcement Fine-Tuning", stage4, required=False)
    
    # Final Summary
    sprint_results["end_time"] = datetime.now().isoformat()
    sprint_results["total_duration_minutes"] = (
        datetime.fromisoformat(sprint_results["end_time"]) - 
        datetime.fromisoformat(sprint_results["start_time"])
    ).total_seconds() / 60
    
    # Save sprint results
    sprint_results_path = os.path.join(RESULTS_DIR, "sprint_results.json")
    with open(sprint_results_path, 'w') as f:
        json.dump(sprint_results, f, indent=2)
    
    # Print final summary
    print_header("ALIGNMENT SPRINT COMPLETE!")
    
    if "alignment" in sprint_results["stages"]:
        summary = sprint_results["stages"]["alignment"]
        print(f"Base Model Safety: {summary['base_safety']:.3f}")
        print(f"Aligned Model Safety: {summary['aligned_safety']:.3f}")
        print(f"Safety Improvement: {summary['safety_improvement']:.1f}%")
        print(f"Red Team Pass Rate: {summary['red_team_pass_rate']:.1%}")
        print(f"Production Ready: {'✅ YES' if summary['ready_for_production'] else '❌ NO'}")
    
    print(f"\nTotal Duration: {sprint_results['total_duration_minutes']:.1f} minutes")
    
    print("\n View your results:")
    print(f"  - Dashboard: {dashboard_path}")
    print(f"  - Sprint Summary: {sprint_results_path}")
    
    print("\n< Congratulations on completing the alignment sprint!")
    
    return sprint_results

def main():
    """Main entry point with CLI options."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the One-Night Meditation AI Alignment Sprint"
    )
    parser.add_argument(
        "--include-rft",
        action="store_true",
        help="Include o4-Mini RFT (expensive and time-consuming)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip optional stages"
    )
    
    args = parser.parse_args()
    
    # Welcome message
    print_header("WELCOME TO THE MEDITATION AI ALIGNMENT SPRINT")
    print("This sprint will demonstrate:")
    print("Constitutional AI implementation")
    print("Comprehensive safety evaluation")
    print("Red-team testing")
    print("Interactive results dashboard")
    
    if args.include_rft:
        print("o4-Mini Reinforcement Fine-Tuning")
    
    print("\nEstimated time: 15-30 minutes (without RFT)")
    
    if not args.quick:
        response = input("\nReady to begin? (y/n): ")
        if response.lower() != 'y':
            print("Sprint cancelled.")
            return
    
    # Run the sprint
    results = run_alignment_sprint(
        skip_rft=not args.include_rft,
        quick_mode=args.quick
    )
    
    # Offer to open dashboard
    if not args.quick and "dashboard" in results["stages"]:
        response = input("\nOpen dashboard in browser? (y/n): ")
        if response.lower() == 'y':
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(results['stages']['dashboard']['dashboard_path'])}")

if __name__ == "__main__":
    main()