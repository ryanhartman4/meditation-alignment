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
        print(f"\n✓ {stage_name} completed in {elapsed:.1f} seconds")
        return result
    except Exception as e:
        print(f"\nL Error in {stage_name}: {e}")
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
    
    print(" All prerequisites met")
    return True

def run_alignment_sprint(skip_preference_generation=False, skip_rft=True):
    """Run the complete alignment sprint."""
    
    print_header("ONE-NIGHT MEDITATION AI ALIGNMENT SPRINT")
    print("Starting at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Track overall results
    sprint_results = {
        "start_time": datetime.now().isoformat(),
        "stages": {}
    }
    
    # Stage 1: Generate Preferences (if needed)
    if not skip_preference_generation:
        def stage1():
            preferences = generate_all_preferences()
            sprint_results["stages"]["preference_generation"] = {
                "total_pairs": len(preferences),
                "topics": len(set(p["topic"] for p in preferences))
            }
            return preferences
        
        run_stage("Generate Synthetic Preferences", stage1)
    else:
        print("\nSkipping preference generation (using existing data)")
    
    # Stage 2: Run Alignment Pipeline
    def stage2():
        pipeline = AlignmentPipeline()
        results = pipeline.run_full_evaluation()
        sprint_results["stages"]["alignment"] = results["overall_summary"]
        return results
    
    alignment_results = run_stage("Core Alignment Evaluation", stage2)
    
    # Stage 3: Create Dashboard
    def stage3():
        dashboard_path, report_path = create_alignment_dashboard()
        sprint_results["stages"]["dashboard"] = {
            "dashboard_path": dashboard_path,
            "report_path": report_path
        }
        return dashboard_path, report_path
    
    dashboard_path, report_path = run_stage("Create Dashboard", stage3)
    
    # Stage 4: Advanced Red-Teaming (Optional)
    try:
        print("\n" + "-"*60)
        response = input("Run advanced red-teaming with Promptfoo? (y/n): ")
        if response.lower() == 'y':
            def stage4():
                from run_promptfoo_eval import run_promptfoo_evaluation
                results = run_promptfoo_evaluation()
                sprint_results["stages"]["promptfoo"] = results
                return results
            
            run_stage("Promptfoo Red-Teaming", stage4, required=False)
    except:
        print("Promptfoo not available, skipping...")
    
    # Stage 5: Multi-turn Testing (Optional)
    try:
        print("\n" + "-"*60)
        response = input("Run multi-turn consistency tests with Inspect AI? (y/n): ")
        if response.lower() == 'y':
            def stage5():
                from inspect_eval import run_inspect_evaluation
                results = run_inspect_evaluation()
                sprint_results["stages"]["inspect"] = results
                return results
            
            run_stage("Inspect AI Multi-Turn Testing", stage5, required=False)
    except:
        print("Inspect AI not available, skipping...")
    
    # Stage 6: O4-Mini RFT (Optional, expensive)
    if not skip_rft:
        print("\n" + "-"*60)
