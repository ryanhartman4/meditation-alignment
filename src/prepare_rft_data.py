"""
Prepare data for O4-Mini Reinforcement Fine-Tuning.
Formats graded preferences into OpenAI fine-tuning format.
"""

import json
import os
import random
from typing import List, Dict, Tuple
from datetime import datetime
from config import DATA_DIR, RESULTS_DIR

class RFTDataPreparer:
    """Prepare and format data for RFT."""
    
    def __init__(self):
        self.system_prompt = """You are a compassionate meditation guide focused on safety and helpfulness. 
Always prioritize user safety by:
- Never providing medical advice
- Redirecting crisis situations to professional help (988 lifeline)
- Maintaining appropriate boundaries
- Using inclusive, accessible language
- Acknowledging when professional support is needed"""
    
    def format_for_openai_rft(self, graded_pairs_path: str, output_path: str):
        """Format graded pairs for OpenAI's RFT training format."""
        
        print(f"Loading graded pairs from {graded_pairs_path}...")
        
        # Load valid pairs
        valid_pairs = []
        with open(graded_pairs_path, 'r') as f:
            for line in f:
                if line.strip():
                    pair = json.loads(line)
                    if pair.get("is_valid_pair", False):
                        valid_pairs.append(pair)
        
        print(f"Found {len(valid_pairs)} valid pairs for RFT")
        
        # Format for OpenAI fine-tuning
        formatted_data = []
        
        for pair in valid_pairs:
            # Create training example with chosen response
            formatted_data.append({
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["chosen"]}
                ],
                "metadata": {
                    "score": pair["chosen_grades"]["overall_score"],
                    "safety_score": pair["chosen_grades"]["safety_score"],
                    "margin": pair["overall_margin"],
                    "pair_quality": pair["pair_quality"]
                }
            })
        
        # Split into train/validation
        random.shuffle(formatted_data)
        split_idx = int(len(formatted_data) * 0.9)
        
        train_data = formatted_data[:split_idx]
        val_data = formatted_data[split_idx:]
        
        # Save training data
        train_path = output_path.replace('.jsonl', '_train.jsonl')
        with open(train_path, 'w') as f:
            for example in train_data:
                # OpenAI format doesn't include metadata in the file
                f.write(json.dumps({
                    "messages": example["messages"]
                }) + '\n')
        
        # Save validation data
        val_path = output_path.replace('.jsonl', '_val.jsonl')
        with open(val_path, 'w') as f:
            for example in val_data:
                f.write(json.dumps({
                    "messages": example["messages"]
                }) + '\n')
        
        # Save metadata separately for analysis
        meta_path = output_path.replace('.jsonl', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump({
                "dataset_stats": self._calculate_dataset_stats(formatted_data),
                "train_size": len(train_data),
                "val_size": len(val_data),
                "system_prompt": self.system_prompt,
                "creation_time": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\nData preparation complete!")
        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        print(f"Files saved:")
        print(f"  - {train_path}")
        print(f"  - {val_path}")
        print(f"  - {meta_path}")
        
        return {
            "train_path": train_path,
            "val_path": val_path,
            "train_size": len(train_data),
            "val_size": len(val_data)
        }
    
    def create_rft_contrast_sets(self, graded_pairs_path: str, output_path: str):
        """Create contrast sets for more effective RFT training."""
        
        print("Creating contrast sets for RFT...")
        
        # Load all pairs (including invalid ones for contrast)
        all_pairs = []
        with open(graded_pairs_path, 'r') as f:
            for line in f:
                if line.strip():
                    all_pairs.append(json.loads(line))
        
        # Group by prompt for contrast learning
        prompt_groups = {}
        for pair in all_pairs:
            prompt = pair["prompt"]
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append(pair)
        
        # Create contrast sets
        contrast_sets = []
        
        for prompt, pairs in prompt_groups.items():
            if len(pairs) >= 2:
                # Sort by overall score
                pairs.sort(key=lambda x: x["chosen_grades"]["overall_score"], reverse=True)
                
                # Create contrasts between different quality levels
                for i in range(len(pairs) - 1):
                    if pairs[i]["chosen_grades"]["overall_score"] - pairs[i+1]["chosen_grades"]["overall_score"] >= 2.0:
                        contrast_sets.append({
                            "prompt": prompt,
                            "better": pairs[i]["chosen"],
                            "worse": pairs[i+1]["chosen"],
                            "score_diff": pairs[i]["chosen_grades"]["overall_score"] - pairs[i+1]["chosen_grades"]["overall_score"],
                            "better_score": pairs[i]["chosen_grades"]["overall_score"],
                            "worse_score": pairs[i+1]["chosen_grades"]["overall_score"]
                        })
        
        # Save contrast sets
        with open(output_path, 'w') as f:
            for contrast in contrast_sets:
                f.write(json.dumps(contrast) + '\n')
        
        print(f"Created {len(contrast_sets)} contrast pairs")
        return contrast_sets
    
    def create_safety_critical_dataset(self, graded_pairs_path: str, test_cases_path: str):
        """Create a dataset focused on safety-critical scenarios."""
        
        print("Creating safety-critical dataset...")
        
        # Load test cases
        with open(test_cases_path, 'r') as f:
            test_data = json.load(f)
        
        critical_prompts = []
        
        # Extract critical test prompts
        for test in test_data["red_team_tests"]:
            if test["severity"] == "critical":
                critical_prompts.append({
                    "prompt": test["prompt"],
                    "category": test["category"],
                    "must_include": test["must_include"],
                    "must_not_include": test["must_not_include"]
                })
        
        # Generate ideal responses for critical scenarios
        from alignment_loop import AlignmentPipeline
        pipeline = AlignmentPipeline()
        
        critical_dataset = []
        
        for critical in critical_prompts:
            # Generate multiple attempts and pick the best
            responses = []
            for _ in range(3):
                response = pipeline.generate_aligned(critical["prompt"])
                
                # Check if response meets requirements
                includes_required = all(
                    any(term.lower() in response.lower() for term in critical["must_include"])
                )
                excludes_forbidden = all(
                    term.lower() not in response.lower() 
                    for term in critical["must_not_include"]
                )
                
                if includes_required and excludes_forbidden:
                    responses.append(response)
            
            if responses:
                # Add to dataset
                critical_dataset.append({
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": critical["prompt"]},
                        {"role": "assistant", "content": responses[0]}
                    ],
                    "metadata": {
                        "category": critical["category"],
                        "severity": "critical",
                        "safety_critical": True
                    }
                })
        
        # Save critical dataset
        critical_path = os.path.join(DATA_DIR, "rft_safety_critical.jsonl")
        with open(critical_path, 'w') as f:
            for example in critical_dataset:
                f.write(json.dumps({
                    "messages": example["messages"]
                }) + '\n')
        
        print(f"Created {len(critical_dataset)} safety-critical examples")
        return critical_dataset
    
    def _calculate_dataset_stats(self, data: List[Dict]) -> Dict:
        """Calculate statistics for the dataset."""
        
        scores = [ex["metadata"]["score"] for ex in data]
        safety_scores = [ex["metadata"]["safety_score"] for ex in data]
        margins = [ex["metadata"]["margin"] for ex in data]
        
        quality_dist = {"high": 0, "medium": 0, "low": 0}
        for ex in data:
            quality_dist[ex["metadata"]["pair_quality"]] += 1
        
        return {
            "total_examples": len(data),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "average_safety_score": sum(safety_scores) / len(safety_scores) if safety_scores else 0,
            "average_margin": sum(margins) / len(margins) if margins else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "quality_distribution": quality_dist
        }
    
    def validate_rft_data(self, train_path: str, val_path: str):
        """Validate RFT data format and quality."""
        
        print("\nValidating RFT data...")
        
        issues = []
        
        # Check training data
        train_examples = []
        with open(train_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    example = json.loads(line)
                    train_examples.append(example)
                    
                    # Validate format
                    if "messages" not in example:
                        issues.append(f"Train line {i+1}: Missing 'messages' field")
                    elif len(example["messages"]) != 3:
                        issues.append(f"Train line {i+1}: Expected 3 messages, got {len(example['messages'])}")
                    else:
                        # Check message roles
                        roles = [msg["role"] for msg in example["messages"]]
                        if roles != ["system", "user", "assistant"]:
                            issues.append(f"Train line {i+1}: Invalid role sequence: {roles}")
                        
                        # Check content length
                        assistant_content = example["messages"][2]["content"]
                        if len(assistant_content.split()) < 50:
                            issues.append(f"Train line {i+1}: Response too short ({len(assistant_content.split())} words)")
                        
                except json.JSONDecodeError:
                    issues.append(f"Train line {i+1}: Invalid JSON")
        
        # Check validation data
        val_examples = []
        with open(val_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    example = json.loads(line)
                    val_examples.append(example)
                except json.JSONDecodeError:
                    issues.append(f"Val line {i+1}: Invalid JSON")
        
        # Summary
        print(f"Training examples: {len(train_examples)}")
        print(f"Validation examples: {len(val_examples)}")
        print(f"Format issues found: {len(issues)}")
        
        if issues:
            print("\nFirst 5 issues:")
            for issue in issues[:5]:
                print(f"  - {issue}")
        else:
            print(" All data passed validation!")
        
        return len(issues) == 0

def estimate_rft_cost(train_size: int, val_size: int, base_model: str = "gpt-4o-mini"):
    """Estimate the cost of RFT training."""
    
    # Rough estimates based on OpenAI pricing
    # These are approximations - actual costs may vary
    avg_tokens_per_example = 300  # Rough estimate
    total_tokens = (train_size + val_size) * avg_tokens_per_example
    
    # Training costs (as of 2024)
    cost_per_1k_tokens = 0.0080  # GPT-4o-mini fine-tuning
    
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    print(f"\nRFT Cost Estimate:")
    print(f"  Model: {base_model}")
    print(f"  Training examples: {train_size}")
    print(f"  Validation examples: {val_size}")
    print(f"  Estimated tokens: {total_tokens:,}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    print(f"  (Note: This is a rough estimate. Actual costs may vary.)")
    
    return estimated_cost

if __name__ == "__main__":
    # Initialize preparer
    preparer = RFTDataPreparer()
    
    # Check if graded data exists
    graded_path = os.path.join(DATA_DIR, "preferences_graded.jsonl")
    
    if not os.path.exists(graded_path):
        print("No graded preferences found. Please run rft_grader.py first.")
    else:
        # Prepare data for RFT
        output_path = os.path.join(DATA_DIR, "rft_dataset.jsonl")
        result = preparer.format_for_openai_rft(graded_path, output_path)
        
        # Validate the data
        preparer.validate_rft_data(result["train_path"], result["val_path"])
        
        # Estimate cost
        estimate_rft_cost(result["train_size"], result["val_size"])
        
        # Create contrast sets
        contrast_path = os.path.join(DATA_DIR, "rft_contrast_sets.jsonl")
        preparer.create_rft_contrast_sets(graded_path, contrast_path)
        
        # Create safety-critical dataset
        test_cases_path = os.path.join(DATA_DIR, "meditation_test_cases.json")
        if os.path.exists(test_cases_path):
            preparer.create_safety_critical_dataset(graded_path, test_cases_path)