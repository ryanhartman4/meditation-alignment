"""
Generate synthetic preference data for meditation AI alignment.
Creates paired good/bad responses demonstrating safety violations.
"""

import json
import os
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
from config import OPENAI_API_KEY, CRITIC_MODEL, DATA_DIR, PREFERENCE_PAIRS_PER_TOPIC

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_preference_batch(topic, n=50):
    """Generate n preference pairs for a given meditation topic."""
    
    prompt = f"""Generate {n} preference pairs for meditation on '{topic}'.
    For each pair, create one GOOD (safe, helpful) and one BAD (potentially harmful) response.
    
    The BAD responses should include realistic problems like:
    - Medical advice (e.g., "this will cure your anxiety")
    - Inappropriate crisis handling (e.g., "meditation is all you need for depression") 
    - Non-inclusive language (e.g., requiring physical abilities)
    - Boundary violations (e.g., overly personal language)
    - Triggering content without warnings
    - False promises or guarantees
    
    Make the differences subtle but important for safety.
    Each meditation snippet should be 100-200 words.
    
    Return as JSON array with this exact format:
    [
      {{
        "chosen": "good meditation text here...",
        "rejected": "problematic meditation text here...",
        "violation_type": "medical_advice|crisis_content|non_inclusive|boundary_violation|triggering|false_promises",
        "explanation": "why the rejected version is problematic"
      }}
    ]
    """
    
    try:
        response = client.chat.completions.create(
            model=CRITIC_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # Parse response - it should be a JSON object with a key containing the array
        try:
            parsed = json.loads(content)
            # Extract the array from whatever key it's under
            if isinstance(parsed, dict):
                # Find the key that contains a list
                for key, value in parsed.items():
                    if isinstance(value, list):
                        return value
            elif isinstance(parsed, list):
                return parsed
        except:
            # Fallback: try to extract JSON array from the content
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                return json.loads(json_content)
        
        return []
        
    except Exception as e:
        print(f"Error generating batch for {topic}: {e}")
        return []

def generate_all_preferences():
    """Generate preference dataset for multiple topics."""
    
    topics = [
        "anxiety relief",
        "depression support", 
        "trauma healing",
        "addiction recovery",
        "grief processing",
        "panic attacks",
        "sleep difficulties",
        "chronic pain",
        "self-esteem",
        "anger management"
    ]
    
    all_preferences = []
    
    print("Generating synthetic preference dataset...")
    for topic in tqdm(topics, desc="Topics"):
        # Generate in smaller batches for better quality
        topic_preferences = []
        batches_needed = PREFERENCE_PAIRS_PER_TOPIC // 50
        
        for i in range(batches_needed):
            print(f"  Generating batch {i+1}/{batches_needed} for {topic}...")
            batch = generate_preference_batch(topic, n=50)
            
            # Add metadata to each preference
            for pref in batch:
                if pref and all(key in pref for key in ["chosen", "rejected", "violation_type", "explanation"]):
                    topic_preferences.append({
                        **pref,
                        "topic": topic,
                        "timestamp": datetime.now().isoformat(),
                        "batch_id": f"{topic}_{i}"
                    })
        
        all_preferences.extend(topic_preferences)
        print(f"  Generated {len(topic_preferences)} preferences for {topic}")
    
    # Save to JSONL file
    output_path = os.path.join(DATA_DIR, "preferences_synthetic.jsonl")
    with open(output_path, 'w') as f:
        for pref in all_preferences:
            f.write(json.dumps(pref) + '\n')
    
    print(f"\nGenerated {len(all_preferences)} preference pairs total")
    print(f"Saved to {output_path}")
    
    # Create summary statistics
    stats = {
        "total_pairs": len(all_preferences),
        "by_topic": {},
        "by_violation": {},
        "generation_time": datetime.now().isoformat()
    }
    
    # Count by topic
    for topic in topics:
        stats["by_topic"][topic] = sum(1 for p in all_preferences if p.get("topic") == topic)
    
    # Count by violation type
    for pref in all_preferences:
        vtype = pref.get("violation_type", "unknown")
        stats["by_violation"][vtype] = stats["by_violation"].get(vtype, 0) + 1
    
    # Save statistics
    stats_path = os.path.join(DATA_DIR, "preference_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {stats_path}")
    
    return all_preferences

def validate_preferences(preferences):
    """Validate the quality of generated preferences."""
    issues = []
    
    for i, pref in enumerate(preferences):
        # Check required fields
        required_fields = ["chosen", "rejected", "violation_type", "explanation", "topic"]
        missing = [f for f in required_fields if f not in pref]
        if missing:
            issues.append(f"Preference {i}: missing fields {missing}")
            continue
        
        # Check text lengths
        if len(pref["chosen"].split()) < 50:
            issues.append(f"Preference {i}: chosen text too short")
        if len(pref["rejected"].split()) < 50:
            issues.append(f"Preference {i}: rejected text too short")
        
        # Check that chosen and rejected are different
        if pref["chosen"] == pref["rejected"]:
            issues.append(f"Preference {i}: chosen and rejected are identical")
        
        # Validate violation type
        valid_violations = ["medical_advice", "crisis_content", "non_inclusive", 
                          "boundary_violation", "triggering", "false_promises"]
        if pref["violation_type"] not in valid_violations:
            issues.append(f"Preference {i}: invalid violation type '{pref['violation_type']}'")
    
    return issues

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Generate preferences
    preferences = generate_all_preferences()
    
    # Validate
    print("\nValidating preferences...")
    issues = validate_preferences(preferences)
    
    if issues:
        print(f"\nFound {len(issues)} validation issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("All preferences passed validation!")
    
    # Show sample preference
    if preferences:
        print("\nSample preference pair:")
        sample = preferences[0]
        print(f"Topic: {sample['topic']}")
        print(f"Violation: {sample['violation_type']}")
        print(f"Chosen: {sample['chosen'][:100]}...")
        print(f"Rejected: {sample['rejected'][:100]}...")
        print(f"Explanation: {sample['explanation']}")