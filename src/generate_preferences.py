"""
Generate synthetic preference data for meditation AI alignment.
Creates paired good/bad responses demonstrating safety violations.
"""

import json
import os
import time
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
from config import (OPENAI_API_KEY, CRITIC_MODEL, DATA_DIR, 
                   PREFERENCE_PAIRS_PER_TOPIC, PREFERENCE_BATCH_SIZE,
                   MAX_TOKENS_PER_BATCH, TOKEN_PRICING, MAX_API_COST_USD,
                   get_preference_generation_prompt)
from api_utils import make_api_call_with_retry

client = OpenAI(api_key=OPENAI_API_KEY)

# Track API costs
api_costs = {"total": 0.0, "calls": 0}

def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens (1 token ≈ 4 characters)."""
    return len(text) // 4

def check_cost_before_call(model: str, estimated_prompt_tokens: int, max_output_tokens: int):
    """Check if API call would exceed cost limit before making it."""
    pricing = TOKEN_PRICING.get(model, TOKEN_PRICING["gpt-4o-mini"])
    
    # Estimate worst-case cost (assuming max output tokens)
    estimated_cost = (estimated_prompt_tokens * pricing["input"]) + (max_output_tokens * pricing["output"])
    
    if api_costs["total"] + estimated_cost > MAX_API_COST_USD:
        raise RuntimeError(
            f"API call would exceed cost limit. Current: ${api_costs['total']:.2f}, "
            f"Estimated call cost: ${estimated_cost:.4f}, Limit: ${MAX_API_COST_USD}"
        )
    
    if api_costs["total"] + estimated_cost > MAX_API_COST_USD * 0.9:
        print(f" Warning: Next call may exceed budget. Current: ${api_costs['total']:.2f}")

def track_api_cost(model: str, input_tokens: int, output_tokens: int):
    """Track API costs and warn if approaching limit."""
    pricing = TOKEN_PRICING.get(model, TOKEN_PRICING["gpt-4o-mini"])
    cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
    api_costs["total"] += cost
    api_costs["calls"] += 1
    
    if api_costs["total"] > MAX_API_COST_USD * 0.8:
        print(f" Warning: API costs at ${api_costs['total']:.2f} (80% of ${MAX_API_COST_USD} limit)")
    
    return api_costs["total"]

def generate_preference_batch(topic: str, n: int = PREFERENCE_BATCH_SIZE) -> List[Dict]:
    """Generate n preference pairs for a given meditation topic using production prompt system."""
    
    # Use centralized production prompt system for consistency
    base_prompt = get_preference_generation_prompt(topic, safety_enhanced=True)
    
    # Add specific batch generation instructions
    prompt = f"""{base_prompt}

BATCH GENERATION INSTRUCTIONS:
Generate exactly {n} preference pairs as a JSON object with this structure:
{{
  "preferences": [
    {{
      "chosen": "safe meditation following all production guidelines...",
      "rejected": "problematic meditation with realistic safety violations...",
      "violation_type": "medical_advice|crisis_content|non_inclusive|boundary_violation|triggering|false_promises",
      "explanation": "specific safety issue in the rejected version"
    }}
  ]
}}

CRITICAL: Each meditation text must be a single-line string with proper JSON escaping.
- Use spaces instead of newlines within the meditation text
- Keep each meditation as one continuous paragraph
- Each meditation should be 150-300 words
- Make violations subtle but safety-critical"""
    
    # Estimate tokens to check if within limits
    prompt_tokens = estimate_tokens(prompt)
    expected_output_tokens = n * 600  # ~600 tokens per pair to ensure complete responses
    
    if prompt_tokens + expected_output_tokens > MAX_TOKENS_PER_BATCH:
        print(f"Warning: Batch size {n} may exceed token limits. Consider reducing.")
        print(f"  Estimated prompt tokens: {prompt_tokens}")
        print(f"  Expected output tokens: {expected_output_tokens}")
        print(f"  Total: {prompt_tokens + expected_output_tokens} > {MAX_TOKENS_PER_BATCH} limit")
    
    # Pre-check cost before making the call
    system_msg = """You are an expert AI safety researcher specializing in meditation and mental health content generation. Your task is to create high-quality training data for AI alignment systems.

CORE MISSION: Generate preference pairs that teach AI systems to distinguish between safe and unsafe meditation guidance. This data will be used to train safety classifiers and alignment systems.

EXPERTISE AREAS:
- Meditation practices and techniques
- Mental health considerations and contraindications  
- AI safety principles and failure modes
- Subtle but critical safety violations in wellness content

QUALITY STANDARDS:
- Create realistic, nuanced examples that challenge safety systems
- Ensure violations are subtle enough to require sophisticated detection
- Make safe versions genuinely helpful while avoiding all safety issues
- Generate content that mirrors real-world meditation guidance quality

OUTPUT REQUIREMENTS:
- Always return valid, properly formatted JSON
- Maintain consistent structure across all examples
- Ensure each preference pair clearly demonstrates the safety distinction
- Include specific, actionable explanations for each violation type

JSON FORMATTING RULES:
- MUST properly escape all special characters in strings
- Use \\n for newlines, \\" for quotes, \\\\ for backslashes
- Keep all string values on a single line (no raw line breaks)
- Ensure valid JSON that can be parsed without errors

Remember: This training data directly impacts AI safety systems that protect vulnerable users seeking meditation guidance."""
    prompt_tokens = estimate_tokens(system_msg + prompt)
    check_cost_before_call(CRITIC_MODEL, prompt_tokens, expected_output_tokens)
    
    # Use exponential backoff retry for API call
    try:
        # Try structured output first (requires newer models)
        try:
            response = make_api_call_with_retry(
                client=client,
                model=CRITIC_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=max(2000, expected_output_tokens),  # Minimum 2000 tokens to avoid cutoff
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "preference_batch",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "preferences": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "chosen": {"type": "string"},
                                            "rejected": {"type": "string"},
                                            "violation_type": {
                                                "type": "string",
                                                "enum": ["medical_advice", "crisis_content", "non_inclusive", 
                                                       "boundary_violation", "triggering", "false_promises"]
                                            },
                                            "explanation": {"type": "string"}
                                        },
                                        "required": ["chosen", "rejected", "violation_type", "explanation"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["preferences"],
                            "additionalProperties": False
                        }
                    }
                }
            )
        except Exception as e:
            # Fallback to basic JSON mode if structured output fails
            print(f"Structured output failed, falling back to json_object mode: {e}")
            response = make_api_call_with_retry(
                client=client,
                model=CRITIC_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=max(2000, expected_output_tokens),  # Minimum 2000 tokens to avoid cutoff
                response_format={"type": "json_object"}
            )
        
        # Track costs and enforce limit
        usage = response.usage
        current_cost = track_api_cost(CRITIC_MODEL, usage.prompt_tokens, usage.completion_tokens)
        if current_cost > MAX_API_COST_USD:
            raise RuntimeError(f"API cost limit exceeded: ${current_cost:.2f} > ${MAX_API_COST_USD}")
        
        content = response.choices[0].message.content
        
        # Parse JSON properly
        try:
            parsed = json.loads(content)
            
            # Validate structure
            if isinstance(parsed, dict) and "preferences" in parsed:
                preferences = parsed["preferences"]
                if isinstance(preferences, list):
                    # Validate each preference
                    valid_prefs = []
                    for pref in preferences:
                        if all(key in pref for key in ["chosen", "rejected", "violation_type", "explanation"]):
                            valid_prefs.append(pref)
                    return valid_prefs
            
            # If structure is unexpected, log and return empty
            print(f"Unexpected JSON structure from API")
            return []
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw content length: {len(content)} characters")
            
            # Try to show where the error occurred
            if hasattr(e, 'pos'):
                start = max(0, e.pos - 100)
                end = min(len(content), e.pos + 100)
                print(f"Content around error position {e.pos}:")
                print(f"...{content[start:end]}...")
            
            # Save problematic response for debugging
            debug_file = f"debug_json_error_{topic.replace(' ', '_')}_{time.time()}.txt"
            with open(debug_file, 'w') as f:
                f.write(content)
            print(f"Saved problematic response to {debug_file}")
            
            return []
            
    except Exception as e:
        print(f"Error generating batch for {topic}: {e}")
        return []

def generate_all_preferences(test_mode=False):
    """Generate preference dataset for multiple topics.
    
    Args:
        test_mode: If True, generates only 1 pair per topic for testing
    """
    
    # Full topic list
    all_topics = [
        # Core mental health conditions
        "anxiety relief",
        "depression support", 
        "trauma healing",
        "addiction recovery",
        "grief processing",
        "panic attacks",
        "sleep difficulties",
        "chronic pain",
        "self-esteem",
        "anger management",
        
        # Relationship and social issues
        "relationship conflicts",
        "social anxiety",
        "loneliness and isolation",
        "family stress",
        "workplace stress",
        "boundary setting",
        
        # Life transitions and challenges
        "career transitions",
        "divorce and breakups",
        "parenting stress",
        "aging and mortality",
        "financial stress",
        "academic pressure",
        
        # Spiritual and existential topics
        "spiritual awakening",
        "meaning and purpose",
        "forgiveness practices",
        "ego dissolution",
        "death contemplation",
        "religious conflict",
        
        # Physical and health-related
        "illness and healing",
        "body image issues",
        "eating disorders",
        "chronic fatigue",
        "pain management",
        "surgery recovery",
        
        # Specific populations
        "teen meditation",
        "elder care meditation",
        "pregnancy meditation",
        "veterans and PTSD",
        "healthcare workers",
        "first responders",
        
        # Advanced practices (higher risk)
        "breathwork and pranayama",
        "visualization techniques",
        "chakra meditation",
        "energy healing",
        "past life regression",
        "shadow work",
        
        # Crisis and emergency situations
        "suicidal ideation",
        "self-harm urges",
        "psychotic episodes",
        "dissociation",
        "flashbacks",
        "panic disorder"
    ]
    
    # In test mode, use only 5 representative topics
    if test_mode:
        topics = [
            "anxiety relief",         # Common use case
            "suicidal ideation",      # Critical safety test
            "chronic pain",           # Medical boundary test
            "teen meditation",        # Vulnerable population test
            "spiritual awakening"     # Non-medical wellness test
        ]
    else:
        topics = all_topics
    
    all_preferences = []
    
    # Adjust pairs per topic for test mode
    pairs_per_topic = 1 if test_mode else PREFERENCE_PAIRS_PER_TOPIC
    
    # Cost estimation
    total_pairs_needed = len(topics) * pairs_per_topic
    estimated_tokens = total_pairs_needed * 2000  # ~2000 tokens per pair 
    estimated_cost = (estimated_tokens / 2000) * TOKEN_PRICING[CRITIC_MODEL]["output"]
    
    mode_str = "TEST MODE - " if test_mode else ""
    print(f"{mode_str}Generating synthetic preference dataset...")
    print(f"  Topics: {len(topics)}")
    print(f"  Pairs per topic: {pairs_per_topic}")
    print(f"  Total pairs: {total_pairs_needed}")
    print(f"  Batch size: {min(PREFERENCE_BATCH_SIZE, pairs_per_topic)}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    
    if estimated_cost > MAX_API_COST_USD:
        response = input(f"⚠️  Estimated cost (${estimated_cost:.2f}) exceeds limit (${MAX_API_COST_USD}). Continue? (y/n): ")
        if response.lower() != 'y':
            print("Generation cancelled.")
            return []
    
    for topic in tqdm(topics, desc="Topics"):
        # Generate in smaller batches for better quality
        topic_preferences = []
        batches_needed = (pairs_per_topic + PREFERENCE_BATCH_SIZE - 1) // PREFERENCE_BATCH_SIZE
        
        for i in range(batches_needed):
            # Calculate how many to generate in this batch
            remaining = pairs_per_topic - len(topic_preferences)
            batch_size = min(PREFERENCE_BATCH_SIZE, remaining)
            
            if batch_size <= 0:
                break
                
            print(f"  Generating batch {i+1}/{batches_needed} for {topic} (size: {batch_size})...")
            batch = generate_preference_batch(topic, n=batch_size)
            
            # Add metadata to each preference
            for pref in batch:
                if pref and all(key in pref for key in ["chosen", "rejected", "violation_type", "explanation"]):
                    # Validate content length
                    if len(pref["chosen"]) < 50 or len(pref["rejected"]) < 50:
                        print(f" Warning: Skipping preference with too short content")
                        continue
                        
                    topic_preferences.append({
                        **pref,
                        "topic": topic,
                        "timestamp": datetime.now().isoformat(),
                        "batch_id": f"{topic}_{i}",
                        "model": CRITIC_MODEL
                    })
            
            # Rate limiting - only in non-test mode
            if i < batches_needed - 1 and not test_mode:
                time.sleep(0.1)  # Minimal delay to avoid rate limits
        
        all_preferences.extend(topic_preferences)
        print(f"  Generated {len(topic_preferences)} preferences for {topic} (target: {pairs_per_topic})")
    
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

def generate_preferences(n_pairs=None, test_mode=False):
    """Public API for generating preferences with custom pair count.
    
    Args:
        n_pairs: Number of pairs to generate per topic (overrides config)
        test_mode: If True, generates only 1 pair per topic
    """
    if test_mode:
        return generate_all_preferences(test_mode=True)
    elif n_pairs is not None:
        # Temporarily override the config
        global PREFERENCE_PAIRS_PER_TOPIC
        original = PREFERENCE_PAIRS_PER_TOPIC
        PREFERENCE_PAIRS_PER_TOPIC = n_pairs
        try:
            return generate_all_preferences(test_mode=False)
        finally:
            PREFERENCE_PAIRS_PER_TOPIC = original
    else:
        return generate_all_preferences(test_mode=False)

if __name__ == "__main__":
    import sys
    
    # Check for test mode flag
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Generate preferences
    preferences = generate_all_preferences(test_mode=test_mode)
    
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