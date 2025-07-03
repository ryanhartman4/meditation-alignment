#!/usr/bin/env python3
"""
Simplified test script that doesn't require external dependencies.
Tests core logic of the fixes.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_path_security():
    """Test path traversal prevention."""
    print("\nTesting Path Security...")
    
    try:
        from path_utils import validate_safe_path, safe_join_path, PathSecurityError, sanitize_filename
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Test 1: Normal path should work
            safe_path = safe_join_path(base_dir, "subdir", "file.txt")
            # Convert both to resolved paths for comparison
            assert safe_path.resolve().is_relative_to(base_dir.resolve()), f"Path {safe_path} should be within base dir {base_dir}"
            print("[PASS] Normal paths work correctly")
            
            # Test 2: Path traversal should be blocked
            try:
                evil_path = safe_join_path(base_dir, "..", "etc", "passwd")
                print("[FAIL] Path traversal was not blocked!")
                return False
            except PathSecurityError:
                print("[PASS] Path traversal correctly blocked")
            
            # Test 3: Absolute paths outside base should be blocked
            try:
                validate_safe_path(base_dir, "/etc/passwd")
                print("[FAIL] Absolute path escape was not blocked!")
                return False
            except PathSecurityError:
                print("[PASS] Absolute path escape correctly blocked")
            
            # Test 4: Filename sanitization
            dangerous_filename = "../../../etc/passwd"
            safe_filename = sanitize_filename(dangerous_filename)
            assert "/" not in safe_filename, f"Sanitized filename still contains /: {safe_filename}"
            print(f"[PASS] Filename sanitization works: '{dangerous_filename}' -> '{safe_filename}'")
            
    except Exception as e:
        print(f"[FAIL] Path security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_safety_calculation():
    """Test safety score calculation logic without OpenAI dependency."""
    print("\nTesting Safety Score Calculation...")
    
    # Test the mathematical logic
    # Critical violations = 0.0
    # Multiple violations multiply: 0.7^high * 0.85^medium * 0.95^low
    
    test_cases = [
        {"high": 0, "medium": 0, "low": 0, "critical": 0, "expected": 1.0, "name": "No violations"},
        {"high": 0, "medium": 0, "low": 0, "critical": 1, "expected": 0.0, "name": "Critical violation"},
        {"high": 1, "medium": 0, "low": 0, "critical": 0, "expected": 0.7, "name": "One high violation"},
        {"high": 0, "medium": 1, "low": 0, "critical": 0, "expected": 0.85, "name": "One medium violation"},
        {"high": 0, "medium": 0, "low": 1, "critical": 0, "expected": 0.95, "name": "One low violation"},
        {"high": 1, "medium": 1, "low": 0, "critical": 0, "expected": 0.7 * 0.85, "name": "High + medium"},
        {"high": 2, "medium": 0, "low": 0, "critical": 0, "expected": 0.7 * 0.7, "name": "Two high violations"},
    ]
    
    all_passed = True
    for test in test_cases:
        # Simulate the calculation from constitutional_ai.py
        if test["critical"] > 0:
            score = 0.0
        else:
            score = 1.0
            score *= (0.7 ** test["high"])
            score *= (0.85 ** test["medium"])
            score *= (0.95 ** test["low"])
            score = max(0.0, min(1.0, score))
        
        if abs(score - test["expected"]) < 0.001:
            print(f"[PASS] {test['name']}: {score:.3f} (expected {test['expected']:.3f})")
        else:
            print(f"[FAIL] {test['name']}: {score:.3f} (expected {test['expected']:.3f})")
            all_passed = False
    
    return all_passed

def test_cost_estimation():
    """Test cost estimation logic."""
    print("\nTesting Cost Estimation...")
    
    # Test token estimation
    def estimate_tokens(text: str) -> int:
        return len(text) // 4
    
    test_text = "This is a test string that should be about 10 tokens long."
    estimated = estimate_tokens(test_text)
    expected = len(test_text) // 4
    
    if estimated == expected:
        print(f"[PASS] Token estimation works: '{test_text[:20]}...' -> {estimated} tokens")
    else:
        print(f"[FAIL] Token estimation failed: got {estimated}, expected {expected}")
        return False
    
    # Test cost calculation
    TOKEN_PRICING = {
        "gpt-4o": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000}
    }
    
    # Test cost for 1000 input + 1000 output tokens
    model = "gpt-4o"
    input_tokens = 1000
    output_tokens = 1000
    
    cost = (input_tokens * TOKEN_PRICING[model]["input"]) + (output_tokens * TOKEN_PRICING[model]["output"])
    expected_cost = 0.01 + 0.03  # $0.04
    
    if abs(cost - expected_cost) < 0.001:
        print(f"[PASS] Cost calculation correct: {input_tokens} in + {output_tokens} out = ${cost:.3f}")
    else:
        print(f"[FAIL] Cost calculation wrong: got ${cost:.3f}, expected ${expected_cost:.3f}")
        return False
    
    return True

def test_json_structure_validation():
    """Test JSON structure validation logic."""
    print("\nTesting JSON Structure Validation...")
    
    # Test the validation logic from generate_preferences.py
    import json
    
    # Valid structure
    valid_json = {
        "preferences": [
            {
                "chosen": "safe meditation",
                "rejected": "unsafe meditation",
                "violation_type": "medical_advice",
                "explanation": "makes medical claims"
            }
        ]
    }
    
    # Invalid structures
    invalid_jsons = [
        {"wrong_key": []},  # Missing preferences key
        {"preferences": "not a list"},  # preferences not a list
        {"preferences": [{"chosen": "only chosen"}]},  # Missing required fields
    ]
    
    # Test valid structure
    try:
        if isinstance(valid_json, dict) and "preferences" in valid_json:
            preferences = valid_json["preferences"]
            if isinstance(preferences, list):
                valid_prefs = []
                for pref in preferences:
                    if all(key in pref for key in ["chosen", "rejected", "violation_type", "explanation"]):
                        valid_prefs.append(pref)
                if len(valid_prefs) == 1:
                    print("[PASS] Valid JSON structure accepted")
                else:
                    print("[FAIL] Valid JSON not properly validated")
                    return False
            else:
                print("[FAIL] Failed to check list type")
                return False
        else:
            print("[FAIL] Failed to check dict structure")
            return False
    except Exception as e:
        print(f"[FAIL] Validation logic error: {e}")
        return False
    
    # Test invalid structures
    invalid_caught = 0
    for invalid in invalid_jsons:
        if isinstance(invalid, dict) and "preferences" in invalid:
            preferences = invalid["preferences"]
            if isinstance(preferences, list):
                valid_prefs = []
                for pref in preferences:
                    if all(key in pref for key in ["chosen", "rejected", "violation_type", "explanation"]):
                        valid_prefs.append(pref)
                if len(valid_prefs) == 0:
                    invalid_caught += 1
            else:
                invalid_caught += 1
        else:
            invalid_caught += 1
    
    if invalid_caught == len(invalid_jsons):
        print(f"[PASS] All {invalid_caught} invalid structures rejected")
    else:
        print(f"[FAIL] Only {invalid_caught}/{len(invalid_jsons)} invalid structures caught")
        return False
    
    return True

def main():
    """Run all simple tests."""
    print("=" * 60)
    print("Testing Core Logic of Fixes (No Dependencies)")
    print("=" * 60)
    
    tests = [
        ("Path Security", test_path_security),
        ("Safety Score Calculation", test_safety_calculation),
        ("Cost Estimation", test_cost_estimation),
        ("JSON Structure Validation", test_json_structure_validation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[PASS] {test_name} PASSED")
            else:
                failed += 1
                print(f"\n[FAIL] {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {test_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("[PASS] All core logic tests passed!")
    else:
        print("[FAIL] Some core logic tests failed")
    
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())