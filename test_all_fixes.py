#!/usr/bin/env python3
"""
Comprehensive test script to verify all fixes implemented.
Tests all critical issues identified in the error review.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_cost_control():
    """Test that cost control properly prevents overspending."""
    from generate_preferences import check_cost_before_call, api_costs
    from config import MAX_API_COST_USD, TOKEN_PRICING
    
    print("\nTesting Cost Control...")
    
    # Reset costs
    api_costs["total"] = 0.0
    api_costs["calls"] = 0
    
    # Test normal call
    try:
        check_cost_before_call("gpt-4o", 1000, 1000)
        print("[PASS] Normal cost check passed")
    except Exception as e:
        print(f"[FAIL] Normal cost check failed: {e}")
        return False
    
    # Set costs near limit
    api_costs["total"] = MAX_API_COST_USD - 0.01
    
    # Test call that would exceed limit
    try:
        check_cost_before_call("gpt-4o", 10000, 10000)
        print("[FAIL] Should have blocked expensive call")
        return False
    except RuntimeError as e:
        print(f"[PASS] Correctly blocked expensive call: {e}")
    
    return True

def test_safety_scoring():
    """Test the improved safety scoring logic."""
    print("\nTesting Safety Scoring...")
    
    try:
        from constitutional_ai import MeditationConstitution
        import json
        
        # Create test constitution with known violations
        test_data = {
            "principles": {
                "critical_test": {
                    "rule": "No medical claims",
                    "patterns": ["cure", "treat", "replace therapy"],
                    "severity": 1.0  # Critical
                },
                "high_test": {
                    "rule": "No physical assumptions",
                    "patterns": ["stand up", "walk"],
                    "severity": 0.8  # High
                },
                "medium_test": {
                    "rule": "Inclusive language",
                    "patterns": ["see the"],
                    "severity": 0.6  # Medium
                }
            },
            "safety_thresholds": {
                "critical": 0.3,
                "high": 0.5,
                "medium": 0.7,
                "low": 0.9
            },
            "response_templates": {}
        }
        
        # Create test constitution
        constitution = MeditationConstitution.__new__(MeditationConstitution)
        constitution.constitution_data = test_data
        constitution.principles = test_data["principles"]
        constitution.safety_thresholds = test_data["safety_thresholds"]
        
        # Compile patterns
        import re
        for principle_name, principle_data in constitution.principles.items():
            principle_data["compiled_patterns"] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in principle_data["patterns"]
            ]
        
        # Test 1: Critical violation should result in 0.0 score
        critical_text = "This meditation will cure your anxiety."
        score, violations = constitution.evaluate(critical_text)
        assert score == 0.0, f"Critical violation should score 0.0, got {score}"
        print("[PASS] Critical violations correctly fail with 0.0 score")
        
        # Test 2: Multiple non-critical violations should reduce score multiplicatively
        multi_text = "Stand up and walk around. See the beautiful sky."
        score, violations = constitution.evaluate(multi_text)
        # Should have 1 high (0.8) and 1 medium (0.6) violation
        # Score = 1.0 * 0.7 * 0.85 = 0.595
        expected_score = 0.7 * 0.85
        assert abs(score - expected_score) < 0.01, f"Expected {expected_score}, got {score}"
        print(f"[PASS] Multiple violations correctly scored: {score:.3f}")
        
        # Test 3: No violations should score 1.0
        safe_text = "Let's practice mindful breathing together."
        score, violations = constitution.evaluate(safe_text)
        assert score == 1.0, f"Safe text should score 1.0, got {score}"
        print("[PASS] Safe text correctly scores 1.0")
        
    except Exception as e:
        print(f"[FAIL] Safety scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_exponential_backoff():
    """Test exponential backoff retry logic."""
    print("\nTesting Exponential Backoff...")
    
    try:
        from api_utils import exponential_backoff_retry
        from openai import RateLimitError
        import time
        
        # Track retry attempts
        attempts = []
        
        def failing_function():
            attempts.append(time.time())
            if len(attempts) < 3:
                raise RateLimitError("Rate limit exceeded")
            return "success"
        
        # Test retry logic
        start_time = time.time()
        result = exponential_backoff_retry(failing_function, max_retries=3, initial_delay=0.1)
        total_time = time.time() - start_time
        
        assert result == "success", "Function should eventually succeed"
        assert len(attempts) == 3, f"Expected 3 attempts, got {len(attempts)}"
        assert total_time > 0.2, "Should have delays between retries"
        
        print(f"[PASS] Exponential backoff working correctly ({len(attempts)} attempts, {total_time:.2f}s total)")
        
    except Exception as e:
        print(f"[FAIL] Exponential backoff test failed: {e}")
        return False
    
    return True

def test_path_security():
    """Test path traversal prevention."""
    print("\nTesting Path Security...")
    
    try:
        from path_utils import validate_safe_path, safe_join_path, PathSecurityError
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Test 1: Normal path should work
            safe_path = safe_join_path(base_dir, "subdir", "file.txt")
            assert str(safe_path).startswith(str(base_dir)), "Path should be within base dir"
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
            
    except Exception as e:
        print(f"[FAIL] Path security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_json_parsing():
    """Test robust JSON parsing implementation."""
    print("\nTesting JSON Parsing...")
    
    try:
        # Check that dangerous string parsing is removed
        from generate_preferences import generate_preference_batch
        import inspect
        
        source = inspect.getsource(generate_preference_batch)
        
        # Check for dangerous patterns
        dangerous_patterns = [
            "json_start = content.find('[')",
            "json_end = content.rfind(']')",
            "content[json_start:json_end+1]"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in source:
                print(f"[FAIL] Dangerous pattern still present: {pattern}")
                return False
        
        # Check for safe patterns
        if "response_format" not in source:
            print("[FAIL] Missing structured output format")
            return False
        
        if "json.loads" not in source:
            print("[FAIL] Missing proper JSON parsing")
            return False
        
        print("[PASS] JSON parsing uses safe methods")
        
    except Exception as e:
        print(f"[FAIL] JSON parsing test failed: {e}")
        return False
    
    return True

def test_error_handling():
    """Test comprehensive error handling."""
    print("\nTesting Error Handling...")
    
    try:
        from alignment_loop import AlignmentPipeline
        
        # Check that base generation raises exceptions instead of returning empty string
        pipeline = AlignmentPipeline()
        
        # Mock a failing API call
        def mock_failing_call(*args, **kwargs):
            raise Exception("API Error")
        
        # Temporarily replace the API call
        from api_utils import make_api_call_with_retry
        original_call = make_api_call_with_retry
        
        # This is a bit hacky but tests the error handling
        import alignment_loop
        alignment_loop.make_api_call_with_retry = mock_failing_call
        
        try:
            result = pipeline.generate_base("test prompt")
            print("[FAIL] Should have raised an exception")
            return False
        except RuntimeError as e:
            print(f"[PASS] Correctly raises exception: {e}")
        finally:
            # Restore original
            alignment_loop.make_api_call_with_retry = original_call
        
    except Exception as e:
        print(f"[FAIL] Error handling test failed: {e}")
        return False
    
    return True

def test_result_versioning():
    """Test result versioning with timestamps."""
    print("\nTesting Result Versioning...")
    
    try:
        from evaluation import AlignmentEvaluator
        import tempfile
        import json
        from datetime import datetime
        
        evaluator = AlignmentEvaluator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override RESULTS_DIR
            import evaluation
            original_results_dir = evaluation.RESULTS_DIR
            evaluation.RESULTS_DIR = temp_dir
            
            try:
                # Save a test result
                test_data = {"test": "data", "score": 0.95}
                evaluator.save_results(test_data, "test_results.json", use_timestamp=True)
                
                # Check that versioned directory was created
                results_path = Path(temp_dir)
                run_dirs = list(results_path.glob("run_*"))
                assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}"
                print("[PASS] Versioned directory created")
                
                # Check that latest symlink/directory exists
                latest_dir = results_path / "latest"
                assert latest_dir.exists(), "Latest directory should exist"
                print("[PASS] Latest directory created")
                
                # Check that file exists in both locations
                versioned_file = run_dirs[0] / "test_results.json"
                latest_file = latest_dir / "test_results.json"
                assert versioned_file.exists(), "Versioned file should exist"
                assert latest_file.exists(), "Latest file should exist"
                print("[PASS] Files saved to both locations")
                
            finally:
                # Restore original
                evaluation.RESULTS_DIR = original_results_dir
        
    except Exception as e:
        print(f"[FAIL] Result versioning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing All Critical Fixes")
    print("=" * 60)
    
    tests = [
        ("Cost Control", test_cost_control),
        ("Safety Scoring", test_safety_scoring),
        ("Exponential Backoff", test_exponential_backoff),
        ("Path Security", test_path_security),
        ("JSON Parsing", test_json_parsing),
        ("Error Handling", test_error_handling),
        ("Result Versioning", test_result_versioning)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print('='*40)
        
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
    print(f"FINAL RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("[PASS] All critical fixes verified!")
    else:
        print("[FAIL] Some fixes need attention")
    
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())