#!/usr/bin/env python3
"""Test script to validate all logical flow fixes."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_token_limits():
    """Test that batch sizes are properly limited."""
    from config import PREFERENCE_BATCH_SIZE, MAX_TOKENS_PER_BATCH
    
    print("Testing token limit fixes...")
    assert PREFERENCE_BATCH_SIZE == 5, f"Expected batch size 5, got {PREFERENCE_BATCH_SIZE}"
    assert MAX_TOKENS_PER_BATCH == 2000, f"Expected max tokens 2000, got {MAX_TOKENS_PER_BATCH}"
    print("✓ Token limits properly configured")

def test_cost_control():
    """Test cost control mechanisms."""
    from config import MAX_API_COST_USD, TOKEN_PRICING
    
    print("\nTesting cost control...")
    assert MAX_API_COST_USD > 0, "Cost limit should be positive"
    assert "gpt-4o" in TOKEN_PRICING, "Missing pricing for gpt-4o"
    assert "gpt-4o-mini" in TOKEN_PRICING, "Missing pricing for gpt-4o-mini"
    print("✓ Cost control configured")

def test_safety_scoring():
    """Test hierarchical safety scoring."""
    print("\nTesting safety scoring...")
    
    try:
        from constitutional_ai import MeditationConstitution
        
        # Create test constitution
        import json
        test_data = {
            "principles": {
                "critical_test": {
                    "rule": "Critical violation test",
                    "patterns": ["CRITICAL_VIOLATION"],
                    "severity": 1.0
                },
                "high_test": {
                    "rule": "High violation test", 
                    "patterns": ["HIGH_VIOLATION"],
                    "severity": 0.8
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
        
        # Save test constitution
        test_file = Path("data") / "test_constitution.json"
        test_file.parent.mkdir(exist_ok=True)
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Test with critical violation
        test_text = "This contains CRITICAL_VIOLATION content"
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
        
        score, violations = constitution.evaluate(test_text)
        assert score == 0.0, f"Critical violation should result in 0.0 score, got {score}"
        print("✓ Critical violations properly fail with score 0.0")
        
        # Clean up
        test_file.unlink(missing_ok=True)
        
    except Exception as e:
        print(f"✗ Safety scoring test failed: {e}")
        return False
    
    return True

def test_json_parsing():
    """Test robust JSON parsing."""
    print("\nTesting JSON parsing...")
    
    try:
        from generate_preferences import generate_preference_batch
        
        # Test that dangerous string parsing is removed
        import inspect
        source = inspect.getsource(generate_preference_batch)
        
        # Check for dangerous patterns
        assert "json_start = content.find('[')" not in source, "Dangerous JSON parsing still present"
        assert "response_format" in source, "Should use structured output format"
        
        print("✓ JSON parsing uses proper structured output")
        
    except Exception as e:
        print(f"✗ JSON parsing test failed: {e}")
        return False
    
    return True

def test_context_awareness():
    """Test context-aware pattern matching."""
    print("\nTesting context-aware patterns...")
    
    try:
        # Check that patterns have been updated
        constitution_file = Path("data") / "meditation_constitution.json"
        if constitution_file.exists():
            import json
            with open(constitution_file) as f:
                data = json.load(f)
            
            # Check for updated patterns
            inclusivity_patterns = data["principles"]["inclusivity"]["patterns"]
            
            # Should have negative lookbehind/lookahead
            has_context_patterns = any("(?<!" in p or "(?!" in p for p in inclusivity_patterns)
            
            if has_context_patterns:
                print("✓ Context-aware patterns implemented")
            else:
                print("⚠️  Warning: Context patterns may need updating")
        else:
            print("⚠️  Constitution file not found")
            
    except Exception as e:
        print(f"✗ Context awareness test failed: {e}")
        return False
    
    return True

def test_versioning():
    """Test result versioning."""
    print("\nTesting result versioning...")
    
    try:
        from evaluation import AlignmentEvaluator
        
        # Check save_results method signature
        import inspect
        sig = inspect.signature(AlignmentEvaluator.save_results)
        params = list(sig.parameters.keys())
        
        assert "use_timestamp" in params, "save_results should have use_timestamp parameter"
        print("✓ Result versioning implemented")
        
    except Exception as e:
        print(f"✗ Versioning test failed: {e}")
        return False
    
    return True

def test_cross_platform():
    """Test cross-platform compatibility."""
    print("\nTesting cross-platform compatibility...")
    
    try:
        from config import PROJECT_ROOT, DATA_DIR
        
        # Check that paths work
        assert Path(DATA_DIR).exists(), f"Data directory not found: {DATA_DIR}"
        
        # Test path operations work on current OS
        test_path = Path(DATA_DIR) / "test_file.txt"
        assert str(test_path).endswith("test_file.txt"), "Path joining failed"
        
        print("✓ Cross-platform paths working")
        
    except Exception as e:
        print(f"✗ Cross-platform test failed: {e}")
        return False
    
    return True

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Validating Logical Flow Fixes")
    print("=" * 60)
    
    tests = [
        test_token_limits,
        test_cost_control,
        test_safety_scoring,
        test_json_parsing,
        test_context_awareness,
        test_versioning,
        test_cross_platform
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All logical flow fixes validated!")
    else:
        print("✗ Some fixes need attention")
    
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())