"""
Constitutional AI system for meditation content safety.
Implements pattern-based safety rules and content rewriting with improved regex handling.
"""

import re
import json
import os
from typing import List, Dict, Tuple
from openai import OpenAI
from config import OPENAI_API_KEY, BASE_MODEL, DATA_DIR
from api_utils import make_api_call_with_retry

client = OpenAI(api_key=OPENAI_API_KEY)

class MeditationConstitution:
    """Constitutional AI system for meditation content safety."""
    
    def __init__(self):
        # Load constitution from JSON file
        constitution_path = os.path.join(DATA_DIR, "meditation_constitution.json")
        try:
            with open(constitution_path, 'r') as f:
                self.constitution_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Constitution file not found at {constitution_path}. Please ensure meditation_constitution.json exists.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in constitution file: {e}")
        
        self.principles = self.constitution_data["principles"]
        self.safety_thresholds = self.constitution_data["safety_thresholds"]
        self.response_templates = self.constitution_data["response_templates"]
        
        # Compile regex patterns for efficiency (both main patterns and exceptions)
        self._compile_patterns()
        
        print(f"Loaded Constitutional AI v{self.constitution_data.get('version', '1.0')} with {len(self.principles)} principles")
    
    def _compile_patterns(self):
        """Compile all patterns and exception patterns with improved error handling."""
        for principle_name, principle_data in self.principles.items():
            # Compile main patterns
            principle_data["compiled_patterns"] = []
            failed_patterns = []
            
            for pattern in principle_data["patterns"]:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                    principle_data["compiled_patterns"].append(compiled)
                except re.error as e:
                    failed_patterns.append((pattern, str(e)))
                    # Create a simple word boundary version as fallback
                    simple_pattern = pattern.replace("\\b", "").replace("\\", "")
                    simple_pattern = f"\\b{re.escape(simple_pattern)}\\b"
                    try:
                        compiled = re.compile(simple_pattern, re.IGNORECASE)
                        principle_data["compiled_patterns"].append(compiled)
                        print(f"Warning: Simplified pattern for {principle_name}: {pattern} -> {simple_pattern}")
                    except re.error:
                        print(f"Error: Could not compile pattern for {principle_name}: {pattern}")
            
            # Compile exception patterns
            principle_data["compiled_exceptions"] = []
            if "exception_patterns" in principle_data:
                for pattern in principle_data["exception_patterns"]:
                    try:
                        compiled = re.compile(pattern, re.IGNORECASE)
                        principle_data["compiled_exceptions"].append(compiled)
                    except re.error as e:
                        print(f"Warning: Could not compile exception pattern for {principle_name}: {pattern} ({e})")
            
            if failed_patterns:
                print(f"Notice: {len(failed_patterns)} patterns simplified for {principle_name}")
    
    def _is_exception_match(self, text: str, principle_name: str, match_start: int, match_end: int) -> bool:
        """Check if a pattern match should be excepted due to exception patterns."""
        principle_data = self.principles[principle_name]
        
        if "compiled_exceptions" not in principle_data:
            return False
        
        # Get surrounding context (200 chars before and after match)
        context_start = max(0, match_start - 200)
        context_end = min(len(text), match_end + 200)
        context = text[context_start:context_end]
        
        # Check if any exception patterns match in the context
        for exception_pattern in principle_data["compiled_exceptions"]:
            if exception_pattern.search(context):
                return True
        
        return False
    
    def evaluate(self, text: str) -> Tuple[float, List[Dict]]:
        """Evaluate text for constitutional violations with improved exception handling."""
        violations = []
        critical_violations = 0
        high_violations = 0
        medium_violations = 0
        low_violations = 0
        
        for principle_name, principle_data in self.principles.items():
            for i, pattern in enumerate(principle_data["compiled_patterns"]):
                matches = list(pattern.finditer(text))
                for match in matches:
                    # Check if this match should be excepted
                    if self._is_exception_match(text, principle_name, match.start(), match.end()):
                        continue
                        
                    violations.append({
                        "principle": principle_name,
                        "rule": principle_data["rule"],
                        "matched_pattern": principle_data["patterns"][i] if i < len(principle_data["patterns"]) else "compiled pattern",
                        "matched_text": match.group(),
                        "position": match.start(),
                        "severity": principle_data["severity"],
                        "context": text[max(0, match.start()-50):match.end()+50]
                    })
                    
                    # Count violations by severity
                    if principle_data["severity"] >= 1.0:
                        critical_violations += 1
                    elif principle_data["severity"] >= 0.8:
                        high_violations += 1
                    elif principle_data["severity"] >= 0.5:
                        medium_violations += 1
                    else:
                        low_violations += 1
        
        # Hierarchical safety score calculation with proper mathematical foundation
        # ANY critical violation = immediate failure
        if critical_violations > 0:
            safety_score = 0.0
        else:
            # Calculate score using multiplicative penalties
            # This ensures each violation reduces the score proportionally
            safety_score = 1.0
            
            # Apply high severity penalties (each reduces score by 30%)
            safety_score *= (0.7 ** high_violations)
            
            # Apply medium severity penalties (each reduces score by 15%)
            safety_score *= (0.85 ** medium_violations)
            
            # Apply low severity penalties (each reduces score by 5%)
            safety_score *= (0.95 ** low_violations)
            
            # Ensure score stays in valid range [0, 1]
            safety_score = max(0.0, min(1.0, safety_score))
        
        return safety_score, violations
    
    def get_safety_level(self, safety_score: float) -> str:
        """Get safety level category from score."""
        for level in ["critical", "high", "medium", "low"]:
            if safety_score <= self.safety_thresholds[level]:
                return level
        return "safe"
    
    def get_detailed_report(self, text: str) -> Dict:
        """Get comprehensive safety evaluation report."""
        safety_score, violations = self.evaluate(text)
        safety_level = self.get_safety_level(safety_score)
        
        # Group violations by principle
        violations_by_principle = {}
        for v in violations:
            principle = v["principle"]
            if principle not in violations_by_principle:
                violations_by_principle[principle] = []
            violations_by_principle[principle].append(v)
        
        # Count violations by severity
        severity_counts = {
            "critical": sum(1 for v in violations if v["severity"] >= 1.0),
            "high": sum(1 for v in violations if 0.8 <= v["severity"] < 1.0),
            "medium": sum(1 for v in violations if 0.5 <= v["severity"] < 0.8),
            "low": sum(1 for v in violations if v["severity"] < 0.5)
        }
        
        return {
            "safety_score": safety_score,
            "safety_level": safety_level,
            "total_violations": len(violations),
            "violations_by_severity": severity_counts,
            "violations_by_principle": violations_by_principle,
            "violations": violations,
            "passed": safety_score >= self.safety_thresholds["low"],
            "requires_intervention": safety_level in ["critical", "high"]
        }
    
    def check_crisis_content(self, text: str) -> Dict:
        """Special check for crisis content that needs immediate resources."""
        crisis_patterns = self.principles["crisis_safety"]["compiled_patterns"]
        
        for pattern in crisis_patterns:
            match = pattern.search(text)
            if match:
                # Check if this is excepted
                if not self._is_exception_match(text, "crisis_safety", match.start(), match.end()):
                    return {
                        "is_crisis": True,
                        "resources": self.principles["crisis_safety"]["crisis_resources"]
                    }
        
        return {"is_crisis": False, "resources": None}
    
    def apply_constitution(self, text: str, context: str = "") -> str:
        """Apply minimal corrections to text for safety while preserving meditation quality."""
        
        safety_score, violations = self.evaluate(text)
        
        # Only intervene for serious violations - raised threshold
        if safety_score >= 0.85 or not any(v["severity"] >= 0.8 for v in violations):
            return text
        
        # Check for crisis content - this always requires intervention
        crisis_check_context = self.check_crisis_content(context)
        crisis_check_text = self.check_crisis_content(text)
        
        if crisis_check_context["is_crisis"] or crisis_check_text["is_crisis"]:
            # Use crisis response template
            resources = crisis_check_context["resources"] if crisis_check_context["is_crisis"] else crisis_check_text["resources"]
            return self.response_templates["crisis_response"].format(
                crisis_resource=resources["General"]
            )
        
        # For severe medical violations, use redirect template
        medical_violations = [v for v in violations if v["principle"] == "medical_safety" and v["severity"] >= 1.0]
        if medical_violations:
            medical_keywords = ["medication", "antidepressant", "therapy", "doctor", "pain", "cure"]
            condition = next((word for word in medical_keywords if word in context.lower()), "concern")
            return self.response_templates["medical_redirect"].format(condition=condition)
        
        # For other violations, attempt minimal corrections
        # Focus only on critical and high severity issues
        critical_violations = [v for v in violations if v["severity"] >= 0.8]
        
        if not critical_violations:
            return text  # Medium/low violations can pass
        
        # Create a more gentle, preservation-focused prompt
        specific_issues = []
        for v in critical_violations[:3]:  # Limit to top 3 issues
            specific_issues.append(f"- {v['matched_text']}: {v['rule']}")
        
        prompt = f"""Make minimal edits to this meditation to address only these specific safety concerns:
{chr(10).join(specific_issues)}

IMPORTANT: 
- Preserve the original tone, flow, and structure as much as possible
- Make the smallest changes necessary to address the safety issues
- Keep the same meditation style and voice
- Don't add unnecessary warnings or disclaimers
- Focus on rephrasing problematic parts rather than removing content

Original meditation:
{text}

Provide the lightly edited meditation that addresses only the specific issues mentioned above."""
        
        try:
            response = make_api_call_with_retry(
                client=client,
                model=BASE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for more conservative edits
                max_tokens=1200
            )
            
            rewritten = response.choices[0].message.content
            
            # Quick safety check - only reject if critical violations remain
            new_safety_score, new_violations = self.evaluate(rewritten)
            
            # Only fall back to generic if critical violations persist
            if any(v["severity"] >= 1.0 for v in new_violations):
                return self.get_generic_safe_meditation(context)
            
            return rewritten
            
        except Exception as e:
            print(f"Error applying minimal corrections: {e}")
            # On error, return original if it's not too unsafe
            if safety_score >= 0.7:
                return text
            return self.get_generic_safe_meditation(context)
    
    def explain_violations(self, violations: List[Dict]) -> str:
        """Create human-readable explanation of violations."""
        if not violations:
            return "No safety violations detected."
        
        explanations = []
        
        # Group by principle
        by_principle = {}
        for v in violations:
            if v["principle"] not in by_principle:
                by_principle[v["principle"]] = []
            by_principle[v["principle"]].append(v)
        
        for principle, viols in by_principle.items():
            severity = viols[0]["severity"]
            severity_text = "Critical" if severity >= 1.0 else "High" if severity >= 0.7 else "Medium"
            
            explanations.append(
                f"{severity_text} - {principle.replace('_', ' ').title()}: "
                f"{viols[0]['rule']} ({len(viols)} instance{'s' if len(viols) > 1 else ''})"
            )
        
        return "\n".join(explanations)
    
    def get_safety_report(self, text: str) -> Dict:
        """Generate comprehensive safety report for text."""
        safety_score, violations = self.evaluate(text)
        safety_level = self.get_safety_level(safety_score)
        crisis_check = self.check_crisis_content(text)
        
        return {
            "safety_score": safety_score,
            "safety_level": safety_level,
            "total_violations": len(violations),
            "critical_violations": sum(1 for v in violations if v["severity"] >= 1.0),
            "violations_by_principle": {
                p: sum(1 for v in violations if v["principle"] == p)
                for p in self.principles.keys()
            },
            "has_crisis_content": crisis_check["is_crisis"],
            "explanation": self.explain_violations(violations),
            "detailed_violations": violations
        }
    
    def get_generic_safe_meditation(self, context: str = "") -> str:
        """Get a generic safe meditation, with crisis handling if needed."""
        # Check if the context contains crisis content
        crisis_check = self.check_crisis_content(context)
        if crisis_check["is_crisis"]:
            return self.response_templates["crisis_response"].format(
                crisis_resource=crisis_check["resources"]["General"]
            )
        
        # Check for medical context
        medical_keywords = ["medication", "antidepressant", "therapy", "doctor", "pain", "cure"]
        if any(keyword in context.lower() for keyword in medical_keywords):
            condition = next((word for word in medical_keywords if word in context.lower()), "concern")
            return self.response_templates["medical_redirect"].format(condition=condition)
        
        # Return generic safe meditation
        return self.response_templates["generic_safe"]

# Test the constitution if run directly
if __name__ == "__main__":
    constitution = MeditationConstitution()
    
    # Test cases
    test_texts = [
        ("This meditation will cure your depression and you won't need therapy anymore.", "depression"),
        ("If you're feeling suicidal, just breathe deeply and it will pass.", "suicidal thoughts"),
        ("Stand up and walk around the room while breathing.", "movement meditation"),
        ("I love you and want us to have a special connection through this meditation.", "connection"),
        ("This practice is guaranteed to eliminate all your anxiety forever.", "anxiety relief"),
        ("Let's take a moment to breathe together and find some calm in this moment.", "stress relief")
    ]
    
    print("Testing Constitutional AI System\n" + "="*50)
    
    for text, context in test_texts:
        print(f"\nContext: {context}")
        print(f"Original: {text[:100]}...")
        
        # Get safety report
        report = constitution.get_safety_report(text)
        print(f"\nSafety Report:")
        print(f"  Score: {report['safety_score']:.2f} ({report['safety_level']})")
        print(f"  Violations: {report['total_violations']} ({report['critical_violations']} critical)")
        print(f"  Explanation: {report['explanation']}")
        
        # Apply constitution if needed
        if report['safety_score'] < 0.9:
            rewritten = constitution.apply_constitution(text, context)
            print(f"\nRewritten: {rewritten[:200]}...")
            
            # Check rewritten safety
            new_report = constitution.get_safety_report(rewritten)
            print(f"\nRewritten Safety Score: {new_report['safety_score']:.2f} ({new_report['safety_level']})")
        
        print("\n" + "-"*50)