"""
Constitutional AI system for meditation content safety.
Implements pattern-based safety rules and content rewriting.
"""

import re
import json
import os
from typing import List, Dict, Tuple
from openai import OpenAI
from config import OPENAI_API_KEY, BASE_MODEL, DATA_DIR

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
        
        # Compile regex patterns for efficiency
        for principle_name, principle_data in self.principles.items():
            principle_data["compiled_patterns"] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in principle_data["patterns"]
            ]
        
        # Add context-aware exceptions
        self.context_exceptions = {
            "inclusivity": [
                # Phrases that make physical instructions optional
                "if you can", "when you're able", "for those who can",
                "if comfortable", "optional:", "alternatively",
                # Mental/visualization context
                "in your mind", "imagine", "visualize", "picture yourself"
            ],
            "medical_safety": [
                # Educational context
                "speak with your doctor", "consult a professional",
                "medical supervision", "healthcare provider"
            ]
        }
    
    def _check_context_exceptions(self, text: str, principle: str, match_pos: int) -> bool:
        """Check if a pattern match should be excepted due to context."""
        if principle not in self.context_exceptions:
            return False
            
        # Get surrounding context (100 chars before and after)
        context_start = max(0, match_pos - 100)
        context_end = min(len(text), match_pos + 100)
        context = text[context_start:context_end].lower()
        
        # Check if any exception phrases appear near the match
        for exception in self.context_exceptions[principle]:
            if exception.lower() in context:
                return True
        return False
    
    def evaluate(self, text: str) -> Tuple[float, List[Dict]]:
        """Evaluate text for constitutional violations with hierarchical scoring."""
        violations = []
        critical_violations = 0
        high_violations = 0
        medium_violations = 0
        low_violations = 0
        
        for principle_name, principle_data in self.principles.items():
            for i, pattern in enumerate(principle_data["compiled_patterns"]):
                matches = list(pattern.finditer(text))
                for match in matches:
                    # Check context exceptions
                    if self._check_context_exceptions(text, principle_name, match.start()):
                        continue
                        
                    violations.append({
                        "principle": principle_name,
                        "rule": principle_data["rule"],
                        "matched_pattern": principle_data["patterns"][i],
                        "matched_text": match.group(),
                        "position": match.start(),
                        "severity": principle_data["severity"]
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
        
        # Hierarchical safety score calculation
        # ANY critical violation = immediate failure
        if critical_violations > 0:
            safety_score = 0.0
        else:
            # Calculate score with exponential penalties
            # High severity violations have much stronger impact
            high_penalty = 1 - (0.7 ** high_violations) if high_violations > 0 else 0
            medium_penalty = 1 - (0.85 ** medium_violations) if medium_violations > 0 else 0
            low_penalty = 1 - (0.95 ** low_violations) if low_violations > 0 else 0
            
            # Weighted combination with high violations having most impact
            total_penalty = (high_penalty * 0.6) + (medium_penalty * 0.3) + (low_penalty * 0.1)
            safety_score = max(0, 1.0 - total_penalty)
        
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
            if pattern.search(text):
                return {
                    "is_crisis": True,
                    "resources": self.principles["crisis_safety"]["crisis_resources"]
                }
        
        return {"is_crisis": False, "resources": None}
    
    def apply_constitution(self, text: str, context: str = "") -> str:
        """Rewrite text to follow constitutional principles."""
        
        safety_score, violations = self.evaluate(text)
        
        # If text is safe enough, return as is
        if safety_score >= 0.95 and not violations:
            return text
        
        # Check for crisis content first
        crisis_check = self.check_crisis_content(context)
        if crisis_check["is_crisis"]:
            # Use crisis response template
            return self.response_templates["crisis_response"].format(
                crisis_resource=crisis_check["resources"]["General"]
            )
        
        # Check for medical content
        medical_keywords = ["medication", "antidepressant", "therapy", "doctor", "pain", "cure"]
        if any(keyword in context.lower() for keyword in medical_keywords):
            # Use medical redirect template
            condition = next((word for word in medical_keywords if word in context.lower()), "concern")
            return self.response_templates["medical_redirect"].format(condition=condition)
        
        # Build constitutional rewrite prompt
        violated_rules = list(set(v["rule"] for v in violations))
        
        # Get alternatives for common violations
        alternatives = []
        if "inclusivity" in [v["principle"] for v in violations]:
            alternatives.extend([
                f"- Replace '{k}' with '{v}'" 
                for k, v in self.principles["inclusivity"].get("alternatives", {}).items()
            ])
        
        prompt = f"""Rewrite this meditation to follow these safety principles:
{chr(10).join(f"- {rule}" for rule in violated_rules)}

Additionally, always follow these rules:
{chr(10).join(f"- {p['rule']}" for p in self.principles.values())}

Specific corrections needed:
{chr(10).join(alternatives) if alternatives else "Make appropriate safety corrections"}

Original meditation:
{text}

Context: {context}

Create a safe, helpful meditation that addresses the same topic without violating any principles.
Keep the same general structure and length. Focus on being supportive and inclusive."""
        
        try:
            response = client.chat.completions.create(
                model=BASE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            rewritten = response.choices[0].message.content
            
            # Check if rewrite is safe
            new_safety_score, new_violations = self.evaluate(rewritten)
            
            # If still has critical violations, return generic safe version
            if new_safety_score < 0.8 or any(v["severity"] >= 1.0 for v in new_violations):
                return self.response_templates["generic_safe"]
            
            return rewritten
            
        except Exception as e:
            print(f"Error applying constitution: {e}")
            return self.response_templates["generic_safe"]
    
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