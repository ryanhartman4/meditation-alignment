"""
Promptfoo provider for meditation AI testing.
Integrates with the alignment pipeline for evaluation.
"""

from alignment_loop import AlignmentPipeline

# Initialize pipeline
pipeline = AlignmentPipeline()

def call_meditation_ai(prompt: str, options: dict = None) -> str:
    """
    Provider function for Promptfoo to call the meditation AI.
    
    Args:
        prompt: The meditation request
        options: Optional configuration (e.g., {"aligned": True/False})
    
    Returns:
        Generated meditation response
    """
    if options and not options.get("aligned", True):
        # Use base model without alignment
        return pipeline.generate_base(prompt)
    else:
        # Use aligned model (default)
        return pipeline.generate_aligned(prompt)

# Export for Promptfoo
__all__ = ['call_meditation_ai']