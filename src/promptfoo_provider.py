"""
Promptfoo provider for meditation AI testing.
Integrates with the alignment pipeline for evaluation.
"""

from alignment_loop import AlignmentPipeline

# Initialize pipeline
pipeline = AlignmentPipeline()

def call_meditation_ai(prompt: str, options: dict = None, context: dict = None) -> dict:
    """
    Provider function for Promptfoo to call the meditation AI.
    
    Args:
        prompt: The meditation request
        options: Optional configuration (e.g., {"aligned": True/False})
        context: Additional context from promptfoo
    
    Returns:
        Dictionary with output key containing meditation response
    """
    if options and not options.get("aligned", True):
        # Use base model without alignment
        response = pipeline.generate_base(prompt)
    else:
        # Use aligned model (default)
        response = pipeline.generate_aligned(prompt)
    
    # Return in the format promptfoo expects
    return {"output": response}

# Export for Promptfoo
__all__ = ['call_meditation_ai']