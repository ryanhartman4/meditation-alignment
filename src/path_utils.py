"""
Utilities for secure path handling to prevent path traversal attacks.
"""

import os
from pathlib import Path
from typing import Union, Optional

class PathSecurityError(Exception):
    """Raised when a path operation would be insecure."""
    pass

def validate_safe_path(base_dir: Union[str, Path], requested_path: Union[str, Path]) -> Path:
    """
    Validate that a requested path is safe and within the base directory.
    
    Args:
        base_dir: The base directory that paths must be within
        requested_path: The requested path (can be relative or absolute)
        
    Returns:
        A validated Path object that is guaranteed to be within base_dir
        
    Raises:
        PathSecurityError: If the path would escape the base directory
    """
    base_dir = Path(base_dir).resolve()
    
    # If requested_path is relative, join it with base_dir
    if not Path(requested_path).is_absolute():
        full_path = base_dir / requested_path
    else:
        full_path = Path(requested_path)
    
    # Resolve to absolute path, following symlinks
    try:
        resolved_path = full_path.resolve()
    except (OSError, RuntimeError) as e:
        raise PathSecurityError(f"Invalid path: {requested_path}") from e
    
    # Check if the resolved path is within the base directory
    try:
        resolved_path.relative_to(base_dir)
    except ValueError:
        raise PathSecurityError(
            f"Path traversal detected: {requested_path} would escape base directory {base_dir}"
        )
    
    return resolved_path

def safe_join_path(base_dir: Union[str, Path], *parts: str) -> Path:
    """
    Safely join path components, preventing directory traversal.
    
    Args:
        base_dir: The base directory
        *parts: Path components to join
        
    Returns:
        A validated Path within base_dir
        
    Raises:
        PathSecurityError: If any component would cause path traversal
    """
    base_dir = Path(base_dir).resolve()
    
    # Filter out any empty parts
    parts = [p for p in parts if p]
    
    # Check for suspicious patterns in any part
    suspicious_patterns = ['..', '~', ':', '*', '?', '"', '<', '>', '|']
    for part in parts:
        # Check each component of the path separately if it contains /
        if '/' in part or '\\' in part:
            sub_parts = part.replace('\\', '/').split('/')
            for sub_part in sub_parts:
                for pattern in suspicious_patterns:
                    if pattern in sub_part:
                        raise PathSecurityError(f"Suspicious path component: {sub_part}")
        else:
            for pattern in suspicious_patterns:
                if pattern in part:
                    raise PathSecurityError(f"Suspicious path component: {part}")
    
    # Join the path safely
    result_path = base_dir
    for part in parts:
        # Split on both forward and back slashes
        sub_parts = part.replace('\\', '/').split('/')
        for sub_part in sub_parts:
            if sub_part:  # Skip empty parts
                result_path = result_path / sub_part
    
    # Validate the final path
    return validate_safe_path(base_dir, result_path)

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename to be safe for filesystem use.
    
    Args:
        filename: The filename to sanitize
        max_length: Maximum length for the filename
        
    Returns:
        A sanitized filename
    """
    # Remove path separators and other dangerous characters
    dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = 'unnamed'
    
    # Truncate if too long
    if len(filename) > max_length:
        # Keep extension if present
        if '.' in filename:
            name, ext = filename.rsplit('.', 1)
            max_name_length = max_length - len(ext) - 1
            filename = name[:max_name_length] + '.' + ext
        else:
            filename = filename[:max_length]
    
    return filename

def create_secure_directory(base_dir: Union[str, Path], subdir: str) -> Path:
    """
    Create a directory within base_dir, ensuring it's secure.
    
    Args:
        base_dir: The base directory
        subdir: Subdirectory to create
        
    Returns:
        Path to the created directory
        
    Raises:
        PathSecurityError: If the path would be insecure
    """
    safe_path = safe_join_path(base_dir, subdir)
    safe_path.mkdir(parents=True, exist_ok=True)
    return safe_path