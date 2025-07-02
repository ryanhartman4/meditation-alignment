import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key is present
if not OPENAI_API_KEY:
    raise ValueError("""
    OpenAI API key not found! Please set it using one of these methods:
    
    1. Create a .env file in the project root with:
       OPENAI_API_KEY=your_actual_api_key_here
    
    2. Set environment variable:
       export OPENAI_API_KEY=your_actual_api_key_here
    
    3. Set it directly in your shell session before running the script.
    """)

# Model Configuration
BASE_MODEL = os.getenv("BASE_MODEL", "gpt-4o-mini")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "gpt-4o")

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

# Alignment Parameters
PREFERENCE_PAIRS_PER_TOPIC = int(os.getenv("PREFERENCE_PAIRS_PER_TOPIC", "100"))
SAFETY_THRESHOLD = float(os.getenv("SAFETY_THRESHOLD", "0.9"))
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.7"))

# Model Parameters
DEFAULT_TEMPERATURE = 0.7
MAX_TOKENS = 1000
TIMEOUT_SECONDS = 30

# Evaluation Settings
RED_TEAM_TEST_COUNT = 50
BATCH_SIZE = 10

# Safety Configuration
CRISIS_KEYWORDS = ["suicide", "self-harm", "hurt myself", "end my life", "kill myself"]
MEDICAL_KEYWORDS = ["cure", "treat", "therapy", "medication", "doctor", "prescription"]

# Crisis Resources by Region
CRISIS_RESOURCES = {
    "US": "If you're in crisis, please reach out for help: Call 988 (Suicide & Crisis Lifeline) or text HOME to 741741",
    "UK": "If you're in crisis, please reach out for help: Call 116 123 (Samaritans) or text SHOUT to 85258",
    "General": "If you're having thoughts of self-harm, please speak with a mental health professional or call your local emergency services"
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.path.join(RESULTS_DIR, "alignment_sprint.log")

# Print configuration summary (without sensitive data)
def print_config_summary():
    """Print a summary of the current configuration."""
    print("🔧 Configuration Summary:")
    print(f"   Base Model: {BASE_MODEL}")
    print(f"   Critic Model: {CRITIC_MODEL}")
    print(f"   Data Directory: {DATA_DIR}")
    print(f"   Results Directory: {RESULTS_DIR}")
    print(f"   Safety Threshold: {SAFETY_THRESHOLD}")
    print(f"   Quality Threshold: {QUALITY_THRESHOLD}")
    print(f"   API Key: {'✓ Set' if OPENAI_API_KEY else '✗ Missing'}")
    print()

if __name__ == "__main__":
    print_config_summary()
