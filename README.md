# Meditation AI Alignment Sprint 

I built aligned prompts and fine-tunes for Waves AI using techniques similar to the ones demonstrated here. Forking this repo will allow you to build your own version of a custom evaluation suite for your AI tools!

I also had Claude review it, here's what it had to say: This is a comprehensive demonstration of AI alignment techniques applied to mental health applications, showcasing RL Fine-tuning, a version of constitutional AI, and advanced red-teaming that can be applied to other developers' applications in a single night sprint.

## Overview

This project demonstrates how to build a production-ready AI alignment pipeline for a meditation app, ensuring safety, helpfulness, and reliability when dealing with sensitive mental health topics. It implements some of the same techniques used by frontier AI labs, scaled down for educational purposes. I have tried my best (with Claude's help) to make each part as understandable as possible. 

The prompts are obviously not the ones used in the Waves production environment, but are meant to illustrate the effectiveness of the evaluation tools used.

## Cost Estimation

- **Basic Sprint**: ~$5-10 in OpenAI API costs (15-30 minutes)
- **With RFT Fine-tuning**: Additional ~$25-50 (GPT-4o-mini fine-tuning)
- **Cost Control**: Built-in budget limits with automatic stopping at $10 (configurable) 

## Key Features

- **Synthetic Preference Generation**: Creates realistic good/bad response pairs for training
- **Constitutional AI**: Pattern-based safety system with automatic content rewriting. The idea of constitutional AI is to use a set of principles to make judgements about the output. In production, this looks much more like using AI or LLMs-as-a-judge, but for the sake of simplicity (and to keep costs down), this version uses hard-coded alignment judgements
- **Comprehensive Evaluation**: Multi-stage testing including red-teaming and edge cases
- **Advanced Red-Teaming**: Optional Integration with Promptfoo for professional evaluation
- **o4-mini RFT**: Optional reinforcement fine-tuning for enhanced safety
- **Interactive Dashboard**: Real-time visualization of alignment metrics and improvements

## Quick Start

### Prerequisites

- Python 3.8+ (must use `python3` command if on Mac)
- OpenAI API key
- Optional: Node.js (for Promptfoo)

### Installation

```bash
# Clone the repository
git clone https://github.com/ryanhartman4/meditation-alignment.git
cd meditation-alignment

# Install dependencies
pip3 install -r requirements.txt

# Set up API key
python3 setup_config.py

# Verify configuration
python3 src/config.py  # Displays configuration summary
```

### Run the Alignment Sprint

```bash
# Full sprint (15-30 minutes, ~$5-10 in API costs)
python3 run_alignment_sprint.py

# Quick mode (skip optional stages)
python3 run_alignment_sprint.py --quick

# Skip preference generation (use existing data)
python3 run_alignment_sprint.py --skip-preferences

# Include RFT (expensive, adds ~$25-50)
python3 run_alignment_sprint.py --include-rft
```

### Testing & Development

```bash
# Test individual components
python3 src/generate_preferences.py  # Test preference generation
python3 src/constitutional_ai.py     # Test safety system
python3 src/evaluation.py           # Test evaluation framework

# Linting and type checking
ruff check src/
ruff check --fix src/
mypy src/  # If mypy is installed
```

## Project Structure

```
meditation-alignment/
├── src/                           # Core source code
│   ├── alignment_loop.py          # Main alignment pipeline
│   ├── api_utils.py               # API utilities and helpers
│   ├── config.py                  # Configuration and settings
│   ├── constitutional_ai.py       # Safety system implementation
│   ├── constitutional_validator.js # JavaScript validator for Promptfoo
│   ├── create_dashboard.py        # Interactive results visualization
│   ├── evaluation.py              # Evaluation framework
│   ├── generate_preferences.py    # Synthetic preference data generation
│   ├── inspect_eval.py            # Inspect AI integration
│   ├── path_utils.py              # Path and file utilities
│   ├── prepare_rft_data.py        # RFT data preparation
│   ├── promptfoo_config.yaml      # Main Promptfoo configuration
│   ├── promptfoo_simple_config.yaml # Simplified Promptfoo config
│   ├── promptfoo_test_config.yaml   # Test Promptfoo configuration
│   ├── promptfoo_provider.py      # Promptfoo integration bridge
│   ├── rft_grader.py              # RFT preference grading
│   ├── rft_training.py            # o4-mini fine-tuning pipeline
│   └── run_promptfoo_eval.py      # Promptfoo evaluation runner
├── data/                          # Data files and configurations
│   ├── meditation_constitution.json # Constitutional AI safety rules
│   ├── meditation_test_cases.json   # Red team test scenarios
│   ├── preference_stats.json        # Preference generation statistics
│   └── preferences_synthetic.jsonl  # Generated preference pairs
├── prompts/                       # Prompt templates
│   └── meditation_prompt.txt      # Base meditation guidance prompt
├── results/                       # Evaluation outputs and dashboards
│   ├── latest/                    # Latest run results
│   │   ├── model_comparison.json  # Model performance comparison
│   │   ├── promptfoo_summary.json # Promptfoo evaluation summary
│   │   └── red_team_results.json  # Red team test results
│   ├── run_YYYYMMDD_HHMMSS/      # Timestamped evaluation runs
│   ├── alignment_dashboard.html   # Interactive results dashboard
│   └── alignment_summary.json     # Overall alignment metrics
├── blog/                          # Blog posts and articles
│   └── alignment_blog_post.md     # Technical blog post about the project
├── run_alignment_sprint.py        # Main runner script
├── setup_config.py               # Configuration setup utility
├── requirements.txt               # Python dependencies (full)
├── requirements-core.txt          # Core dependencies only
├── SETUP.md                       # Setup and configuration guide
├── Project-outline.md             # Project structure outline
└── README.md                      # This file
```

### Version Control Notes

**Tracked Files** (in git):
- All source code (`src/`)
- Configuration examples (`data/meditation_constitution.json`, `data/meditation_test_cases.json`)
- Documentation (`README.md`, `SETUP.md`, `INTEGRATION_GUIDE.md`)
- Requirements files

**Generated/Local Files** (not tracked):
- Environment files (`.env`, `.env.local`)
- API keys and credentials
- Results directories (`results/`)
- Generated preference data (`data/preferences_synthetic.jsonl`)
- Python cache files (`__pycache__/`, `.pyc`)
- Virtual environments

## Alignment Pipeline

The sprint orchestrates these stages through `run_alignment_sprint.py`:

### Stage 1: Core Alignment Evaluation (Required)
Runs the main alignment pipeline (`alignment_loop.py`) with three sub-stages:

**1. Model Comparison:**
   - **Base Model**: Uses production prompt without safety enhancements
   - **Aligned Model**: Production prompt + constitutional AI safety layer
   - Tests both models on identical scenarios from `meditation_test_cases.json`
   - Applies constitutional AI to aligned model responses:
     - Pattern-based safety detection using `meditation_constitution.json`
     - Hierarchical severity levels (critical/high/medium/low)
     - Automatic content rewriting while preserving helpfulness
     - Crisis resource injection (988 lifeline)

**2. Red Team Evaluation:**
   - Runs comprehensive red team test suite on aligned model
   - Tests multiple critical safety scenarios
   - Evaluates crisis handling, medical safety, and boundary violations
   - Parallel execution for faster results

**3. Edge Case Testing:**
   - Tests dangerous edge cases (e.g., "meditate while driving")
   - Ensures appropriate safety redirects
   - Validates crisis response handling

### Stage 2: Dashboard Creation (Required)
Generates interactive HTML dashboard:
- Model performance comparison charts
- Safety improvement metrics
- Red team test results
- Violation breakdown by category
- Overall production readiness assessment

### Stage 3: Advanced Red-Teaming (Optional)
Interactive prompt for Promptfoo evaluation:
- YAML-based test configuration
- Custom safety validators in JavaScript
- Extended test scenarios beyond core red team
- Results integrated into dashboard

To run manually:
```bash
cd src
npx promptfoo@latest eval --no-cache
```

### Stage 4: o4-mini RFT (Optional)
When using `--include-rft` flag:

1. **Preference Data Check**: 
   - Checks if `data/preferences_synthetic.jsonl` exists
   - If missing, prompts for preference generation:
     - Warns about synthetic unsafe content creation
     - Generates preference pairs with safety violations
     - Topics: medical claims, crisis handling, physical assumptions, boundaries
     - Uses `generate_preferences.py` to create training data

2. **Interactive Consent**: 
   - Warns about costs (~$25-50)
   - Requires explicit user confirmation

3. **RFT Pipeline** (if preferences exist):
   - Grades preferences using GPT-4o (`rft_grader.py`)
   - Prepares OpenAI fine-tuning format (`prepare_rft_data.py`)
   - Manages fine-tuning job (`rft_training.py`)
   - Evaluates base vs aligned vs fine-tuned models

### Final Output
Sprint completes with:
- Safety improvement percentage
- Red team pass rate
- Production readiness verdict 
- Total runtime and cost summary
- Paths to dashboard and detailed results

## Results

The alignment pipeline typically achieves:
- Improvement in crisis response accuracy
- Reduction in harmful medical advice
- Approximately maintained quality score
- High pass rates on critical safety tests

View detailed results in the interactive dashboard after running the sprint.

## Safety Principles

The system enforces six core safety principles:

1. **Medical Safety**: Never provide medical/therapeutic advice
2. **Crisis Safety**: Always redirect to professional help
3. **Inclusivity**: Accessible language for all abilities
4. **Boundaries**: Maintain appropriate therapeutic distance
5. **Age Appropriateness**: Content suitable for all ages
6. **Trigger Warnings**: Identify potentially triggering content

## Advanced Features

### Custom Test Cases

Add your own red team tests in `data/meditation_test_cases.json`:

```json
{
  "id": "custom_001",
  "prompt": "Your test prompt here",
  "category": "test_category",
  "must_include": ["required", "phrases"],
  "must_not_include": ["forbidden", "phrases"],
  "severity": "critical"
}
```

### Constitutional Rules

Modify safety rules in `data/meditation_constitution.json`:

```json
{
  "rule": "Your safety rule description",
  "patterns": ["\\bpattern1\\b", "\\bpattern2\\b"],
  "exception_patterns": ["\\bexception1\\b"],
  "severity": 0.8,
  "examples": {
    "violations": ["Example violation text"],
    "acceptable": ["Example acceptable text"]
  }
}
```

## Monitoring & Observability

The system generates comprehensive logs and metrics:
- Safety score tracking over time
- Violation categorization
- Response quality metrics
- Red team test results
- Fine-tuning progress (if enabled)

## Production Deployment

Before deploying to production:

1. Ensure near-perfect pass rate on critical tests
2. Review all red team failures
3. Test with real user scenarios
4. Implement gradual rollout
5. Set up continuous monitoring

## Contributing

Contributions are welcome! Areas for improvement:
- Additional red team test cases
- More sophisticated constitutional rules
- Integration with other evaluation frameworks
- Performance optimizations
- Multi-language support

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{meditation-alignment-2025,
  title={One-Night Meditation AI Alignment Sprint},
  author={Ryan Hartman},
  year={2025},
  url={https://github.com/ryanhartman4/meditation-alignment}
}
```

## Acknowledgments

- OpenAI for GPT-4o, o4-mini, and fine-tuning APIs
- Anthropic for constitutional AI concepts, RFT Alignment Ideas, and claude code
- Promptfoo team for evaluation framework
- UK AISI for Inspect AI
- The AI safety community for red-teaming best practices

---

Built for safer AI in mental health applications