# Quick Setup Guide

## 🔐 Secure API Key Configuration

For security reasons, we don't store API keys directly in code files. Instead, follow these steps:

### Option 1: Automated Setup (Recommended)
```bash
python setup_config.py
```

This script will:
- ✅ Securely prompt for your API key
- ✅ Create a `.env` file with proper permissions
- ✅ Update `.gitignore` to protect sensitive files
- ✅ Test your configuration

### Option 2: Manual Setup
1. Create a `.env` file in the project root:
```bash
touch .env
```

2. Add your API key to the `.env` file:
```
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
BASE_MODEL=gpt-4o-mini
CRITIC_MODEL=gpt-4o
```

3. Secure the file (Unix/Linux/Mac):
```bash
chmod 600 .env
```

### Option 3: Environment Variable
```bash
export OPENAI_API_KEY="sk-proj-your-actual-api-key-here"
python run_alignment_sprint.py
```

## 🔒 Security Best Practices

1. **Never commit `.env` files** - They're automatically ignored by git
2. **Regenerate API keys** if accidentally exposed
3. **Use restricted permissions** on environment files
4. **Monitor API usage** in your OpenAI dashboard

## ⚡ Quick Start

Once configured, run the full alignment pipeline:
```bash
python run_alignment_sprint.py
```

Or test individual components:
```bash
python src/config.py  # Test configuration
python src/generate_preferences.py  # Generate training data
python src/constitutional_ai.py  # Test safety system
```

## 🆘 Troubleshooting

**"OpenAI API key not found" error:**
- Check that your `.env` file exists and contains `OPENAI_API_KEY=...`
- Verify the API key starts with `sk-`
- Try running `python setup_config.py` again

**Permission errors:**
- Make sure you have write access to the project directory
- On Unix systems, check file permissions with `ls -la .env`

**API connection errors:**
- Verify your API key is valid at https://platform.openai.com/api-keys
- Check your internet connection
- Ensure you have sufficient OpenAI credits 