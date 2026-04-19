# Setup & Installation Guide

This guide will help you get the **Slip Intelligence Platform** up and running on your local machine in just a few minutes.

## 🚀 Quick Start (macOS/Linux)

If you're already familiar with Python environments, run these commands to launch the dashboard:

```bash
# 1. Prepare your environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Synchronize libraries
pip install -r requirements.txt

# 3. Launch Slip
streamlit run app.py
```

---

## Detailed Walkthrough

### Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.9+** (Check with `python3 --version`)
- **Git** (Check with `git --version`)
- **API Keys**: To use the "AI Strategist" features, you'll need a **Google Gemini**, **Groq**, or **Mistral** API key in your `.env` file.

### Step 1: Clone the Project
Open your terminal and clone the repository to your local workspace:

```bash
git clone https://github.com/Amansoni045/GEN_AI_MID_TERM.git
cd Slip
```

### Step 2: Initialize the Virtual Environment
We recommend using a virtual environment to keep your global Python installation clean.

```bash
# Create the environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
Slip relies on several modern libraries for ML and Agentic logic. Install them all at once:

```bash
pip install -r requirements.txt
```

### Step 4: Launch the Platform
You're ready! Start the Streamlit server with one command:

```bash
streamlit run app.py
```

The platform will automatically detect if your model artifacts need training, build them if necessary, and then open the dashboard in your default browser (usually at `http://localhost:8501`).

---

## Exploring the Platform
- **Phase 1: Overview**: Dive into the high-level metrics and churn trends of the Telco base.
- **Phase 2: Prediction**: Input a specific customer profile to see an instant, data-driven risk verdict.
- **Phase 3: AI Strategist**: Let the AI agent reason through the risk and generate a professional retention plan.
- **Phase 4: Performance**: Inspect the statistical health and configuration of the underlying ML models.
