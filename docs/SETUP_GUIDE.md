# How to Run the Telco Churn Dashboard

This guide will walk you through cloning the repository, setting up your environment, and running the Streamlit dashboard locally.

## 🚀 Quick Start (macOS/Linux)

Copy and paste these commands in order to get the project running in seconds:

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the environment
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

---

## Detailed Installation Steps

## Prerequisites

- **Python 3.9+** installed on your system.
- **Git** installed on your system.

## Step 1: Clone the Repository

Open your terminal and run the following command to clone the project:

```bash
git clone https://github.com/MAYANKSHARMA01010/Slip
```

Navigate into the project directory:

```bash
cd GEN_AI_MID_TERM
```

## Step 2: Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

Create a virtual environment named `.venv`:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

- **On macOS/Linux:**

  ```bash
  source .venv/bin/activate
  ```

- **On Windows:**

  ```bash
  .venv\Scripts\activate
  ```

## Step 3: Install Dependencies

With your virtual environment active, install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

*Note: Depending on your system and Python setup, you might need to use `pip3` instead of `pip`.*

## Step 4: Run the Application

Now you are ready to launch the Streamlit dashboard! Run the following command:

```bash
streamlit run app.py
```

The application will automatically train the model (if no saved model artifacts are found) and then launch the dashboard in your default web browser at `http://localhost:8501`.

## What's Next?

- Explore the **Data Overview** tab to see visual distributions of the dataset.
- Navigate to the **Predict Churn** tab to enter custom user data and see real-time churn predictions with actionable insights.
