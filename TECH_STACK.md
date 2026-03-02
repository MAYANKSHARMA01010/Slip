# Technology Stack & Libraries

This project uses a carefully selected stack of modern Python libraries to handle everything from data processing to deploying our machine learning models via an interactive web interface.

Here is a breakdown of the core libraries we rely on, why we chose them, and what they do in the context of this project.

---

## Frontend & UI

**[Streamlit](https://streamlit.io/)**

- **What it is:** An open-source Python framework for building custom web apps for machine learning and data science.
- **Why we use it:** It allows us to build a highly polished, interactive web dashboard entirely in Python without needing to manage separate frontend code (like HTML/JS/CSS). It drastically reduces the time to prototype and deploy our models.
- **How it's used:** It powers `app.py`, creating the "Data Overview" and "Predict Churn" tabs, managing user inputs globally, and natively rendering matplotlib charts and UI elements.

---

## Data Handling & Preprocessing

**[Pandas](https://pandas.pydata.org/)**

- **What it is:** A powerful data manipulation and analysis library.
- **Why we use it:** It provides fast, flexible, and expressive data structures (DataFrames) designed to make working with tabular data easy and intuitive.
- **How it's used:** We use pandas to load our CSV dataset, handle missing values (like coercing empty strings to NaNs), map human-readable outputs to binary columns, and pass the data between the Streamlit UI and our models.

**[NumPy](https://numpy.org/)**

- **What it is:** The fundamental package for scientific computing with Python.
- **Why we use it:** It adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- **How it's used:** Often used behind the scenes by pandas and scikit-learn. We also use it for numerical operations and array reshaping when converting isolated user inputs into inference-ready matrices.

---

## Machine Learning & Modeling

**[Scikit-Learn](https://scikit-learn.org/)**

- **What it is:** A robust machine learning library for Python.
- **Why we use it:** It features top-tier documentation and a highly consistent API. It provides a wide array of classification, regression, and clustering algorithms alongside world-class preprocessing capabilities.
- **How it's used:** We use it extensively to build our core `Pipeline`. It handles data scaling (`StandardScaler`), categorical encoding (`OneHotEncoder`), and ultimately trains our primary classifier (`LogisticRegression`). We also heavily rely on its `metrics` module for model evaluation (accuracy, precision, recall, f1, confusion matrix).

**[Imbalanced-Learn (imblearn)](https://imbalanced-learn.org/)**

- **What it is:** A python library relying on scikit-learn that tackles the problem of imbalanced datasets.
- **Why we use it:** Our dataset is skewed (there are far more non-churners than churners). Without correction, a model might "cheat" and simply predict "No Churn" every time to achieve high accuracy.
- **How it's used:** We use its `SMOTE` (Synthetic Minority Over-sampling Technique) implementation to synthetically generate new examples of the minority class (churners) during training, forcing the model to learn the true underlying patterns rather than the class imbalance.

**[XGBoost](https://xgboost.readthedocs.io/)**

- **What it is:** An optimized distributed gradient boosting library.
- **Why we use it:** It is widely recognized as one of the fastest and most highly performant algorithms for tabular data.
- **How it's used:** We trained an `XGBClassifier` alongside our Logistic Regression model as an alternative, highly complex tree-based approach during the evaluation phase.

**[Joblib](https://joblib.readthedocs.io/)**

- **What it is:** A set of tools to provide lightweight pipelining in Python, specialized for fast disk I/O.
- **Why we use it:** Training machine learning models takes time. It's much faster to train the model once, save it to disk, and just load the pre-trained model when the app boots up.
- **How it's used:** We use `joblib.dump` to serialize our trained `Pipeline` and encoders to disk as `.pkl` files, and `joblib.load` within the Streamlit UI to deserialize them for instant inference.

---

## Visualization

**[Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)**

- **What they are:** Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations. Seaborn is a data visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
- **Why we use them:** They provide immense granular control over visual output, allowing us to build highly customized charts that perfectly integrate into our premium dark theme UI.
- **How they are used:** We use matplotlib to generate the core layouts and bar/pie charts inside the Streamlit dashboard app natively. We use Seaborn within our jupyter notebooks for exploratory data analysis (EDA) and to visualize complex output matrices like the final test-set confusion matrix.

---

## Core Project Files

To tie all these technologies together, the project is structured around three main foundational files:

**`churn.ipynb`**

- **Purpose:** The Research and Modeling Environment.
- **Details:** This Jupyter Notebook is where all the initial Exploratory Data Analysis (EDA) happens. It contains the code for cleaning the data, analyzing correlations, handling the dataset imbalance via SMOTE, and training/evaluating the actual Logistic Regression and XGBoost classifiers. It is the "sandbox" where the data science work is verified before being exported for production.

**`app.py`**

- **Purpose:** The Production Application Interface.
- **Details:** This is the main python script that runs the Streamlit web server. It contains the logic for rendering the UI, the custom CSS for the dark theme, and the functions that load the pre-trained `Pipeline` to execute real-time churn predictions based on user inputs.

**`requirements.txt`**

- **Purpose:** The Dependency Manager.
- **Details:** A standardized text file documenting every external library (and its version constraints, if applicable) required to run this project. Running `pip install -r requirements.txt` reads this file and exactly replicates the technology stack environment on any new machine.
