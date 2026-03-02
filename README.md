# Customer Churn Prediction & Agentic Retention Strategy

## From Predictive Analytics to Intelligent Intervention

### Project Overview

Customer churn is one of the biggest challenges telecom companies face today. Losing a customer isn't just about one cancelled subscription; it snowballs into lost revenue, higher acquisition costs, and a weakened brand. This project tackles that problem head on by building an AI powered system that not only predicts which customers are likely to leave, but eventually evolves into an intelligent agent that can suggest personalized retention strategies.

We worked with the **Telco Customer Churn** dataset, which captures real world customer behavior like how long they've been with the company, what services they use, how much they pay, and whether or not they ended up churning. The goal was to dig into this data, find meaningful patterns, and train a model that can flag at risk customers before it's too late.

**Milestone 1** focuses on classical machine learning. We used techniques like Logistic Regression along with thoughtful feature engineering to build a churn prediction pipeline. The results are served through a clean, interactive Streamlit dashboard where users can explore the data visually and even test predictions for individual customers.

**Milestone 2** takes things further by introducing an agent based AI layer. The idea here is to move beyond just predicting churn and actually reason about it. Using frameworks like LangGraph and retrieval augmented generation (RAG), the system will pull in retention best practices and generate structured intervention plans tailored to each customer's situation.

---

### Constraints & Requirements

This project was built within a set of guidelines to keep things fair, consistent, and accessible for everyone on the team.

We worked as a team of **3 to 4 students**, collaborating across data exploration, model building, UI development, and documentation. Everything was built using **free tier tools only**, meaning we stuck to open source models and APIs that don't require paid subscriptions. No hidden costs, no premium keys.

For the agent based components in Milestone 2, **LangGraph** is the recommended framework. It gives us the flexibility to define agent workflows with clean state management and tool integration.

One important rule: the final version of the project **must be hosted and publicly accessible**. A localhost only demo won't be accepted for the end semester submission. We're expected to deploy using platforms like **Hugging Face Spaces**, **Streamlit Cloud**, or **Render** so that anyone with the link can try it out.

---

### Technology Stack

Here's a quick look at the tools and technologies powering each part of the project.

| Component | Technology |
| :--- | :--- |
| **ML Models** | Logistic Regression, StandardScaler, Scikit Learn |
| **UI Framework** | Streamlit |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |

We chose **Streamlit** for the frontend because it lets us build interactive dashboards with minimal boilerplate, perfect for quickly turning a machine learning pipeline into something visual and usable. For the ML side, **Scikit Learn** handles everything from preprocessing to model training and evaluation. **Matplotlib** and **Seaborn** power the charts and visualizations across the dashboard, while **Joblib** takes care of saving and loading the trained model artifacts.

---

### Setup and Installation

Ready to run this project yourself? Check out our detailed **[Setup and Installation Guide](SETUP_GUIDE.md)** for step-by-step instructions on how to clone the repository, install dependencies, and launch the Streamlit dashboard.