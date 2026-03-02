## Video Script: Model Training & App Demo

**(1. Start on the Model Training section of the notebook)**
"Hi everyone. After the data was cleaned and balanced using SMOTE, we moved to Model Training.
We evaluated three algorithms: **Decision Tree**, **Random Forest**, and **XGBoost**.
We used 5-fold cross-validation to test them safely.
**Random Forest** gave us the best and most stable results, so we picked it as our final model."

**(2. Scroll down to Model Evaluation cell)**
"Here is the final evaluation on our 20% test data.
The Random Forest model achieved an **Accuracy of 78.2%**.
More importantly, our **Confusion Matrix** shows it successfully caught 216 customers who were actually going to churn. Since telecom data is based on unpredictable human behavior, 78% is a very realistic and strong score without overfitting."

**(3. Switch over to the live Streamlit App: Data Overview Tab)**
"Now, let's look at the deployment. We built an interactive Streamlit dashboard.
On the **Data Overview** tab, you can see our key metrics: over 7,000 customers with a 26% churn rate.
We also added interactive charts to explore Contract Types, Tenure, and Monthly Charges so stakeholders can visualize the data easily."

**(4. Switch to the Predict Churn Tab and fill out the form)**
"The core feature is the **Predict Churn** tab.
Here, a user or agent can input a customer's specific details. 
*(Click Predict Churn button here)*
Behind the scenes, the app instantly scales the data, encodes it, and feeds it into our Random Forest pipeline.
For this customer, it predicts **[read whatever the screen says, e.g. 'Likely to Stay']**."

**(5. Point at the bottom sections of the App)**
"It doesn't just give a prediction—it gives actionable insights. 
It highlights the **Key Risk Factors** for that specific customer, gives **Recommended Actions** to retain them, and compares them against the average customer.
Finally, we can easily **export the prediction as a CSV** for batch processing.
That completes the end-to-end pipeline for Milestone 1."
