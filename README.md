# Customer Purchase Prediction using Machine Learning

## 📋 Project Overview

This is a comprehensive machine learning project that predicts whether a customer will make a purchase decision based on their demographic and behavioral characteristics. The project demonstrates end-to-end ML workflow implementation, from data analysis to model deployment.

**Project Status:** ✅ Complete and Production-Ready

---

## 🎯 Academic Context & Learning Objectives

This project was developed as part of a machine learning curriculum to demonstrate proficiency in:

- **Data Preprocessing:** Handling real-world customer data with feature scaling and encoding
- **Exploratory Data Analysis:** Identifying patterns and correlations in customer behavior
- **Classification Algorithms:** Implementing and comparing multiple ML algorithms
- **Model Evaluation:** Using appropriate metrics (Accuracy, Precision, Recall, F1-Score)
- **Web Deployment:** Creating interactive applications for ML models
- **Software Engineering:** Version control, documentation, and code organization

---

## 🔍 Problem Statement

**Challenge:** Predict customer purchase behavior based on their profile

**Input Features:**
- Age (18-70 years)
- Gender (Binary)
- Annual Income ($20,000-$150,000)
- Number of Previous Purchases (0-20)
- Product Category (0-4)
- Time Spent on Website (hours)
- Loyalty Program Status (Binary)
- Discounts Availed (0-5)

**Target Variable:**
- Purchase Status (Binary: 0 = No Purchase, 1 = Purchase)

**Dataset:** 1,500 customer records with 9 features

---

## 📊 Methodology

### 1. Data Loading & Exploration
- Loaded dataset using pandas
- Analyzed data shape, info, and descriptive statistics
- Verified data integrity (no missing values)
- Identified purchase rate: 43.2%

### 2. Exploratory Data Analysis
- **Correlation Analysis:** Identified key features influencing purchase decisions
  - Annual Income: Higher income correlates with purchases
  - Number of Purchases: Previous purchase behavior is a strong indicator
  - Time Spent on Website: Engagement level matters
- **Distribution Analysis:** Analyzed target variable class balance
- **Behavioral Patterns:** Customers who purchase tend to have:
  - Lower average age (39.7 years vs 47.8 years)
  - Higher annual income ($92,367 vs $78,074)
  - More previous purchases (11.9 vs 9.3)

### 3. Data Preprocessing
- **Train-Test Split:** 80-20 split with random_state=42 for reproducibility
- **Feature Scaling:** StandardScaler normalization
- **No Categorical Encoding Needed:** Binary features already encoded

### 4. Model Selection & Implementation

**Model 1: Logistic Regression**
- Linear classification model
- Hyperparameters: max_iter=1000, random_state=42
- Use Case: Baseline model for interpretability

**Model 2: Decision Tree Classifier (RECOMMENDED)**
- Tree-based classification
- Hyperparameters: max_depth=10, random_state=42
- Use Case: Better accuracy and feature importance analysis

### 5. Model Evaluation Results

| Metric | Logistic Regression | Decision Tree |
|--------|-------------------|---------------|
| **Training Accuracy** | 81.58% | 99.58% |
| **Testing Accuracy** | **83.67%** | **89.33%** |
| **Precision** | 87.62% | 91.38% |
| **Recall** | 71.88% | 82.81% |
| **F1-Score** | 0.7897 | 0.8689 |

**Conclusion:** Decision Tree outperforms Logistic Regression with 89.33% accuracy.

---

## 📁 Project Structure

```
customer-purchase-prediction-ml/
├── README.md                              # Project documentation
├── notebook.ipynb                          # Complete ML analysis notebook
├── app.py                                 # Streamlit web application
├── requirements.txt                       # Python dependencies
├── models/
│   ├── logistic_regression_model.pkl      # Trained LR model
│   ├── decision_tree_model.pkl            # Trained DT model
│   ├── scaler.pkl                         # StandardScaler for preprocessing
│   └── feature_names.pkl                  # Feature column names
├── data/
│   └── customer_purchase_data.csv         # Original dataset
├── docs/
│   ├── PROJECT_ANALYSIS.md                # Detailed analysis report
│   ├── METHODOLOGY.md                     # Technical methodology
│   └── DEPLOYMENT_GUIDE.md               # How to deploy
└── .gitignore                             # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jaiswal-Sudhanshu/customer-purchase-prediction-ml.git
   cd customer-purchase-prediction-ml
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

**Option 1: Run Jupyter Notebook**
```bash
jupyter notebook notebook.ipynb
```

**Option 2: Run Streamlit Web App**
```bash
streamlit run app.py
```
Access the app at: `http://localhost:8501`

---

## 💻 Technology Stack

| Category | Technologies |
|----------|---------------|
| **Data Processing** | pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Version Control** | Git, GitHub |
| **Environment** | Python 3.8+ |

---

## 📈 Key Insights

1. **Customer Characteristics:** Customers who purchase are typically older professionals with higher income
2. **Purchase Indicators:** 
   - Annual income is the strongest predictor
   - Previous purchase history indicates future purchases
   - Website engagement correlates with purchase intent
3. **Model Performance:** Decision Tree achieves 89.33% accuracy, suitable for production use
4. **Business Implication:** This model can help identify high-potential customers for targeted marketing

---

## 🌐 Web Application Features

The Streamlit app provides:

- **Interactive Dashboard:** Real-time prediction interface
- **Model Comparison:** Side-by-side comparison of both models
- **Probability Estimates:** Confidence scores for predictions
- **Data Visualization:** Charts for better understanding
- **Feature Information:** Explanation of all input features

---

## 📝 How to Use This Project for Learning

1. **Study the Methodology:** Understand the ML workflow step-by-step
2. **Review the Code:** Learn Python and scikit-learn best practices
3. **Experiment:** Modify hyperparameters and observe results
4. **Deploy:** Practice model deployment on Streamlit Cloud
5. **Document:** Use this as a template for your own projects

---

## 🤝 Contributing & Feedback

This is a student project for educational purposes. For suggestions or improvements:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

This project is provided for educational purposes. Feel free to use it as a reference for your learning.

---

## 👤 Author

**Sudhanshu Jaiswal**
- GitHub: [@Jaiswal-Sudhanshu](https://github.com/Jaiswal-Sudhanshu)
- Project Date: January 2026
- Location: Uttar Pradesh, India

---

## 📞 Support & Questions

For questions about this project:
- Check the documentation in `/docs` folder
- Review the Jupyter notebook for detailed analysis
- Check GitHub Issues section

---

## 🎓 Educational Value

This project demonstrates:
- ✅ Professional ML development practices
- ✅ Proper documentation and organization
- ✅ End-to-end ML pipeline implementation
- ✅ Model deployment and web integration
- ✅ Academic integrity and originality

**Last Updated:** January 13, 2026
