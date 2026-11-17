# Explainable and Reliable AI for Software Defect Prediction

An AI-powered Software Defect Prediction system that uses Machine Learning to identify potentially defective software modules. The project provides model comparison, performance evaluation, SHAP-based explainability, individual predictions, and an interactive Streamlit dashboard.

## 1. Project Overview

Software defects can increase development costs, reduce software quality, and affect system reliability.

This project uses historical software metrics from the NASA PROMISE CM1 dataset to predict whether a software module is defective or non-defective.

The system also provides explainable AI techniques to help users understand why a machine learning model makes a particular prediction.

### Objectives

- Predict defective and non-defective software modules.
- Compare multiple machine learning algorithms.
- Evaluate model performance using standard metrics.
- Explain model predictions using SHAP.
- Visualize dataset patterns and feature importance.
- Generate a research report.

## 2. Key Features

- **Dataset Analysis:** Explore dataset information, distributions, and feature relationships.
- **Model Comparison:** Compare Logistic Regression, Random Forest, and Gradient Boosting.
- **Model Evaluation:** View confusion matrix, ROC curve, Precision-Recall curve, and evaluation metrics.
- **Explainable AI:** Understand feature contributions using SHAP.
- **Individual Prediction:** Predict whether a software module is defective.
- **Research Report:** Generate a PDF report containing project results.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Plotly, Matplotlib, Seaborn |
| Explainable AI | SHAP |
| Report Generation | ReportLab |
| Dataset | NASA PROMISE CM1 |

## 4. Project Structure

```text
Software-Defect-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── cm1.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── models.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── prediction.py
│   └── report_generator.py
│
├── assets/
│   └── style.css
│
└── outputs/
    ├── figures/
    └── reports/
```

*Note: The structure above describes the expected project organization. Your actual folders and files may vary.*

## 5. System Requirements

Recommended environment:

- Operating System: Windows 10/11, Linux, or macOS
- Python: 3.10–3.12 recommended
- RAM: 4 GB minimum; 8 GB recommended
- Storage: Sufficient space for dependencies, dataset, and generated reports
- Internet: Required for initial dependency installation

Use the versions specified in `requirements.txt` whenever available.

## 6. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rupeshkumar96/Software-Defect-Prediction.git
```

### Step 2: Navigate to Project Directory

```bash
cd Software-Defect-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows CMD:**

```cmd
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 5: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 6: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 7. Dataset Configuration

This project uses the NASA PROMISE CM1 software defect dataset.

**Dataset file:**

```text
data/cm1.csv
```

Ensure that the dataset is available at the expected location before running the application.

### Target Variable

The target variable represents the defect status of a software module.

| Value | Meaning |
|---|---|
| 0 | Non-defective |
| 1 | Defective |

The dataset contains software metrics that are used as input features for machine learning models.

**Important:** The target column must be correctly detected or configured according to the actual column name in your dataset.

## 8. Run the Application

After installing dependencies and configuring the dataset, execute:

```bash
python -m streamlit run app.py
```

The dashboard will open in your default browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

To stop the application, press:

```text
Ctrl + C
```

## 9. Dashboard Modules

### 1. Dataset Analysis
- Dataset overview
- Statistical summaries
- Feature distributions
- Exploratory data analysis

### 2. Model Comparison
- Logistic Regression
- Random Forest
- Gradient Boosting
- Comparison of model performance

### 3. Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve and AUC
- Precision-Recall Curve

### 4. Explainable AI (SHAP)
- Global feature importance
- SHAP summary visualization
- Individual prediction explanations
- Feature contribution analysis

### 5. Individual Prediction
- Enter software metric values
- Predict defect status
- View prediction results

### 6. Research Report
- Generate a PDF report
- Review model evaluation results
- Access project findings

## 10. Troubleshooting

### Issue 1: ModuleNotFoundError

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

### Issue 2: NumPy trapz AttributeError

If you encounter:

```text
AttributeError: module 'numpy' has no attribute 'trapz'
```

Replace:

```python
np.trapz(y, x)
```

with:

```python
np.trapezoid(y, x)
```

### Issue 3: Dataset Not Found

Verify that the dataset exists:

```text
data/cm1.csv
```

Check that the file path in the application matches its actual location.

### Issue 4: Streamlit Not Recognized

Run Streamlit through Python:

```bash
python -m streamlit run app.py
```

### Issue 5: Wrong Python Environment

Check the active Python interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

Make sure dependencies are installed in the same environment used to run Streamlit.

## 11. Future Enhancements

- Integrate additional NASA PROMISE datasets.
- Experiment with advanced machine learning algorithms.
- Improve prediction reliability through hyperparameter tuning.
- Add model monitoring and performance tracking.
- Develop an API for external application integration.
- Improve explainability and reporting capabilities.

## 12. Author

**Rupesh Kumar**

MCA (AI/ML)  
Lovely Professional University

GitHub: [Rupeshkumar96](https://github.com/Rupeshkumar96)

Project Demo: [Software-Defect-Prediction](https://software-defect-prediction-gtgdtzwkgzwrvwag68vlls.streamlit.app/)

---

## License

This project is developed for academic and educational purposes. Add a suitable open-source license if you intend to distribute or reuse the project publicly.
