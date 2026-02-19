# 🚀 Loan Approval Prediction using Artificial Neural Network (ANN)

A Machine Learning web application that predicts loan approval decisions using an Artificial Neural Network (ANN).  
Built with TensorFlow/Keras and deployed using Streamlit.

---

## 🎯 Project Overview

This project uses a feedforward neural network (ANN) to predict whether a loan application will be **approved or rejected** based on customer features such as:

- Age  
- Income  
- Credit Score  
- Loan Amount  
- Employment Status  
- Marital Status  
- Existing Loans  

---

## 📋 Features

- ✅ Interactive Streamlit Web App for real-time predictions  
- ✅ ANN Model with 2 hidden layers (64 → 32 neurons)  
- ✅ Automatic Data Preprocessing (Encoding + Scaling)  
- ✅ Displays Approval Probability with Confidence Level  
- ✅ End-to-End ML Pipeline (Training → Saving → Deployment)

---

## 🏗️ Model Architecture

- **Input Layer**: 9 features  
- **Hidden Layer 1**: 64 neurons (ReLU activation)  
- **Hidden Layer 2**: 32 neurons (ReLU activation)  
- **Output Layer**: 1 neuron (Sigmoid activation for binary classification)  
- **Total Parameters**: ~2,753 trainable parameters  

---

## 📁 Project Structure

```
ANN_project/
│
├── app.py                         # Streamlit web application
├── training.ipynb                 # Model training notebook
├── model.h5                       # Trained ANN model
├── label_encoder_married.pkl      # Label encoder for 'Married'
├── one_hot_encoded_EmpStatus.pkl  # One-hot encoder for 'EmploymentStatus'
├── Scaler.pkl                     # StandardScaler
├── test_data.csv                  # Dataset (5000+ samples)
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or 3.12  
  (TensorFlow does NOT support Python 3.14+)
- pip package manager  

---

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd ANN_project
```

---

### Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv
```

Activate environment:

**Mac/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Run the Application

```bash
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## 📊 Dataset

The model is trained on a dataset containing:

**Features**
- Age
- Income
- CreditScore
- LoanAmount
- EmploymentStatus
- Married
- ExistingLoans

**Target**
- LoanApproved (0 = Rejected, 1 = Approved)

**Total Samples**
- 5000+ loan applications

---

## 📈 Feature Ranges

- Age: 21 – 59 years  
- Income: 20,000 – 149,999  
- Credit Score: 300 – 849  
- Loan Amount: 50,000 – 499,999  
- Existing Loans: 0 – 3  
- Married: Yes / No  
- Employment Status: Salaried / SelfEmployed / Unemployed  

---

## 🔧 Model Training

To train the model yourself:

1. Open `training.ipynb`
2. Run all cells sequentially

The notebook includes:

- Data loading
- Encoding (LabelEncoder & OneHotEncoder)
- Feature scaling (StandardScaler)
- ANN architecture definition
- Training with EarlyStopping
- Model & encoder saving

---

### ⚙️ Training Configuration

- Train/Test Split: 80/20  
- Optimizer: Adam  
- Loss Function: Binary Crossentropy  
- Metric: Accuracy  
- EarlyStopping: patience=10 (monitors validation loss)

---

## 🎮 How to Use

1. Run the Streamlit app  
2. Enter customer details  
3. Click **Predict Loan Approval**  
4. View:
   - Approval / Rejection status  
   - Probability percentage  
   - Confidence indicator  

---

## 🛠️ Technologies Used

- TensorFlow / Keras  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  

---

## 📦 Dependencies

```
ipykernel
numpy
pandas
streamlit>=1.54.0
scikit-learn
tensorflow
```

---

## 🌐 Deployment (Streamlit Cloud)

1. Push code to GitHub  
2. Go to Streamlit Cloud  
3. Connect your repository  
4. Set main file path to `app.py`  
5. Deploy  

---

## 🔍 Model Performance

- Type: Binary Classification  
- Architecture: Feedforward Neural Network (ANN)  
- Activation: ReLU (Hidden), Sigmoid (Output)  
- EarlyStopping used to prevent overfitting  

---
 Scikit-learn Preprocessing Guides  
