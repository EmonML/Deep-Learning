# 🚢 Titanic Survival Prediction Using ANN

## 📌 Overview

এই project-এ **Titanic dataset** ব্যবহার করে একটি **Artificial Neural Network (ANN)** মডেল তৈরি করা হয়েছে, যার মাধ্যমে একজন passenger-এর **বেঁচে থাকা (Survived) অথবা মারা যাওয়া (Not Survived)** prediction করা হয়।

এই notebook-এর বিশেষ বৈশিষ্ট্য হলো প্রতিটি গুরুত্বপূর্ণ step বাংলায় ব্যাখ্যা করা হয়েছে, যাতে একজন beginner থেকে intermediate-level learner সহজে বুঝতে পারে কীভাবে একটি tabular dataset-এর উপর ANN classification model তৈরি করা যায়।

এই project-এ শুধুমাত্র ANN model তৈরি করা হয়নি; বরং একটি complete machine learning/deep learning workflow অনুসরণ করা হয়েছে:

```text
Titanic Dataset
      ↓
Data Inspection
      ↓
Missing Value Handling
      ↓
Feature Selection
      ↓
Categorical Encoding
      ↓
Feature Scaling
      ↓
Train / Validation / Test Split
      ↓
ANN Model
      ↓
Training
      ↓
Model Evaluation
      ↓
Confusion Matrix
      ↓
ROC Curve & AUC
      ↓
Passenger Survival Prediction
```

---

## 🎯 Project Objective

এই project-এর মূল উদ্দেশ্য হলো Titanic passenger-এর বিভিন্ন বৈশিষ্ট্য ব্যবহার করে তাদের survival prediction করা।

Model-এর target:

```text
0 → Not Survived
1 → Survived
```

---

## 📊 Dataset

এই project-এ **Titanic dataset** ব্যবহার করা হয়েছে।

Dataset-এর গুরুত্বপূর্ণ feature:

| Feature    | Description                       |
| ---------- | --------------------------------- |
| `pclass`   | Passenger class                   |
| `sex`      | Passenger gender                  |
| `age`      | Passenger age                     |
| `sibsp`    | Number of siblings/spouses aboard |
| `parch`    | Number of parents/children aboard |
| `fare`     | Passenger fare                    |
| `embarked` | Port of embarkation               |
| `survived` | Target variable                   |

---

## 🧹 Data Preprocessing

ANN model-এ দেওয়ার আগে dataset preprocessing করা হয়েছে।

### Numerical Features

Numerical features:

```text
pclass
age
sibsp
parch
fare
```

Missing numerical values-এর জন্য **Median Imputation** ব্যবহার করা হয়েছে।

এরপর numerical features-এর উপর:

```python
StandardScaler()
```

ব্যবহার করা হয়েছে।

---

## 🔤 Categorical Encoding

Categorical features:

```text
sex
embarked
```

এসবকে numerical representation-এ পরিবর্তন করার জন্য:

```python
OneHotEncoder()
```

ব্যবহার করা হয়েছে।

---

## ⚖️ Feature Scaling

ANN-এর training আরও stable করার জন্য numerical features-কে StandardScaler-এর মাধ্যমে scale করা হয়েছে।

Standardization:

$$
z = \frac{x-\mu}{\sigma}
$$

এখানে:

* `x` = original value
* `μ` = mean
* `σ` = standard deviation

---

## 🧠 ANN Architecture

এই project-এ একটি Feed-Forward Artificial Neural Network ব্যবহার করা হয়েছে।

Architecture:

```text
Input Layer
     ↓
Dense(64, ReLU)
     ↓
Batch Normalization
     ↓
Dropout(0.30)
     ↓
Dense(32, ReLU)
     ↓
Batch Normalization
     ↓
Dropout(0.20)
     ↓
Dense(16, ReLU)
     ↓
Output Dense(1, Sigmoid)
```

### Output Layer

যেহেতু এটি binary classification problem:

```python
Dense(1, activation="sigmoid")
```

ব্যবহার করা হয়েছে।

Output:

```text
0 → Not Survived
1 → Survived
```

---

## ⚙️ Model Configuration

ANN model-এর জন্য:

```text
Optimizer      → Adam
Learning Rate  → 0.001
Loss Function  → Binary Crossentropy
Batch Size     → 32
Maximum Epochs → 100
```

ব্যবহার করা হয়েছে।

---

## 🛑 Callbacks

Overfitting কমানো এবং best model সংরক্ষণের জন্য:

### EarlyStopping

Validation loss উন্নতি না করলে training বন্ধ করে।

```python
EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
```

### ModelCheckpoint

সবচেয়ে ভালো validation accuracy-এর model সংরক্ষণ করে:

```text
best_titanic_ann.keras
```

---

## 📈 Model Evaluation

Model-এর performance বিভিন্ন metric দিয়ে evaluate করা হয়েছে।

### Accuracy

সঠিক prediction-এর অনুপাত।

### Precision

যেসব passenger-কে survived বলা হয়েছে, তাদের মধ্যে কতজন সত্যিই survived।

### Recall

যারা সত্যিই survived, তাদের মধ্যে কতজনকে model সঠিকভাবে শনাক্ত করেছে।

### F1-Score

Precision এবং Recall-এর harmonic mean।

### ROC-AUC

Different classification threshold-এ model-এর discrimination ability পরিমাপ করা হয়েছে।

---

## 📊 Visualizations

Project-এ নিম্নলিখিত visualization রয়েছে:

* Training vs Validation Accuracy
* Training vs Validation Loss
* Confusion Matrix
* ROC Curve
* Performance Metrics Comparison

---

## 🔲 Confusion Matrix

Confusion Matrix-এর মাধ্যমে দেখা যায়:

```text
True Negative
False Positive
False Negative
True Positive
```

এটি model কোন ধরনের ভুল prediction করছে তা বুঝতে সাহায্য করে।

---

## 📉 ROC Curve & AUC

ROC curve-এ:

```text
X-axis → False Positive Rate
Y-axis → True Positive Rate
```

এবং AUC score model-এর classification performance সম্পর্কে ধারণা দেয়।

---

## 💾 Saved Model

Training শেষে best model সংরক্ষণ করা হয়:

```text
best_titanic_ann.keras
```

পরবর্তীতে এই model load করে নতুন passenger-এর prediction করা যায়।

---

## 🔮 New Passenger Prediction

Project-এ একটি নতুন passenger-এর তথ্য দিয়ে survival probability বের করার উদাহরণও রয়েছে।

Example:

```python
new_passenger = {
    "pclass": 1,
    "sex": "female",
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 80.0,
    "embarked": "S"
}
```

Model একটি survival probability প্রদান করবে:

```text
Survival Probability: XX%
Prediction: Survived / Not Survived
```

---

## 📁 Project Structure

Recommended GitHub structure:

```text
Titanic-ANN/
│
├── Titanic_ANN_Bangla_Explained.ipynb
├── best_titanic_ann.keras
├── README.md
├── requirements.txt
│
└── images/
    ├── accuracy.png
    ├── loss.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## 🛠️ Technologies & Libraries

এই project-এ ব্যবহৃত প্রধান technologies:

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📦 Installation

প্রথমে repository clone করুন:

```bash
git clone https://github.com/your-username/Titanic-ANN.git
cd Titanic-ANN
```

তারপর প্রয়োজনীয় package install করুন:

```bash
pip install -r requirements.txt
```

Jupyter Notebook চালানোর জন্য:

```bash
jupyter notebook
```

এরপর:

```text
Titanic_ANN_Bangla_Explained.ipynb
```

open করুন।

---

## 📋 Requirements

`requirements.txt`-এ আনুমানিকভাবে:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
jupyter
```

---

## 🚀 How to Run

### Step 1

Repository clone করুন।

### Step 2

Dependencies install করুন:

```bash
pip install -r requirements.txt
```

### Step 3

Notebook open করুন:

```bash
jupyter notebook
```

### Step 4

`Titanic_ANN_Bangla_Explained.ipynb` open করুন।

### Step 5

প্রতিটি cell sequentially run করুন।

---

## 📚 Learning Outcomes

এই project সম্পন্ন করার পর আপনি বুঝতে পারবেন:

* Tabular dataset কীভাবে inspect করতে হয়
* Missing values কীভাবে handle করতে হয়
* Categorical data কীভাবে encode করতে হয়
* ANN-এর আগে feature scaling কেন করা হয়
* Train/Validation/Test split কী
* Fully Connected Neural Network কীভাবে তৈরি করতে হয়
* Dense layer কীভাবে কাজ করে
* ReLU ও Sigmoid activation কী
* Dropout কেন ব্যবহার করা হয়
* Batch Normalization কী
* Binary Crossentropy কী
* Adam optimizer কী
* EarlyStopping কীভাবে কাজ করে
* ModelCheckpoint কী
* Classification metrics কীভাবে evaluate করতে হয়
* Confusion Matrix কীভাবে interpret করতে হয়
* ROC Curve এবং AUC কী
* Saved ANN model দিয়ে নতুন data prediction কীভাবে করতে হয়

---

## ⚠️ Important Note

এই project মূলত **educational purpose**-এর জন্য তৈরি করা হয়েছে।

Model-এর performance dataset split, preprocessing, random seed, architecture, hyperparameters এবং training environment-এর উপর নির্ভর করতে পারে। তাই notebook পুনরায় run করলে ফলাফলে সামান্য পরিবর্তন হতে পারে।

---

## 👨‍💻 Author

**Md Emon Islam**

AI Enthusiast

---

## ⭐ If You Find This Project Useful

Repository-টি useful মনে হলে GitHub repository-তে ⭐ **Star** দিতে পারেন এবং project-টি অন্যদের সাথে share করতে পারেন।
