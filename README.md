# 🚗 BMW Car Price Prediction

A machine learning project that predicts the **selling price of BMW cars** based on vehicle characteristics such as model, year, mileage, transmission, fuel type, tax, MPG, and engine size.

This project demonstrates an end-to-end **regression machine learning workflow** using Python and Scikit-learn, including data exploration, preprocessing, feature encoding, feature scaling, model training, evaluation, and cross-validation.

---

## 📌 Project Overview

Car prices depend on multiple factors such as the vehicle's age, mileage, engine size, fuel type, transmission, and fuel efficiency.

The goal of this project is to build a **Linear Regression model** that learns relationships between these features and the car's selling price.

### Workflow

```text
BMW Dataset
     ↓
Data Loading
     ↓
Data Exploration
     ↓
Data Cleaning
     ↓
Feature / Target Separation
     ↓
Numerical + Categorical Feature Identification
     ↓
Feature Scaling + One-Hot Encoding
     ↓
Train/Test Split
     ↓
Linear Regression
     ↓
Model Prediction
     ↓
Evaluation
     ↓
5-Fold Cross-Validation
```

---

## 🎯 Objectives

* Explore the BMW car dataset
* Understand the relationships between vehicle features and price
* Remove duplicate records
* Separate features and target variable
* Handle numerical and categorical features appropriately
* Apply **StandardScaler** to numerical features
* Apply **OneHotEncoder** to categorical features
* Build a machine learning preprocessing pipeline
* Train a **Linear Regression** model
* Evaluate the model using multiple regression metrics
* Validate model performance using **5-Fold Cross-Validation**

---

## 📊 Dataset

The project uses a BMW car dataset stored in:

```text
bmw.csv
```

### Dataset Features

| Feature        | Description                     | Type        |
| -------------- | ------------------------------- | ----------- |
| `model`        | BMW vehicle model               | Categorical |
| `year`         | Manufacturing year              | Numerical   |
| `price`        | Selling price / target variable | Numerical   |
| `transmission` | Type of transmission            | Categorical |
| `mileage`      | Distance travelled              | Numerical   |
| `fuelType`     | Type of fuel                    | Categorical |
| `tax`          | Vehicle tax                     | Numerical   |
| `mpg`          | Miles per gallon                | Numerical   |
| `engineSize`   | Engine size in litres           | Numerical   |

### Target Variable

```text
price
```

The model learns to predict the BMW's selling price from the remaining features.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation and analysis
* **NumPy** — Numerical operations
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualization
* **Scikit-learn** — Machine learning and preprocessing
* **Google Colab / Jupyter Notebook** — Development environment

---

## 🤖 Machine Learning Model

### Linear Regression

The project uses **Linear Regression** as the primary prediction algorithm.

Linear Regression attempts to model the relationship between the input features and the target price.

The general form is:

```text
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:

* `y` = predicted BMW price
* `x` = input features
* `β₀` = intercept
* `β₁ ... βₙ` = learned coefficients

---

## 🔄 Data Preprocessing

The project uses a Scikit-learn `Pipeline` and `ColumnTransformer` to organize preprocessing and model training.

### 1. Duplicate Removal

Duplicate rows are removed from the dataset:

```python
df = df.drop_duplicates(keep='first')
```

This helps prevent repeated records from unnecessarily influencing the model.

---

### 2. Feature and Target Separation

The target column is:

```python
y = df['price']
```

All remaining columns are used as input features:

```python
X = df.drop('price', axis=1)
```

---

### 3. Numerical Features

The numerical features are automatically identified using:

```python
num_x = X.select_dtypes(include=['int64', 'float64'])
```

Numerical features are standardized using:

```python
StandardScaler()
```

This places numerical features on comparable scales.

---

### 4. Categorical Features

Categorical features are identified using:

```python
cat_x = X.select_dtypes(include='object')
```

They are converted into numerical representations using:

```python
OneHotEncoder(handle_unknown='ignore')
```

Using `handle_unknown='ignore'` also makes the preprocessing pipeline safer when an unseen category appears during prediction.

---

## 🧩 Preprocessing Pipeline

The preprocessing is implemented using `ColumnTransformer`:

```python
preprocessor = ColumnTransformer([
    ('num', num_x_enc, num_x.columns),
    ('cat', cat_x_enc, cat_x.columns)
])
```

The complete machine learning pipeline is then created as:

```python
model = Pipeline([
    ('preprocessor', preprocessor),
    ('lr', LinearRegression())
])
```

This ensures that preprocessing and model training happen consistently.

---

## 📈 Train-Test Split

The dataset is divided into training and testing data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=df['fuelType']
)
```

### Split

```text
80% → Training Data
20% → Testing Data
```

`random_state=42` ensures reproducibility.

The split is stratified using `fuelType` to help preserve the fuel-type distribution between the training and testing sets.

---

## 🧠 Model Training

The Linear Regression model is trained using:

```python
model.fit(X_train, y_train)
```

Predictions are then generated using:

```python
y_pred = model.predict(X_test)
```

---

## 📏 Model Evaluation

The project evaluates the model using four important regression metrics.

### Mean Absolute Error — MAE

```python
mean_absolute_error(y_test, y_pred)
```

MAE measures the average absolute difference between actual and predicted prices.

Lower MAE is better.

---

### Mean Squared Error — MSE

```python
mean_squared_error(y_test, y_pred)
```

MSE calculates the average squared difference between actual and predicted values.

Lower MSE is better.

---

### Root Mean Squared Error — RMSE

```python
rmse = np.sqrt(mse)
```

RMSE is the square root of MSE and expresses prediction error in the same unit as the target variable.

Lower RMSE is better.

---

### R² Score

```python
r2_score(y_test, y_pred)
```

R² measures how much of the variation in BMW prices is explained by the model.

A higher R² generally indicates better explanatory performance.

---

## 🔁 Cross-Validation

To evaluate the model more robustly, the project also uses **5-Fold Cross-Validation**.

```python
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The model is evaluated using:

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring='r2'
)
```

The individual fold scores and their mean are then displayed:

```python
print("Fold R² scores:", scores)
print("Mean R²:", scores.mean())
```

### Why Cross-Validation?

Instead of evaluating the model on only one train/test split, cross-validation repeatedly trains and evaluates the model on different portions of the dataset.

```text
Dataset
   │
   ├── Fold 1 → Validation
   ├── Fold 2 → Validation
   ├── Fold 3 → Validation
   ├── Fold 4 → Validation
   └── Fold 5 → Validation
```

This provides a better understanding of how consistently the model performs.

---

## 📊 Exploratory Data Analysis

The notebook performs exploratory analysis before model training.

Examples include:

### Fuel Type Distribution

```python
df['fuelType'].value_counts()
```

### Dataset Shape

```python
df.shape
```

### Dataset Information

```python
df.info()
```

### Missing Values

```python
df.isnull().sum()
```

### Feature vs Price Analysis

Scatter plots are created for:

* Year vs Price
* Engine Size vs Price
* MPG vs Price
* Tax vs Price

Example:

```python
sns.scatterplot(data=df, x=col, y=df['price'])
plt.show()
```

These visualizations help understand potential relationships between vehicle characteristics and price.

---

## 📁 Project Structure

```text
BMW-Car-Price-Prediction/
│
├── BMW_Car_price_prediction(1).ipynb
├── bmw.csv
└── README.md
```

---

## 🚀 How to Run the Project

### Option 1 — Google Colab

1. Open the notebook in Google Colab.
2. Upload `bmw.csv` to the Colab environment.
3. Run the cells sequentially.
4. Review the exploratory analysis.
5. Train the model.
6. Generate predictions.
7. Evaluate the model.
8. Run the cross-validation section.

---

### Option 2 — Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/jillasrivardhan/BMW-Car-Price-Prediction.git
```

Move into the project directory:

```bash
cd BMW-Car-Price-Prediction
```

Install the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
BMW_Car_price_prediction(1).ipynb
```

Then run the notebook cells sequentially.

---

## 📦 Required Libraries

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Install them using:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## 🔍 Key Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* Regression
* Linear Regression
* Exploratory Data Analysis
* Data Cleaning
* Duplicate Removal
* Feature Engineering
* Numerical Feature Scaling
* Categorical Feature Encoding
* One-Hot Encoding
* `Pipeline`
* `ColumnTransformer`
* Train-Test Split
* Model Evaluation
* MAE
* MSE
* RMSE
* R² Score
* K-Fold Cross-Validation
* Scikit-learn workflows

---

## 💡 What I Learned

Through this project, I practiced building a complete regression workflow from raw data to model evaluation.

### Key takeaways

* How to identify numerical and categorical features
* Why categorical variables need encoding before machine learning
* How feature scaling can be integrated into a pipeline
* How `ColumnTransformer` can apply different transformations to different feature types
* How to build reusable Scikit-learn pipelines
* How to evaluate regression models using multiple metrics
* Why cross-validation is useful for measuring model stability
* How exploratory data analysis can reveal relationships between features and the target

---

## 🚧 Future Improvements

The project can be further improved by experimenting with additional regression algorithms.

Potential improvements include:

* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost
* Random Forest
* Extra Trees Regressor
* Ridge Regression
* Lasso Regression
* Hyperparameter tuning with `GridSearchCV`
* Feature importance analysis
* Residual analysis
* Prediction visualization
* Model comparison
* Model deployment using Streamlit
* Saving the trained model using Joblib
* Building a web-based BMW price prediction application

---

## 📌 Disclaimer

This project is created for **educational and machine learning practice purposes**.

Predicted prices should not be considered professional automotive valuations. Real-world vehicle prices can depend on additional factors such as condition, service history, location, ownership history, optional equipment, market demand, and other factors not represented in the dataset.

---

## 👨‍💻 Author

**Jilla Srivardhan**

GitHub: [Jillas Rivardhan on GitHub](https://github.com/jillasrivardhan?utm_source=chatgpt.com)

---

## ⭐ If You Found This Useful

If you found this project useful for learning machine learning and regression, consider giving the repository a ⭐ on GitHub.

```text
Machine Learning → Data → Preprocessing → Regression → Evaluation → Improvement
```

**Keep learning. Keep building. Keep experimenting. 🚀**
