# Medical Insurance Prediction 🏥

## 📌 Project Objective
The goal of this project is to predict medical insurance costs (`claim`) based on different personal, lifestyle, and health-related features.
This project aims to explore and compare different Machine Learning regression models on a real-world dataset.

# 📊 Dataset
The dataset contains information about individuals, including:

- Age
- Gender
- BMI
- Number of dependents
- Lifestyle information
- Medical information
- City
- Job title
- Insurance cost (`claim`)
The target variable is:
claim
This is the value that the Machine Learning models will learn to predict.

# 🔧 Data Preparation

## 1. Loading the Data
The dataset is loaded using Pandas:
```
data = pd.read_csv("dataset.csv")
```

## 2. Splitting Features and Target
The dataset is separated into:

- `X` → input features used for prediction
- `y` → target variable (`claim`)
```
X = data.drop("claim", axis=1)
y = data["claim"]
```

## 3. Encoding Categorical Features
Some features contain text values:

- `city`
- `job_title`
- `hereditary_diseases`
Machine Learning models cannot directly process text categories.
Therefore, categorical features are converted into numerical values using:
```
X = pd.get_dummies(X, dtype=int)
```

## 4. Train/Test Split
The dataset is divided into:

- 80% training data
- 20% testing data
The training set is used to teach the model, while the test set evaluates how well it generalizes to unseen data.
```
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

# 🤖 Models Tested

## 📈 Linear Models
Linear models are used as the first baseline models for this project.
They try to find relationships between the input features and the insurance cost (`claim`).
These models are useful because they are:

- Simple to understand
- Fast to train
- Good references before testing more complex algorithms

## 1. 📊 Linear Regression

### Description
Linear Regression was the first baseline model used in this project.
It tries to find a linear relationship between the input features and the target variable (`claim`).

### Results
**Mean Absolute Error (MAE):**
3988.56

**R² Score:**
0.7489

### Analysis
Linear Regression achieved an R² score of approximately 0.75.
This means that the model explains around 75% of the variation in insurance costs.
As a first baseline model, the results are promising and provide a reference point for future improvements.

## 2. 🛡️ Ridge Regression

### Description
Ridge Regression is a Linear Regression model with L2 regularization.
It reduces the impact of large coefficients to make the model more stable, especially when features are correlated.

### Results
**Mean Absolute Error (MAE):**
3988.56

**R² Score:**
0.7489

### Analysis
Ridge Regression achieved the same performance as Linear Regression with the default parameters.
This suggests that the original Linear Regression model was already stable and that L2 regularization did not provide a significant improvement on this dataset.

## 3. ✂️ Lasso Regression

### Description
Lasso Regression is a Linear Regression model using L1 regularization.
Unlike Ridge, Lasso can reduce some feature coefficients to zero, which allows it to perform feature selection.

### Results
**Mean Absolute Error (MAE):**
3976.25

**R² Score:**
0.7495

### Analysis
Lasso Regression slightly improved the results compared to Linear Regression and Ridge Regression.
The improvement is small, but it suggests that reducing the influence of some features helped the model generalize slightly better.
Further experiments with more complex nonlinear models will determine whether additional patterns can be captured.

# 🌳 Tree-Based Models
Tree-based models use decision rules instead of mathematical formulas.
They are able to capture complex and nonlinear relationships between features.
Advantages:

- 🌲 Can learn complex patterns
- 🔀 Handles nonlinear relationships
- 🧩 Easy to visualize and understand

## 4. 🌳 Decision Tree Regressor

### 🔎 Description
Decision Tree Regressor predicts values by creating a sequence of decision rules based on feature splits.
Instead of finding a mathematical equation like linear models, it learns rules from the data.
Example:
```
IF smoker = yes
AND age > 40
THEN claim is probably high
```

### 📈 Initial Results (No Restrictions)
**Mean Absolute Error (MAE):**
413.95

**R² Score:**
0.9544

### 🧪 Overfitting Detection
To analyze the model behavior, training and testing performance were compared.
**Training R²:**
1.0000

**Testing R²:**
0.9583

### ⚠️ Analysis
The perfect training score shows that the Decision Tree was able to completely fit the training data.
Although the test score remained very high, the difference between training and testing performance indicated possible overfitting.
The model was learning some specific patterns from the training data instead of only learning general rules.

# 🛠️ Controlling Tree Complexity with `max_depth`
To reduce overfitting, the maximum depth of the tree was limited.
The goal was to create a simpler model that generalizes better.

### Results with `max_depth`
**Training R²:**
0.8614

**Testing R²:**
0.8464

### 📝 Analysis
Reducing the tree depth successfully prevented the model from memorizing the training data.
However, the performance decreased significantly compared to the unrestricted tree.
This shows the trade-off between:

- 🧠 Model complexity
- 📚 Learning capacity
- 🎯 Generalization ability
The next step will be to test ensemble methods, which combine multiple trees to achieve better performance while reducing overfitting.

# 🌲 Ensemble Learning Models
Ensemble learning combines multiple models to create a stronger and more stable predictor.
Instead of relying on a single decision tree, ensemble methods combine multiple trees to reduce errors and improve generalization.

## 5. 🌲 Random Forest Regressor

### 🔎 Description
Random Forest Regressor is an ensemble model that combines multiple Decision Trees.
Each tree learns different patterns from the data, and their predictions are combined to produce a final prediction.
This helps reduce the instability and overfitting problems of individual decision trees.

### 📈 Results
**Training R²:**
0.8763

**Testing R²:**
0.8702

### 📝 Analysis
Random Forest achieved better generalization than the limited Decision Tree.
The training and testing scores are close, showing that the model is less prone to overfitting.
However, the unrestricted Decision Tree achieved a higher test score, showing that some datasets can still benefit from highly complex individual trees.
Further optimization and more advanced ensemble methods will be explored.

# 🚀 Boosting Models
Boosting models build a sequence of models where each new model tries to correct the errors made by previous models.
Unlike Random Forest, where trees work independently, boosting models create a chain of improvements.

## 6. 🚀 Gradient Boosting Regressor

### 🔎 Description
Gradient Boosting Regressor is an ensemble model that combines multiple Decision Trees sequentially.
Each new tree focuses on correcting the errors made by previous trees.
This allows the model to capture complex nonlinear relationships while maintaining good generalization.

### 📈 Results
**Training R²:**
0.8725
**Testing R²:**
0.8658

## ⚡ Advanced Boosting Models
XGBoost is an optimized implementation of Gradient Boosting.
It improves the original algorithm with additional regularization techniques and performance optimizations.

## 7. ⚡ XGBoost Regressor

### 🔎 Description
XGBoost Regressor is an advanced boosting algorithm that builds multiple Decision Trees sequentially.
Each tree corrects the errors made by previous trees while using additional techniques to reduce overfitting.

### 📈 Results
**Training R²:**
0.9907
**Testing R²:**
0.9604

### 📝 Analysis
XGBoost achieved the best testing performance among all models tested so far.
Without any hyperparameter tuning, it slightly outperformed the Decision Tree and significantly improved over Random Forest and Gradient Boosting.
The small gap between training and testing performance shows that the model is powerful while maintaining good generalization.
The next step will be hyperparameter tuning to optimize the model further.

# ⚙️ Hyperparameter Experiment — n_estimators
The parameter n_estimators controls the number of trees used by the model.
Increasing this value allows the model to perform more correction steps and improve the model's predictions.
**Experiment:**
n_estimators = 500

### 📈 Results with n_estimators = 500
**Training R²:**
0.9334
**Testing R²:**
0.9126

### 📝 Analysis
Increasing the number of estimators significantly improved the model performance.
The testing score increased from 0.8658 to 0.9126, showing that additional boosting steps helped the model learn more complex patterns.
The gap between training and testing scores remained reasonable, meaning the model improved without strong overfitting.
This experiment demonstrates the importance of hyperparameter tuning in Machine Learning.
The next step will be to explore more advanced boosting algorithms such as XGBoost.

# ⚙️ Hyperparameter Experiment — max_depth
The parameter max_depth controls the maximum depth of each Decision Tree inside XGBoost.
A higher value allows trees to learn more complex patterns, but it can also increase the risk of overfitting.
Several values were tested:

### max_depth = 2
**Training R²:**
0.8551
**Testing R²:**
0.8467

### max_depth = 3
**Training R²:**
0.9138
**Testing R²:**
0.8981

### max_depth = 5
**Training R²:**
0.9803
**Testing R²:**
0.9565

### max_depth = 7
**Training R²:**
0.9973
**Testing R²:**
0.9636

### 📝 Analysis
Increasing max_depth improved the model performance by allowing the trees to learn more complex relationships.
Very small values caused underfitting because the model was too simple.
The best result was achieved with max_depth=7, reaching a Testing R² score of 0.9636.
Although the training score is very high, the gap with testing performance remains acceptable, showing that the model still generalizes well.

# ⚙️ Hyperparameter Experiment — `learning_rate`
The `learning_rate` parameter controls how much each new tree corrects the errors made by the previous trees.
A high learning rate makes large corrections, while a low learning rate makes smaller and more gradual corrections.
Several values were tested to observe their impact on model performance.

### 📈 Results
| Learning Rate | Training R² | Testing R² |
|---|---|---|
| 1.0 | 0.9995 | 0.9559 |
| 0.3 (default) | 0.9907 | 0.9604 |
| 0.1 | 0.9622 | 0.9414 |
| 0.01 | 0.7569 | 0.7511 |

### 📝 Analysis
The default value (`learning_rate = 0.3`) achieved the best balance between training and testing performance.
A very high learning rate (`1.0`) allowed the model to learn extremely quickly but slightly reduced its ability to generalize.
Reducing the learning rate to `0.1` decreased the model's performance because each tree corrected the errors more cautiously.
With a very small learning rate (`0.01`), the model severely underfit the data. Since the number of estimators remained unchanged, the model did not have enough trees to compensate for the small correction applied at each boosting step.
This experiment highlights an important concept in Gradient Boosting and XGBoost: **`learning_rate`**** and ****`n_estimators`**** should usually be tuned together**. A smaller learning rate often requires a larger number of estimators to achieve optimal performance.

# 🤖 Automated Hyperparameter Tuning
GridSearchCV is a Scikit-Learn tool that automatically tests different combinations of hyperparameters to find the best configuration.
Instead of manually testing each value, GridSearchCV evaluates multiple models and selects the one with the best cross-validation score.

## ⚙️ GridSearchCV Experiment — XGBoost max_depth
The parameter tested:
**max_depth:**
[3, 5, 7]
The goal was to automatically find the best tree depth for the XGBoost model.

### 📈 Results
**Best Parameters:**
max_depth = 7
**Training R²:**
0.9973
**Testing R²:**
0.9636
**Cross Validation Score:**
0.9626

### 📝 Analysis
GridSearchCV successfully found the same optimal parameter that was discovered through manual testing.
The best configuration was max_depth=7, which achieved the highest testing performance.
This experiment shows how automated hyperparameter tuning can replace manual experimentation when many parameters need to be tested.
The next step will be to explore more advanced tuning methods by combining multiple hyperparameters together.

# 🤖 Automated Hyperparameter Optimization with GridSearchCV
After manually testing individual hyperparameters, GridSearchCV was used to automatically search for the best combination of parameters.
Instead of testing one parameter at a time, GridSearchCV evaluates multiple combinations and selects the configuration that achieves the best cross-validation score.

## ⚙️ Parameters Tested
The following hyperparameters were explored:

- `max_depth`: Controls the maximum depth of each tree.
- `learning_rate`: Controls how much each tree contributes to the final prediction.
- `n_estimators`: Defines the number of trees used in the boosting process.
Parameter grid:
**max_depth:**
[3, 5, 7]
**learning_rate:**
[0.1, 0.3]
**n_estimators:**
[100, 500]
Total combinations tested:
3 × 2 × 2 = 12 models

### 🏆 Best Hyperparameters Found
GridSearchCV selected:

- `max_depth = 7`
- `learning_rate = 0.1`
- `n_estimators = 500`

### 📈 Final Results
**Training R²:**
0.9986
**Testing R²:**
0.9652

### 📝 Analysis
GridSearchCV improved the model compared to the default XGBoost configuration.
The best configuration used a deeper tree structure combined with a lower learning rate and a higher number of estimators.
The lower learning rate allowed the model to learn more gradually, while the increased number of trees provided enough boosting steps to reach better performance.
This experiment demonstrated an important relationship between hyperparameters:

- 🐢 Lower `learning_rate` usually requires more `n_estimators`.
- 🌳 Higher `max_depth` increases model complexity and learning capacity.
The optimized XGBoost model achieved the best performance obtained during this project.
The next step will be to improve the project structure using Scikit-Learn Pipelines, which will combine preprocessing and model training into a single workflow.

# 💾 Model Saving with Joblib
After finding the best XGBoost configuration, the trained model was saved using Joblib.
The goal was to avoid retraining the model every time it is needed.
Instead of:
Training data
↓
Model training
↓
Prediction
The workflow becomes:
Training data
↓
Model training
↓
Save model (.pkl)
Later:
Saved model
↓
Load model
↓
Prediction

## 📦 Saving the Model
The trained XGBoost model was stored as:
AI_model.pkl
```
The function used:
```
`joblib.dump(model, "AI_model.pkl")`
```

```

## 🧠 Understanding dump()
`joblib.dump()` acts like an archive system.
It takes the trained model containing all learned patterns and stores it inside a file.
The model can then be recovered later without training again.

## 📂 Loading the Model
The saved model can be recovered using:
`joblib.load("AI_model.pkl")`
```
This allows the project to directly use the already trained model for predictions.
```

### 📝 Analysis
Saving the model makes the project more practical.
The training phase only needs to be performed once.
After that, the model can be loaded instantly and used for new predictions.
This is an important step toward deploying a Machine Learning model in a real application.

# 💾 Model Persistence with Joblib
After selecting the best XGBoost configuration, the trained model was saved using Joblib.
Saving the model avoids retraining it every time the project is executed, making predictions significantly faster and preparing the project for real-world usage.
The trained model is stored as:
AI_model.pkl
```
A dedicated prediction script (predict.py) was also created.
```
Instead of training a new model, this script simply loads the saved model, imports the exported test dataset, performs predictions, and evaluates the model using the R² score.
This separation between training and prediction follows common Machine Learning development practices.

## 📁 Project Structure
```
Insurance-Claim-Prediction/
│
├── main.py               # Data preprocessing and model training
├── predict.py            # Loads the saved model and performs predictions
├── dataset.csv
├── AI_model.pkl
├── X_test_data.csv
├── y_test_data.csv
├── README.md
└── requirements.txt
```

## 🚀 How to Run

1. **Install the required dependencies:**

   ```
   pip install -r requirements.txt
   ```

2. **Train the model (or regenerate the exported test files if needed):**

   ```
   python main.py
   ```

3. **Run predictions using the saved model:**

   ```
   python predict.py
   ```

The prediction script loads the trained model from `AI_model.pkl`, predicts the insurance claim values for the exported test dataset, and displays the final R² score.

## 👤 Author
**Adymoor**
Machine Learning learning project focused on regression algorithms, model comparison, hyperparameter tuning, and model deployment using XGBoost and Joblib.

 
 
