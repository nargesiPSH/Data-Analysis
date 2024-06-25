# Predict Credit Card Approval
Commercial banks receive _a lot_ of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). 
The mission is to build an automatic credit card approval predictor using machine learning techniques.

### **The Data**

The data is a small subset of the Credit Card Approval dataset from the UCI Machine Learning Repository showing the credit card applications a bank receives. This dataset has been loaded as a `pandas` DataFrame called `cc_apps`. The last column in the dataset is the target value.

### **Import necessary libraries**
```{python}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the dataset
cc_apps = pd.read_csv("cc_approvals.data", header=None)

# Preparing the data
print(sum(cc_apps.isnull()))

# Replacing "?" with NAN
cc_apps_imputed = cc_apps.replace("?", np.NaN)
```
### **Handling missing values**
```{python}
#replacing most fequent value for Categorical data and mean value for numerical data
cc_apps_imputed = cc_apps_imputed.copy()

for col in cc_apps_imputed.columns.values :
    if cc_apps_imputed[col].dtype == "object":
        cc_apps_imputed[col] = cc_apps_imputed[col].fillna(cc_apps_imputed[col].value_counts().index[0])
    else:
        cc_apps_imputed[col] = cc_apps_imputed[col].fillna(cc_apps_imputed[col].mean())

# Preparing Categorical Data
cc_apps_encoded = pd.get_dummies(cc_apps_imputed, drop_first=True)
```
### **Data modeling**
```{python}
# extracting features and taregt values

X= cc_apps_encoded.iloc[:,:-1].values
y = cc_apps_encoded.iloc[:,[-1]].values


# Split data into training and test datasets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

# Using Standard scaler for transformation

scaler = StandardScaler()
Rescaled_X_train = scaler.fit_transform(X_train)
Rescaled_X_test = scaler.fit_transform(X_test)


# Model selection
logreg = LogisticRegression()
logreg.fit(Rescaled_X_train, y_train)

# Prediction on test data
y_train_predict = logreg.predict(Rescaled_X_train)

# Confusion metrix

print(confusion_matrix(y_train, y_train_predict))
```
### **Grid Search for parameter tuning and optimisation**
```{python}
# Define the grid values for tol and max_itr

tol = [0.01, 0.001, 0.0001]
max_iter = [100,150,200]

# create dictionary

param_grid = dict(tol = tol, max_iter = max_iter)

grid_model = GridSearchCV(estimator= logreg, param_grid= param_grid, cv = 5)

grid_model_result = grid_model.fit(Rescaled_X_train, y_train)

# summarise the result
best_train_score, best_train_params = grid_model_result.best_score_, grid_model_result.best_params_

best_model = grid_model_result.best_estimator_
best_score = best_model.score(Rescaled_X_test,y_test)

print("Accuracy :", best_score)
```
