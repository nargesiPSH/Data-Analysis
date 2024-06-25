# Facial Recognition
The  mission is to deploy AI-driven solutions that can accurately distinguish between images of notable personalities and the general populace, enhancing the personal security of such high-profile individuals. The project to focus on Arnold Schwarzenegger, a figure whose accomplishments span from bodybuilding champion to Hollywood icon, and from philanthropist to the Governor of California. 

The goal is to find the best model (Supervised Learning) and their parameters in distinguishing images.

### **The Data**
The `data/lfw_arnie_nonarnie.csv` dataset contains processed facial image data derived from the "Labeled Faces in the Wild" (LFW) dataset, focusing specifically on images of Arnold Schwarzenegger and other individuals not identified as him. This dataset has been prepared to aid in the development and evaluation of facial recognition models. There are 40 images of Arnold Schwarzenegger and 150 of other people.

```{python}
# Import required libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file 
df = pd.read_csv("data/lfw_arnie_nonarnie.csv")

# Seperate the predictor and class label
X = df.drop('Label', axis=1)
y = df['Label'] 

# Split the data into training and testing sets using stratify to balance the class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### **Initialising three models**
```{python}
# import the models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
```
### **Store the initialised model in a dictionary**
models = { "LogisticRegression" : LogisticRegression(),
          "KNeighborsClassifier" : KNeighborsClassifier(),
          "DecisionTreeClassifier" : DecisionTreeClassifier()
}

### **Store model parameters in a dictionary**
```{python}
param_grid = {
    "LogisticRegression" : {"LogisticRegression__C" : [0.01, 0.1, 1, 10]},
    "KNeighborsClassifier" : {"KNeighborsClassifier__n_neighbors" : range(1,10)},
    "DecisionTreeClassifier": {"DecisionTreeClassifier__max_depth":[2,5,10],
                             "DecisionTreeClassifier__min_samples_split" : [2,5,10,20],
                             "DecisionTreeClassifier__random_state":[42]}
}
```
### **Pipeline and iterative process to find the best model and its parameters**
```{python}
# using KFold 
kf = KFold(n_splits = 5, random_state = 42, shuffle = True)

pipe_accuracies = {}
pipe_params = {}
pipelines = {}

# Iterative process to find the best model parameters
for name, model in models.items():
    pipeline = Pipeline(steps=[("scaler", StandardScaler()),(name, model)])  
    grid_search = GridSearchCV(pipeline, param_grid[name], cv= kf, scoring="accuracy")
    
    grid_search.fit(X_train,y_train)
    pipe_accuracies[name] = grid_search.best_score_
    pipe_params[name] = grid_search.best_params_
    pipelines[name] = grid_search 

best_model_name = max(pipe_accuracies)
best_model_cv_score = max(pipe_accuracies.values())
best_model_info = pipe_params[best_model_name]

print(f"Best Model : {best_model_name}")
print(f"Best Model Parameters : {best_model_info}")
print(f"BEst Model CV Score : {best_model_cv_score}")


y_pred = pipelines[best_model_name].predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```
Best Model : LogisticRegression

Best Model Parameters : {'LogisticRegression__C': 1}

BEst Model CV Score : 0.8286021505376345

Accuracy: 0.7368



