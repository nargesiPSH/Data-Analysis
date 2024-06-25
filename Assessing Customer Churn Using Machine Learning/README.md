# Assessing Customer Churn Using Machine Learning
The telecommunications (telecom) sector in India is rapidly changing, with more and more telecom businesses being created and many customers deciding to switch between providers. "Churn" refers to the process where customers or subscribers stop using a company's services or products. Understanding the factors that influence keeping a customer as a client in predicting churn is crucial for telecom companies to enhance their service quality and customer satisfaction. As the data scientist on this project, you aim to explore the intricate dynamics of customer behavior and demographics in the Indian telecom sector in predicting customer churn, utilizing two comprehensive datasets from four major telecom partners: Airtel, Reliance Jio, Vodafone, and BSNL:

### **The Data**

- `telecom_demographics.csv` contains information related to Indian customer demographics:

| Variable             | Description                                      |
|----------------------|--------------------------------------------------|
| `customer_id `         | Unique identifier for each customer.             |
| `telecom_partner `     | The telecom partner associated with the customer.|
| `gender `              | The gender of the customer.                      |
| `age `                 | The age of the customer.                         |
| `state`                | The Indian state in which the customer is located.|
| `city`                 | The city in which the customer is located.       |
| `pincode`              | The pincode of the customer's location.          |
| `registration_event` | When the customer registered with the telecom partner.|
| `num_dependents`      | The number of dependents (e.g., children) the customer has.|
| `estimated_salary`     | The customer's estimated salary.                 |

- `telecom_usage` contains information about the usage patterns of Indian customers:

| Variable   | Description                                                  |
|------------|--------------------------------------------------------------|
| `customer_id` | Unique identifier for each customer.                         |
| `calls_made` | The number of calls made by the customer.                    |
| `sms_sent`   | The number of SMS messages sent by the customer.             |
| `data_used`  | The amount of data used by the customer.                     |
| `churn`    | Binary variable indicating whether the customer has churned or not (1 = churned, 0 = not churned).|

### **Import libraries and methods/functions**
```{python}
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
```
### **Read Data from files**
```{python}
telecom_demographics = pd.read_csv("telecom_demographics.csv")
telecom_usage = pd.read_csv("telecom_usage.csv")

# Checking data status
print(telecom_demographics.info())
print(telecom_usage.info())
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6500 entries, 0 to 6499
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype 
---  ------              --------------  ----- 
 0   customer_id         6500 non-null   int64 
 1   telecom_partner     6500 non-null   object
 2   gender              6500 non-null   object
 3   age                 6500 non-null   int64 
 4   state               6500 non-null   object
 5   city                6500 non-null   object
 6   pincode             6500 non-null   int64 
 7   registration_event  6500 non-null   object
 8   num_dependents      6500 non-null   int64 
 9   estimated_salary    6500 non-null   int64 
dtypes: int64(5), object(5)
memory usage: 507.9+ KB
None
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6500 entries, 0 to 6499
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   customer_id  6500 non-null   int64
 1   calls_made   6500 non-null   int64
...
 4   churn        6500 non-null   int64
dtypes: int64(5)
memory usage: 254.0 KB
None
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```
### **Merging the data into one dataframe**
```{python}
churn_df = telecom_demographics.merge(telecom_usage, on="customer_id")
print(churn_df)
```
```
      customer_id telecom_partner gender  age             state       city  \
0           15169          Airtel      F   26  Himachal Pradesh      Delhi   
1          149207          Airtel      F   74       Uttarakhand  Hyderabad   
2          148119          Airtel      F   54         Jharkhand    Chennai   
3          187288    Reliance Jio      M   29             Bihar  Hyderabad   
4           14016        Vodafone      M   45          Nagaland  Bangalore   
...           ...             ...    ...  ...               ...        ...   
6495        78836          Airtel      M   54            Odisha    Chennai   
6496       146521            BSNL      M   69    Andhra Pradesh  Hyderabad   
6497        40413          Airtel      M   19           Gujarat  Hyderabad   
6498        64961        Vodafone      M   26         Meghalaya    Chennai   
6499        60427    Reliance Jio      M   19           Mizoram  Hyderabad   

      pincode registration_event  num_dependents  estimated_salary  \
0      667173         2020-03-16               4             85979   
1      313997         2022-01-16               0             69445   
2      549925         2022-01-11               2             75949   
3      230636         2022-07-26               3             34272   
4      188036         2020-03-11               4             34157   
...       ...                ...             ...               ...   
6495   125785         2021-01-29               4            124805   
6496   923076         2022-01-03               1             65605   
6497   152201         2020-07-21               0             28632   
6498   782127         2020-11-21               3            119757   
6499   918496         2020-10-29               3            119368   
...
6498          52         8       6835      0  
6499          90        12       5531      0  

[6500 rows x 14 columns]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```
### **Training the model**
```{python}
# Dealing with Categorical data
churn_df_encoded = pd.get_dummies(churn_df, drop_first=True)


X = churn_df_encoded.drop(["churn"], axis = 1)
y = churn_df_encoded["churn"]

# Splitting train test datasets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

# Scaling
scaler = StandardScaler()
Rescaled_X_train = scaler.fit_transform(X_train)
Rescaled_X_test = scaler.fit_transform(X_test)

ridge_model = RidgeClassifier()
random_forest = RandomForestClassifier()

# Training
ridge_model.fit(Rescaled_X_train, y_train)

random_forest.fit(Rescaled_X_train, y_train)


# Models Predictions

logreg_pred = ridge_model.predict(Rescaled_X_train)
rf_pred = random_forest.predict(Rescaled_X_train)


# Confusion Matrix

logreg_confusion = confusion_matrix(y_train,logreg_pred)
rf_confusion = confusion_matrix(y_train,rf_pred)
```

