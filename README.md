# Import important liberaries
import pandas as pd
from sklearn.linear_model import LinearRegression

# insert the data
data = {
    'area_sq_ft': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'bedrooms': [3, 3, 2, 4, 2, 3, 4, 4, 3, 3],
    'age_years': [5, 8, 15, 2, 20, 10, 1, 3, 7, 9],
    'price_lakhs': [45, 51, 48, 60, 38, 55, 78, 80, 49, 58]
}

# converting data into DataFrame
df = pd.DataFrame(data)

# Seperate input feature and target variable
X = df[['area_sq_ft','bedrooms','age_years']]
y = df['price_lakhs']

# We will create a linear regression object and train our model
model = LinearRegression()

# Fitting the data with ( .fit) function which will train our data 
# in this step we will train the best m1, m2, m3 and c from model.fit(X,y)
model.fit(X,y)

# We will find the Coefficient of (m1, m2 and m3)
coefficient = model.coef_
# intercept
intercept = model.intercept_

print("Model Result")
print(f"Coefficient of (m1,m2,m3) {coefficient}")
print(f"intercept (c) is {intercept}")

print("Meaning of intercept")
print(f"coefficient of area {coefficient[0]:.2f}. means 1sq ft increase in area means price will increase {coefficient[0]:.2f} in lakh and rest feature will constant")
print(f"coefficient of bedroom {coefficient[1]:.2f}. means increase in 1 bedroom then price will increase {coefficient[1]:.2f} in lakh")
print(f"coefficient of age {coefficient[2]:.2f}. means if age increase by 1 year then the price {coefficient[2]:.2f} lakh will decrease bsc this value is negative")

# We want to predict price for new house
# Area = 1800 sq ft
# bedrooms = 3
# age = 5yrs

new_house = [[1800,3,5]]
predict_price = model.predict(new_house)

print("Price of new house will be")
print(f"area = 1800 sq ft/n 3 Bedrooms/n 5years old house price will be {predict_price[0]:.2f} lakh")

#results :
Model Result
Coefficient of (m1,m2,m3) [0.02551236 9.33933225 0.67755177]
intercept (c) is -21.926033366554165
Meaning of intercept
coefficient of area 0.03. means 1sq ft increase in area means price will increase 0.03 in lakh and rest feature will constant
coefficient of bedroom 9.34. means increase in 1 bedroom then price will increase 9.34 in lakh
coefficient of age 0.68. means if age increase by 1 year then the price 0.68 lakh will decrease bsc this value is negative
Price of new house will be
area = 1800 sq ft/n 3 Bedrooms/n 5years old house price will be 55.40 lakh #

# I want to understand with the age and salary of a customer that he will purchase a product or not?

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'Age': [25, 30, 45, 22, 35, 40, 28, 50, 55, 60],
    'Salary': [40000, 50000, 80000, 30000, 60000, 75000, 45000, 90000, 120000, 130000],
    'Purchased': [0, 0, 1, 0, 0, 1, 0, 1, 1, 1] }

df = pd.DataFrame(data)

## seperate Input (X) and output(y) feature  

X = df[['Age','Salary']]
y = df['Purchased']

## We will train the data and split it
## divide the data into 2 parts 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

## Feature Scaling
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

## Model training

model = LogisticRegression()
model.fit(X_train_scaler,y_train)

# Prediction in test data
y_pred = model.predict(X_test_scaler)

# Checking the probability
y_pred_prob = model.predict_proba(X_test_scaler)

print("Prediction on test data")
print(f"Actual values y_test {y_test.values}")
print(f"Predicted values y_pred {y_pred}")
print(y_pred_prob)

# results 
Prediction on test data
Actual values y_test [1 0]
Predicted values y_pred [1 0]
[[0.0508183  0.9491817 ]
 [0.74385733 0.25614267]]
 #

 import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = titanic[['survived','pclass','sex','age','fare']].dropna()

df['sex'] = df['sex'].map({'male':0,'female':1})

X = df[['pclass','sex','age','fare']]
y = df['survived']
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# DT model
dt_model = DecisionTreeClassifier(max_depth=3,random_state=42)
dt_model.fit(x_train,y_train)

#prediction
y_pred = dt_model.predict(x_test)

# accuracy
print("The accuracy score is :",accuracy_score(y_test,y_pred))

# results
X = df[['pclass','sex','age','fare']]
y = df['survived']
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# DT model
dt_model = DecisionTreeClassifier(max_depth=3,random_state=42)
dt_model.fit(x_train,y_train)

#prediction
y_pred = dt_model.predict(x_test)

# accuracy
print("The accuracy score is :",accuracy_score(y_test,y_pred))
#
## Graph

import matplotlib.pyplot as plt
from sklearn import tree

# Plot the decision tree
plt.figure(figsize=(15, 8))
tree.plot_tree(
    dt_model,
    feature_names=X.columns.tolist(),        
    class_names=['Not Survived', 'Survived'], 
    filled=True, 
    rounded=True
)
plt.show()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Importing ML liberaries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import statsmodels.api as ap


df = pd.read_csv(r"C:\Users\JayantRai\Desktop\Dataset\Customer_churn\Bank Customer Churn Prediction.csv")
df.head()
plt.figure(figsize=(12,9))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title("Correlation Heatmap")
df = df.drop(columns=['customer-id'],errors='ignore')
df.shape
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['country'] = le.fit_transform(df['country'])
x = df.drop('churn',axis=1)
y = df['churn']
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train,y_train)
y_pred_log = log_model.predict(X_test)

print("---- Logistic Regression-----")
print("Classification report :\n", classification_report(y_test,y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:,1]))

# results ---- Logistic Regression-----
Classification report :
               precision    recall  f1-score   support

           0       0.80      0.97      0.88      2389
           1       0.33      0.05      0.08       611

    accuracy                           0.79      3000
   macro avg       0.56      0.51      0.48      3000
weighted avg       0.70      0.79      0.72      3000

ROC AUC: 0.7587798413212768 #
coeff = pd.DataFrame(log_model.coef_[0],index=x.columns, columns=['Coefficient'])
print(coeff)
# results                    Coefficient
customer_id      -2.263826e-07
credit_score     -1.051146e-03
country           4.853679e-04
gender           -1.112157e-03
age               5.828003e-02
tenure            4.275075e-04
balance           4.776942e-06
products_number  -1.681358e-06
credit_card      -4.727925e-05
active_member    -1.581496e-03
estimated_salary  1.022495e-06
#
x_const = ap.add_constant(X_train)
sm_model = ap.Logit(y_train, x_const).fit(disp=False)
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# results 
--- Decision Tree ---
Accuracy: 0.8586666666666667
Classification Report:
               precision    recall  f1-score   support

           0       0.86      0.98      0.92      2389
           1       0.82      0.39      0.53       611

    accuracy                           0.86      3000
   macro avg       0.84      0.68      0.72      3000
weighted avg       0.85      0.86      0.84      3000

#
