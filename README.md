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
