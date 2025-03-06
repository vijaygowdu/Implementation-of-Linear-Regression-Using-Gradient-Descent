## Implementation of Simple Linear Regression Model Using Gradient descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:

## Program to implement the linear regression using gradient descent.
## Developed by: k vijay
## RegisterNumber:23004034
 ```
 import numpy as np
 import pandas as pd
 from sklearn.preprocessing import StandardScaler
 def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
 data=pd.read_csv('50_Startups.csv',header=None)
 X=(data.iloc[1:, :-2].values)
 X1=X.astype(float)
 scaler=StandardScaler()
 y=(data.iloc[1:,-1].values).reshape(-1,1)
 X1_Scaled=scaler.fit_transform(X1)
 Y1_Scaled=scaler.fit_transform(y)
 print(data.head())
 theta=linear_regression(X1_Scaled, Y1_Scaled)
 new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
 new_Scaled=scaler.fit_transform(new_data)
 prediction=np.dot(np.append(1, new_Scaled),theta)
 prediction=prediction.reshape(-1,1)
 pre=scaler.inverse_transform(prediction)
 print(f"Predicted value: {pre}")
```

## Output:

![Screenshot (44)](https://github.com/user-attachments/assets/757c4adb-fc63-496a-84fd-06098f20578c)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
