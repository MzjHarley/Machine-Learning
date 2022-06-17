# Model Representation
##  Linear regression
The example of house price prediction,Training set are as follows:  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/1.png)  
The meaning of each variable is defined as follows:
|variable|meaning|
|---|---|
|$m$|Number of training examples|
|$x$|"input" variable/features|
|$y$|"output" variable/"target" variable|
|$(x,y)$|one training example|
|$x^{(i)}$,$y^{(i)}$| $i^{th}$ training example|
|$h$|hypothesis function|
##  Linear regression with one variable
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/2.png)  

we get y value based on input x value via h function.So,h is a function mapping from x to y.
One possible expression for h is as follows.
$h_θ(x)=θ_0+θ_1x$  
Since there is only one feature/input variable, such a problem is called linear with one variable regression problem.

------------------------------------------------------------------------------------------------------
# Cost Function
The goal of linear regression algorithm optimization(优化) is to select the straight line that is most likely to fit the data. The error between the data and the straight line is called modeling error.  
In order to minimize the modeling error, we need to adjust the parameters $θ_0$,$θ_1$ so that the value of the cost function $J(θ_0,θ_1)$ is minimized.  
Among various cost functions, the most commonly used is the squared error cost function.  
## How to select the parameter θ of the model
$h_θ(x)=θ_0+θ_1x$,it corresponds to two model parameters $θ_0$,$θ_1$ .

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/4.png) 

## Modeling error
## Squared error cost function
## Intuitive understanding of the cost function I
## Intuitive understanding of the cost function II
------------------------------------------------------------------------------------------------------
# Gradient descent
## Local optimum
## Gradient descent algorithm
## Intuitive understanding of Gradient descent 
### Update rule of Gradient descent algorithm
### The selection of learning rate α
### without adjusting α,J(θ) can also converge
------------------------------------------------------------------------------------------------------
# Gradient Descent For Linear Regression
## Gradient descent and linear regression combined
## batch gradient descent
