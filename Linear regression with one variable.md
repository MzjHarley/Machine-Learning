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

we get$~y$ value based on input $x$ value via $h$ function.So,$h$ is a function mapping from $x$ to $y$.
One possible expression for h is as follows.
$h_θ(x)=θ_0+θ_1x$  
Since there is only one feature/input variable, such a problem is called linear with one variable regression problem.

------------------------------------------------------------------------------------------------------
# Cost Function
The goal of linear regression algorithm optimization(优化) is to select the straight line that is most likely to fit the data. The error between the data and the straight line is called modeling error(建模误差).  
In order to minimize the modeling error, we need to adjust the parameters $θ_0,θ_1$ so that the value of the cost function $J(θ_0,θ_1)$ is minimized.  
Among various cost functions, the most commonly used is the squared error cost function(平方误差函数).  
## How to select the parameter θ of the model
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/32.png) 

Selecting different parameters $θ_0,θ_1$, the resulting $h$ is different, and the final straight line is also different:

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/34.png) 

## Modeling error
The parameter determines the accuracy of the straight line relative to(相对于) the training set. The difference(差距) between the predicted value of the model and the actual value of the training set (indicated by the blue line in the figure below) is the modeling error.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/33.png)  

adjust parameters $θ_0$,$θ_1$ to minimize the sum of the squares of modeling error.
## Squared error cost function
In order to minimize modeling error, we need to minimize the cost function $J(θ_0,θ_1)$ ,the formula is as follows.  
$J(θ_0,θ_1)=\frac{1}{2m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})^2$  
$(h(x)-y)$ is the difference between the predicted value and the actual value,here we take the sum of its squares, multiplied by $\frac{1}{2m}$ for ease of(为了便于) calculation.  
$m$ refers to the size of the dataset.  
The $J(θ_0,θ_1)$ is usually called the Squared error function, sometimes called the Squared error cost function.  
Find parameters $θ_0$,$θ_1$,minimize $J(θ_0,θ_1)$.   
$\underset{(θ_0,θ_1)}{minimize} ~~~ J(θ_0,θ_1)$   
We draw a contour(等高) plot,the three coordinates are $θ_0,~ θ_1,~ J(θ_0,θ_1)$ , then it can be seen that there is a point in the three-dimensional space that minimizes $J(θ_0,θ_1)$.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/9.png)  

## Intuitive understanding of the cost function I
The hypothesis, parameters, cost function, and goal of the linear regression model are as follows.  
**Hypothesis:**  
$h_θ(x)=θ_0+θ_1x$    
**Parameters:**  
$θ_0,θ_1$  
**Cost Function:**  
$J(θ_0,θ_1)=\frac{1}{2m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})^2$  
**Goal:**   
$\underset{(θ_0,θ_1)}{minimize} ~~~ J(θ_0,θ_1)$      
let$ ~ θ_0=0$,the cost function reduces to a function only about $θ_1:h_θ(x)=θ_1x$.  
In the example below, the coordinates of the three data points are$~(1,1),(2,2),(3,3)$. When$~θ_0=0$ and only $θ_1$ is changed, the cost function is a quadratic curve.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/35.png)
## Intuitive understanding of the cost function II
When both$ ~ θ_0$ and $θ_1$ change, the graph of cost function $J(θ_0,θ_1)$ in three-dimensional space is as follows:  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/36.png)  

since the 3D image looks too complicated, project it to a 2D plane. Introduce the concept of contour plot, also called contour figure. For points on the contour line, the corresponding cost function $J(θ_0,θ_1)$ takes the same value.  
In the following two figures, the line corresponding to the red dot on the right is shown in the left figure, which can be seen that it doesn't fit well.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/37.png)  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/38.png)  

The value in the figure below is located at the lowest point of the 3D graph, and at the center of the contour line on the 2D graph. The corresponding hypothetical function $h(x)$ line is shown on the left. Although there is some error in the fitted data (blue vertical line), it is very close to the minimum.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/39.png)  


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
