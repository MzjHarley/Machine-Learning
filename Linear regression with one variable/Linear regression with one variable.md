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
Since there is only one feature/input variable, such a problem is called linear regression with one variable  problem.

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

Since the 3D image looks too complicated, project it to a 2D plane. Introduce the concept of contour plot, also called contour figure. For points on the contour line, the corresponding cost function $J(θ_0,θ_1)$ takes the same value.  
In the following two figures, the line corresponding to the red dot on the right is shown in the left figure, which can be seen that it doesn't fit well.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/37.png)  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/38.png)  

The value in the figure below is located at the lowest point of the 3D graph, and at the center of the contour line on the 2D graph. The corresponding hypothetical function $h(x)$ line is shown on the left. Although there is some error in the fitted data (blue vertical line), it is very close to the minimum.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/39.png)  

------------------------------------------------------------------------------------------------------
# Gradient descent
## Local optimum
There is already a cost function, and our goal is to minimize it. Normally, start with$ ~ θ_0=0,θ_1=0$, adjust $θ_0, θ_1$, and end at the minimum value of $J(θ_0, θ_1)$.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/40.png)  

In the example below,$θ_0$ and $θ_1$ don't start at 0,0. When selecting two different starting points and gradient descent along different directions, we will reach two different optimal solutions, which are called local optimum solutions.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/41.png)![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/42.png)  
## Gradient descent algorithm
The gradient descent algorithm assigns a value to$ ~ θ$, so that $J(θ)$ proceeds in the fastest direction of gradient descent, and iterates continuously until a local minimum value is finally obtained, that is, convergence(收敛). The gradient descent algorithm is not only used for linear regression, but can be used to minimize any cost function $J$. The formula is as follows.  
$θ_j:=θ_j-\alpha\frac{\partial}{\partial{θ_j}}J(θ)$  
$\alpha$ is learning rate,which determines how big a step is taken down in the direction that reduces the cost function the most.  
In the gradient descent algorithm, the two parameters are updated simultaneously(同步地) (bottom left). If it is a non-simultaneous update (bottom right), it is not gradient descent.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/43.png)   

## Intuitive understanding of Gradient descent 
### Update rule of Gradient descent algorithm
The gradient descent algorithm is as follows.  
$θ_j:=θ_j-\alpha\frac{\partial}{\partial{θ_j}}J(θ)$  
The purpose of derivation(求导) can basically be said to take the tangent of the red point, that is, the red line. Since the slope(斜率) on the right side of the curve is positive, the derivative term is positive. Therefore,$ ~ θ_1$ minus a positive number multiplied by $α$, the value becomes smaller.
The slope on the left side of the curve is negative and the derivative term is negative. Therefore,$ ~ θ_1$ minus a negative number multiplied by $α$, the value becomes larger.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/44.png)  

### The selection of learning rate α
If $α$ is too small, it can only descend(下降) in small steps, and it takes many steps to reach the global minimum, which is very slow.  
If $α$ is too large, the algorithm may go over the lowest point. Crossed the lowest point again and again, farther and farther away from it. It will lead to failing to converge, or even divergence.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/45.png)   

### without adjusting α,J(θ) can also converge  
Suppose$ ~ θ_1$ is initialized at the local minimum. A derivative of 0 will make $θ_1$ no longer change, and won't change the value of the parameter.  
It also explains why gradient descent can converge to a local minimum even when the learning rate $α$ remains constant.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/46.png)  

Why is it possible to reach the local optimum without adjusting $α$? Because after one step of gradient descent, the new derivative will become smaller, and the magnitude(量级) of the movement will automatically become smaller. Until the final movement is very small, it has converged to a local minimum.  

![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Linear%20regression%20with%20one%20variable/48.png)   

------------------------------------------------------------------------------------------------------
# Gradient Descent For Linear Regression
## Gradient descent and linear regression combined
Combining the squared error function$ ~ h(x)$ with the gradient descent method and the squared cost function $J(Θ)$, we get the first machine learning algorithm, that is Linear Regression.  
**gradient descent algorithm:**  
$repeat ~ until ~ convergence \lbrace$  
$θ_j:=θ_j-\alpha\frac{\partial}{\partial{θ_j}}J(θ)$  
$for(j=0 ~ and ~ j=1)$  
$\rbrace$  
**Linear Regression Model:**  
$h_θ(x)=θ_0+θ_1x$   
$J(θ_0,θ_1)=\frac{1}{2m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})^2$   
Using the gradient descent method for the previous linear regression problem, the key is to find the derivative of the cost function $J(θ)$
$\frac{\partial}{\partial{θ_j}}J(θ)=\frac{\partial}{\partial{θ_j}}\frac{1}{2m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})^2$  
$ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =\frac{1}{m}\sum\limits_{i=1}\limits^m\[(h_θ(x^{(i)})-y^{(i)})\frac{\partial}{\partial{θ_j}}(h_θ(x^{(i)})-y^{(i)})\]$  
$ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =\frac{1}{m}\sum\limits_{i=1}\limits^m\[(h_θ(x^{(i)})-y^{(i)})x_j^{(i)}\]$  
when$ ~ j=0$,$\frac{\partial}{\partial{θ_j}}J(θ_0)=\frac{1}{m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})$.  
when$ ~ j=1$,$\frac{\partial}{\partial{θ_j}}J(θ_1)=\frac{1}{m}\sum\limits_{i=1}\limits^m\[(h_θ(x^{(i)})-y^{(i)})x_1^{(i)}\]$.  
Bring the above two derivatives into the gradient descent algorithm to replace the original $\frac{\partial}{\partial{θ_j}}J(θ)$,the gradient descent algorithm becomes:  
$repeat ~ until ~ convergence \lbrace$  
$θ_0:=θ_0-\frac{\alpha}{m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})$  
$θ_1:=θ_1-\frac{\alpha}{m}\sum\limits_{i=1}\limits^m\[(h_θ(x^{(i)})-y^{(i)})x_1^{(i)}\]$  
$\rbrace$  
Although gradient descent is generally susceptible(受影响的) to local minimum, the optimization problem we propose in linear regression has only one global optimal solution and no other local optimal solutions, and the cost function is a convex(凸的) quadratic function. Therefore, gradient descent always converges to a global minimum (assuming the learning rate α is not too large).
## batch gradient descent(批处理梯度下降)
The algorithm used above is also called batch gradient descent, which means that each step of gradient descent involves all training instances. There are also other types of non-batch gradient descent that only focus on a small subset of the training set at a time.  
In advanced linear algebra(代数学), there is a numerical solution for calculating the minimum value of the cost function $J$. It doesn't require an iterative algorithm such as gradient descent to solve the minimum value of the cost function $J$. This is another method called normal equations(正规方程) method. In fact, in the case of a large amount of data, the gradient descent method is more suitable than the normal equation.
