# Linear Regression with Multiple Variables
## Multiple Features
**Multivariate linear regression**  
Univariate regression models were discussed earlier. Now discussing the multivariate model, the features in the model are $(x_1,x_2,\cdots,x_n)$.  
photo1  
Introduce new annotations:
|Variable|meaning|
|---|---|
|$x_{j}^{(i)}$|value of feature$ ~ j$ in the $i^{th}$ training example|
|$x^{(i)}$|the input (features) of the $i^{th}$ training example|
|$m$|the number of training examples|
|$n$|the number of features|

The hypothesis$ ~ h$ supporting multivariate is expressed as follows:  
$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n$.  
There are$ ~ n+1$ parameters and $n$ variables in this formula.  
In order to simplify the formula,$x_0=1$ is introduced, and the formula is transformed into:  
$h_\theta(x)=\theta_0x_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n$  
At this time, the parameter$ ~ \Theta$ in the model is an$ ~ n+1$-dimensional vector, any training instance is also an $n+1$-dimensional vector, and the dimension of the feature matrix $X$ is $m\times(n+1)$.  
$\Theta=\begin{bmatrix} \theta_0 \\\ \theta_1 \\\ \theta_2 \\\ \vdots \\\ \theta_n \end{bmatrix},x^{(i)}=\begin{bmatrix} x_0^{i} &x_1^{i} &x_2^{i} &\cdots &x_n^{i}  \end{bmatrix},X=\begin{bmatrix} x_0^{1} &x_1^{1} &x_2^{1} &\cdots &x_n^{1} \\\ x_0^{2} &x_1^{2} &x_2^{2} &\cdots &x_n^{2}\\\ \vdots &\vdots &\vdots &\ddots &\vdots\\\ x_0^{m} &x_1^{m} &x_2^{m} &\cdots &x_n^{m}\end{bmatrix}$  
The hypothesis function $h$ can be simplified as follows:  
when inputing$ ~ x$,$ ~ h_\theta(x)= ~ x\Theta$.  
when inputing$ ~ X$,$ ~ h_\theta(X)= ~ X\Theta$.
## Gradient Descent for Multiple Variables
In linear regression with multiple variables, define the cost function $J(Θ)$ as follows:  
$J(θ_0,θ_1,\cdots,\theta_n)=\frac{1}{2m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})^2$  
$J(\Theta)=\frac{1}{2m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})^2$   
The multivariate linear regression model is as follows.  To simplify, we add$ ~ x_0 = 1$ , and the parameter$ ~ Θ$ is an$~ n+1$-dimensional vector. The algorithm updates each$ ~ Θ_j$ synchronously(同步地) $(j = 0 ~ to ~ n)$.  
|Model Element|Formula|
|---|---|
|$Hypothesis$|$h_\theta(x)=x\Theta=\theta_0x_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n$|
|$Parameter$|$\Theta=\begin{bmatrix} \theta_0 &\theta_1 &\theta_2 &\cdots &\theta_n \end{bmatrix}^T$|
|$Cost Function$|$J(\Theta)=\frac{1}{2m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})^2$|
|$Gradient Descent$|$repeat ~ until ~ convergence \lbrace θ_j:=θ_j-\alpha\frac{\partial}{\partial{θ_j}}J(\Theta) ~ ~ (simultaneously ~ update ~ for ~ every ~ j=0,\cdots,n) \rbrace$|

Comparing univariate gradient descent (left) and multivariate gradient descent (right). Since we introduced$ ~ x_0^{(i)}=1$,so the first two terms,$\theta_0$ and $\theta_1$ of multivariate gradient descent are the same as univariate gradient descent.  
photo2  
## Gradient Descent in Practice I - Feature Scaling
In the multi-dimensional feature problem, to help the gradient descent algorithm converge faster, the features need to have a similar scale, which requires us to perform feature scaling(特征缩放).  
Assuming two features, the value of house size is$ ~ 0-2000$, and the amount of rooms is $0-5$, the corresponding cost function contour map(left image) will be very flat (skewed elliptical shape), and gradient descent requires a lot of iterations to converge.  
But if we divide house size by 2000 and amount of room by 5,try to scale all features between $-1$ and $1$ as much as possible, then we will get a nearly circular contour map(right image),from which it can be seen that gradient descent requires a few iterations to converge.  
photo3  
The scale doesn‘t have to be -1 to 1, but the range cann’t be very large or very small.for example:  
photo3   
In addition to dividing all feature values by the maximum value of the feature, we can also use the following method for feature scaling.   
|Normalization|Fomula|
|---|---|
|Mean Normalization|$x_i:=\frac{x_i-\mu_i}{s_i}, ~ s_i=(x_i.max) ~ or ~ (x_i.max-x_i.min)$|
|Most Value Normalization|$x_i:=\frac{x_i-x_i.min}{x_i.max-x_i.min}$|
|Mean Variance Normalization|$x_i:=\frac{x_i-\mu_i}{\sigma_i}$|
## Gradient Descent in Practice II - Learning Rate
To ensure that the gradient descent algorithm works correctly, you can graph the relationship between iterations number and the cost function, and observe when the algorithm tends to converge (left).  
There are also some methods to automatically test whether to converge or not, such as using a threshold(阈值) $\epsilon$ (right). Because the size of the threshold is difficult to choose, the graph on the left is better.  
photo  
As the number of iterations increases, the cost function should tend to decrease. If it rises or rises and falls frequently, it means that$ ~ α$ is obtained too large, which may lead to failing to converge. If $α$ is taken too small, the algorithm will run slowly but still decreasing, and usually converges after many iterations.  
photo  
For learning rate$ ~ \alpha$,you can try to use the following values:  
$\cdots,0.001,0.003,0.01,0.03,\cdots,1,\cdots$
## Features and Polynomial Regression
### Create new feature
You don't have to use existing features, you can create new features, for example: area = length $\times$ width. Then the quadratic function becomes a univariate function.
photo
### Polynomial Regression  
Quadratic equation model: $h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2^{2}$  
Cubic equation model: $h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2^{2}+\theta_3x_3^{3}$   
Because in real life, as the housing area increases, the housing price cann't decrease, and the quadratic curve will firstly rise and then fall. Choose a cubic model, introduce additional variables to replace higher powers, and convert it to a linear regression model.  
photo  
In order to fit the curve better, you can also use the square root.  
photo
## Normal Equation
### Normal Equation  
The idea of normal equation: Assuming that the partial derivative of the cost function$ ~ J(Θ)$ is equal to 0, solve the equation to get the parameter $Θ$ that minimizes the cost function $J(Θ)$. That is, find the lowest point of the curve (the slope of the tangent is 0).  
In the simplest case, with only one dimension, the cost function is a quadratic curve:  
photo   
If there are $n$ features, then $Θ$ is $n+1$\-dimensional. For each term $J(Θ_j)$ of the cost function $J(Θ)$, let its partial derivative be 0. Solve the equation mathematically(从数学上) to get $Θ_j$ that minimizes the cost function $J(Θ_j)$.  
photo  
### Solution to Normal Equations  
$h_\theta(x)=x\Theta=\theta_0x_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n$  
$J(\Theta)=\frac{1}{2m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})^2$  
Assuming that the training set feature matrix is $X$ (including $x_0$ = 1), and the result is a vector $Y$, the solution $\Theta=\begin{bmatrix} \theta_0 &\theta_1 &\theta_2 &\cdots &\theta_n \end{bmatrix}^T$ can be obtained by the formula:  
$\Theta=(X^TX)^{-1}X^TY$  
$Y=\begin{bmatrix} y^{(1)} \\\ y^{(2)}  \\\ \vdots \\\ y^{(m)}  \end{bmatrix},X=\begin{bmatrix} x_0^{(1)} &x_1^{(1)} &x_2^{(1)} &\cdots &x_n^{(1)} \\\ x_0^{(2)} &x_1^{(2)} &x_2^{(2)} &\cdots &x_n^{(2)}\\\ \vdots &\vdots &\vdots &\ddots &\vdots\\\ x_0^{(m)} &x_1^{(m)} &x_2^{(m)} &\cdots &x_n^{(m)}\end{bmatrix}$    
for example:  
photo  
$\Theta=\begin{pmatrix}\begin{bmatrix} 1 &1 &1 &1 \\\ 2104 &1416 &1534 &852 \\\ 5 &3 &3 &2 \\\ 1 &2 &2 &1\\\ 45 &40 &30 &36\end{bmatrix}\times\begin{bmatrix} 1 &2104 &5 &1 &45 \\\ 1 &1416 &3 &2 &40 \\\ 1 &1534 &3 &2 &40\\\ 1 &852 &2 &1 &36\end{bmatrix}\end{pmatrix}^{-1}\times\begin{bmatrix} 1 &1 &1 &1 \\\ 2104 &1416 &1534 &852 \\\ 5 &3 &3 &2 \\\ 1 &2 &2 &1\\\ 45 &40 &30 &36\end{bmatrix}\times \begin{bmatrix} 460 \\\ 232 \\\ 315 \\\ 178\end{bmatrix}$  
In the normal equation method,feature scaling isn't required.
### The Derivation Process of Θ of Normal Equation Solving
$h_\theta(x)=x\Theta=\theta_0x_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n$  
$J(\Theta)=\frac{1}{2m}\sum\limits_{i=1}\limits^m(h_θ(x^{(i)})-y^{(i)})^2$   
$\Theta=\begin{bmatrix} \theta_0 \\\ \theta_1\\\ \vdots\\\ \theta_n\end{bmatrix},Y=\begin{bmatrix} y^{(1)} \\\ y^{(2)}  \\\ \vdots \\\ y^{(m)}  \end{bmatrix},X=\begin{bmatrix} x_0^{(1)} &x_1^{(1)} &x_2^{(1)} &\cdots &x_n^{(1)} \\\ x_0^{(2)} &x_1^{(2)} &x_2^{(2)} &\cdots &x_n^{(2)}\\\ \vdots &\vdots &\vdots &\ddots &\vdots\\\ x_0^{(m)} &x_1^{(m)} &x_2^{(m)} &\cdots &x_n^{(m)}\end{bmatrix}$    

so $J(\Theta)= \frac{1}{2m}(X\Theta-Y)^T(X\Theta-Y)$  
$ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =\frac{1}{2m}((X\Theta)^T-Y^T)(X\Theta-Y)$  
$ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =\frac{1}{2m}(\Theta^TX^T-Y^T)(X\Theta-Y)$  
$ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =\frac{1}{2m}(\Theta^TX^TX\Theta-\Theta^TX^TY-Y^TX\Theta-Y^TY)$.  
  
we need to let $\frac{\partial J(\Theta)}{\partial \Theta}=\frac{1}{2m}\frac{\partial (\Theta^TX^TX\Theta-\Theta^TX^TY-Y^TX\Theta-Y^TY)}{\partial \Theta}=\frac{1}{2m}(\frac{\partial (\Theta^TX^TX\Theta)}{\partial \Theta}-\frac{\partial (\Theta^TX^TY)}{\partial \Theta}-\frac{\partial (Y^TX\Theta)}{\partial \Theta}-\frac{\partial(Y^TY)}{\partial \Theta})=0$ to get$ ~ \Theta$.  
  
Now we need to solve $\frac{\partial (\Theta^TX^TX\Theta)}{\partial \Theta},\frac{\partial (\Theta^TX^TY)}{\partial \Theta},\frac{\partial (Y^TX\Theta)}{\partial \Theta},\frac{\partial(Y^TY)}{\partial \Theta}$.   
  
$\frac{\partial(Y^TY)}{\partial \Theta} ~ ~ ~ =0$  

$\frac{\partial (Y^TX\Theta)}{\partial \Theta} ~ =\begin{bmatrix} \frac{\partial (Y^TX\Theta)}{\partial \theta_0}\\\ \vdots\\\ \frac{\partial (Y^TX\Theta)}{\partial \theta_n}\end{bmatrix} ~ =\begin{bmatrix} x_0^{(1)}y^{(1)}+ \cdots+x_0^{(m)}y^{(m)}\\\ \vdots \\\ x_n^{(1)}y^{(1)}+ \cdots+x_n^{(m)}y^{(m)}\end{bmatrix}=X^TY$  

$\frac{\partial (\Theta^TX^TY)}{\partial \Theta}=\begin{bmatrix} \frac{\partial (\Theta^TX^TY)}{\partial \theta_0}\\\ \vdots\\\ \frac{\partial (\Theta^TX^TY)}{\partial \theta_n}\end{bmatrix}=\begin{bmatrix} x_0^{(1)}y^{(1)}+ \cdots+x_0^{(m)}y^{(m)}\\\ \vdots \\\ x_n^{(1)}y^{(1)}+ \cdots+x_n^{(m)}y^{(m)}\end{bmatrix}=X^TY$  

$\frac{\partial (\Theta^TX^TX\Theta)}{\partial \Theta}=2X^TX\Theta$,here we use Scalar(标量) to Vector Derivation Fomula：$\frac{\partial (x^TAx)}{\partial x}=2Ax$,and you can also compute it directly.  
  
Now we substitute the values of$ ~ \frac{\partial (\Theta^TX^TX\Theta)}{\partial \Theta},\frac{\partial (\Theta^TX^TY)}{\partial \Theta},\frac{\partial (Y^TX\Theta)}{\partial \Theta},\frac{\partial(Y^TY)}{\partial \Theta}$ into the normal equation$ ~ \frac{\partial J(\Theta)}{\partial \Theta}=0$.  
  
$\frac{\partial J(\Theta)}{\partial \Theta}=\frac{1}{2m}(\frac{\partial (\Theta^TX^TX\Theta)}{\partial \Theta}-\frac{\partial (\Theta^TX^TY)}{\partial \Theta}-\frac{\partial (Y^TX\Theta)}{\partial \Theta}-\frac{\partial(Y^TY)}{\partial \Theta})=\frac{1}{2m}(2X^TX\Theta-X^TY-X^TY-0)=\frac{1}{2m}(2X^TX\Theta-2X^TY)=\frac{1}{m}(X^TX\Theta-X^TY)=0$    
   
 Then we need to solve$ ~ \frac{1}{m}(X^TX\Theta-X^TY)=0$ to get $\Theta$.  
   
$\frac{1}{m}(X^TX\Theta-X^TY)=0$  
$ ~ ~ ~ ~ ~ ~ ~ X^TX\Theta-X^TY=0$  
$ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ X^TX\Theta=X^TY$  
$ ~ ~ ~ ~ (X^TX)^{-1}X^TX\Theta=(X^TX)^{-1}X^TY$  
$ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \Theta=(X^TX)^{-1}X^TY$
## Normal Equation Noninvertibility
