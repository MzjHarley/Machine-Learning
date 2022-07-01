# Regularization
## The Problem of Overfitting
+ underfitting (inaccurate predictions)  
+ just right  
+ overfitting  (poor generalization ability)  
eg:Linear Regression    
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Regulation/1.png)  
eg:Classification Regression    
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Regulation/2.png)  
  
So how to solve the problem of overfitting?  
(1)reduce the number of features:  
+ Manually select which features to keep.  
+ Use a model selection algorithm.  
  
(2)regularization:  
+ Keep all the features, but reduce the magnitude of parameters.  
+ Regularization works well when we have a lot of slightly useful features.
## Cost Function  
If the linear regression is overfitting, the curve equation is as follows:  
$\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4$  
If you want to eliminate the influence of high power terms, you can modify the cost function , and set some penalties(乘法) on some parameters to reduce the influence of these parameters to a certain extent:  
$\underset {\theta}{min}\frac{1}{2m}\sum\limits_{i=1}\limits^{m}(h_\theta(x^{(i)})-y^{(i)})^2+1000\theta_3^2+1000\theta_4^2$  
To make the cost function tend to 0, the values ​​of$ ~ θ_3$ and $θ_4$ need to be reduced. Because the quadratic term is ≥ 0, the cost function is the smallest when they are 0, which reduces their influence on the hypothesis function, thereby reducing overfitting. This is the idea of regularization.  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Regulation/3.png)   
In actual use, because you don't know which parameters should be punished. So add a coefficient(系数) $λ$ to all parameters:  
$\underset {\theta}{min}\frac{1}{2m}(\sum\limits_{i=1}\limits^{m}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum\limits_{j=1}\limits^{n}\theta_j^2)$   
$ ~ λ$ is called regularization parameter, and the item after $\lambda$ is called regularization term.
+ If $λ$ = 0 or very small, it will not work and still overfit.  
+ If $λ$ is chosen too large, all parameters are penalized(parameters is tend to 0). The final assumption is that hypothesis function may become $h(x) = θ_0$, resulting in underfitting.  
## Regularized Linear Regression
The cost function for regularized linear regression is:  
$J(\Theta)= \frac{1}{2m}(\sum\limits_{i=1}\limits^{m}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum\limits_{j=1}\limits^{n}\theta_j^2)$  
Since regularization doesn't involve $θ_0$, the gradient descent algorithm is as follows:  
$Repeat ~ \lbrace$  
$ ~ ~ ~ ~ ~ ~ \theta_0:=\theta_0-\alpha\frac{1}{m}\sum\limits_{i=1}\limits^{m}(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}$  
$ ~ ~ ~ ~ ~ ~ \theta_j:=\theta_j-\alpha[\frac{1}{m}\sum\limits_{i=1}\limits^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j] ~ ~ ~  ~  ~ ~ ~ j\in\lbrace 1,2,\cdots,n\rbrace$  
$\rbrace$  
Adjust the second formula of the above algorithm to get:  
$\theta_j:=(1-\alpha\frac{\lambda}{m})\theta_j-\alpha\frac{1}{m}\sum\limits_{i=1}\limits^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} ~ ~ ~  ~  ~ ~ ~ j\in\lbrace 1,2,\cdots,n\rbrace$   
The change of the gradient descent algorithm of regularized linear regression is that each time $θ$ is reduced by an additional value based on the update rule of the original algorithm.  
If the Normal Equation method is used, a $(n+1)×(n+1)$-dimensional square matrix L is introduced, and the regularization is as follows:   
$\Theta=(X^TX+\lambda L)^{-1}X^TY$  
$L=\begin{bmatrix}0 \\\ ~ &1\\\ ~ &  &1 \\\ ~ &  &  &\ddots\\\ ~ & & & &1\end{bmatrix}$  
Here I don't give details of derivation process.But I will give$ ~ J(\theta)$ and use $\frac{\partial J(\Theta)}{\partial \Theta}=0$ to get$ ~ \Theta$.

$J(\Theta)= \frac{1}{2m}[(X\Theta-Y)^T(X\Theta-Y)+\lambda \Theta_T L \Theta]$  
$ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =\frac{1}{2m}(\Theta^TX^TX\Theta-\Theta^TX^TY-Y^TX\Theta-Y^TY+\lambda \Theta_T L \Theta)$.  
$\Theta=\begin{bmatrix} \theta_0 \\\ \theta_1\\\ \vdots\\\ \theta_n\end{bmatrix},Y=\begin{bmatrix} y^{(1)} \\\ y^{(2)}  \\\ \vdots \\\ y^{(m)}  \end{bmatrix},X=\begin{bmatrix} x_0^{(1)} &x_1^{(1)} &x_2^{(1)} &\cdots &x_n^{(1)} \\\ x_0^{(2)} &x_1^{(2)} &x_2^{(2)} &\cdots &x_n^{(2)}\\\ \vdots &\vdots &\vdots &\ddots &\vdots\\\ x_0^{(m)} &x_1^{(m)} &x_2^{(m)} &\cdots &x_n^{(m)},\end{bmatrix},L=\begin{bmatrix}0 \\\ ~ &1\\\ ~ &  &1 \\\ ~ &  &  &\ddots\\\ ~ & & & &1\end{bmatrix}$    

$ ~ \frac{\partial J(\Theta)}{\partial \Theta}=\frac{1}{2m}(2X^TX\Theta-2X^TY+2\lambda L\Theta)=0$   
$X^TX\Theta-X^TY+\lambda L\Theta=0$  
$(X^TX+\lambda L)\Theta-X^TY=0$  
$(X^TX+\lambda L)\Theta=X^TY$  
$(X^TX+\lambda L)^{-1}(X^TX+\lambda L)\Theta=(X^TX+\lambda L)^{-1}X^TY$   
$\Theta=(X^TX+\lambda L)^{-1}X^TY$  
## Regularized Logistic Regression
The cost function for logistic regression is:  
$J(\Theta) = -\frac{1}{m} \sum\limits_{i=1}\limits^{m} [y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$  
After adding the regular term:  
$J(\Theta) = -\frac{1}{m} \sum\limits_{i=1}\limits^{m} [y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum\limits_{j=1}\limits^{n}\theta_j^2$  

Since regularization doesn't involve $θ_0$, the gradient descent algorithm is as follows:  
$Repeat ~ \lbrace$  
$ ~ ~ ~ ~ ~ ~ \theta_0:=\theta_0-\alpha\frac{1}{m}\sum\limits_{i=1}\limits^{m}(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}$  
$ ~ ~ ~ ~ ~ ~ \theta_j:=\theta_j-\alpha[\frac{1}{m}\sum\limits_{i=1}\limits^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j] ~ ~ ~  ~  ~ ~ ~ j\in\lbrace 1,2,\cdots,n\rbrace$  
$\rbrace$   
It is still possible to use the $scipy.optimize.fmin\underline{~}tnc()$ function to solve the cost function minimization parameters , but we implement the regularization in the cost function.  
Effect (blue line is before regularization, pink line is after regularization):  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Regulation/4.png)  
