# Logistic RegressionRegression
Although logistic regression has a regression in its name, it is not a regression algorithm.It's a very powerful, perhaps even the most widely used classification algorithm in the world.  
Feature scaling also works for logistic regression.
## Classification
Binary classification problem is defined as follows:  
$Classification: y = 0 ~ or ~ 1$   
$Linear ~ Regression:h_\theta(x) >1 ~ or ~ <0$  
$Logistic ~ Regression: 0 \leq h_\theta(x) \leq1$
## Hypothesis Representation
Introduce a new model: logistic regression. The output variable range is always between 0 and 1. The assumptions of the logistic regression model are as follows:  
$h_\theta(x)=g(x\Theta)$   
$z=x\Theta$  
$g(z)=\frac{1}{1+e^{-z}}$  
$\Theta=\begin{bmatrix}\theta_0\\\ \theta_1\\\ \vdots\\\ \Theta_n\end{bmatrix},x=\begin{bmatrix} x_0^{(i)},x_1^{(i)},\cdots,x_n^{(i)}\end{bmatrix}$   
$g(z)$ Represents Logistic Function, also called Sigmoid Function,its curve is as follows:  
  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Logistic%20Regression/1.png) 

$h_\theta(x)=P\lbrace y=1|x;\Theta \rbrace=1-P\lbrace y=0|x;\Theta \rbrace $
## Decision Boundary
The decision boundary is the dividing line that separates the area where$~ y = 0$ and $y = 1$.  
According to the sigmoid curve:  
when$~ g(x\Theta) \geq 0.5$,that is $x\Theta \geq 0,y=1.$  
when$~ g(x\Theta) < 0.5$,that is $x\Theta < 0,y=0.$  
So the curve that$~ x\Theta=0$ represents is the $Decision ~ Boundary$.   
Linear decision boundary:  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Logistic%20Regression/2.png)  
Non-linear Decision Boundary:  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Logistic%20Regression/3.png)  
## Cost Function
If we continue to use the cost function in linear regression, $J(\Theta)$ won't be a convex function, resulting in many local optimal solutions.  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Logistic%20Regression/4.png)    
To fit the parameters $\Theta$ of the logistic regression model, the cost function is as follows:  
$J(\Theta) = \frac{1}{m} \sum\limits_{i=1}\limits^{m} Loss(h_\theta(x^{(i)},y^{(i)}))$  
$Loss(h_\theta(x^{(i)},y^{(i)}))=\begin{cases}-log(h_\theta(x^{(i)}))  &,y^{(i)}=1\\\ -log(1-h_\theta(x^{(i)}))&,y^{(i)}=0\end{cases}$  
  
When$~ y^{(i)} = 1$, the corresponding curve of $Loss(h_\theta(x^{(i)},y^{(i)}))$ is as follows:  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Logistic%20Regression/5.png)  
When$~ y^{(i)} = 0$, the corresponding curve of $Loss(h_\theta(x^{(i)},y^{(i)}))$ is as follows:  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Logistic%20Regression/6.png)  

The two curves all have the following rules:  
$Loss(h_\theta(x^{(i)},y^{(i)}))=\begin{cases}0  &,h_\Theta(x^{(i)})=y^{(i)}\\\ \rightarrow \infty&,h_\Theta(x^{(i)})\neq y^{(i)}\end{cases}$
## Simplified Cost Function
$Loss(h_\theta(x^{(i)},y^{(i)}))=\begin{cases}-log(h_\theta(x^{(i)}))  &,y^{(i)}=1\\\ -log(1-h_\theta(x^{(i)}))&,y^{(i)}=0\end{cases}$  
Simplify the above two formulas into the following formula:  
$Loss(h_\theta(x^{(i)},y^{(i)}))=-y^{(i)}log(h_\theta(x^{(i)}))-(1-y^{(i)})log(1-h_\theta(x^{(i)}))$  
So the complete cost function is as follows:  
$J(\Theta) = -\frac{1}{m} \sum\limits_{i=1}\limits^{m} [y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$
###  A vector implementation of J(θ)
$J(\Theta) = -\frac{1}{m} [Y^Tlog(g(X\Theta))+(1-Y)^Tlog(1-g(X\Theta))]$  
$\Theta=\begin{bmatrix}\theta_0\\\ \theta_1\\\ \vdots\\\ \Theta_n\end{bmatrix},Y=\begin{bmatrix} y^{(1)}\\\ y^{(2)}\\\ \vdots\\\ y^{(m)}\end{bmatrix},X=\begin{bmatrix} x_0^{(1)} &x_1^{(1)} &x_2^{(1)} &\cdots &x_n^{(1)} \\\ x_0^{(2)} &x_1^{(2)} &x_2^{(2)} &\cdots &x_n^{(2)}\\\ \vdots &\vdots &\vdots &\ddots &\vdots\\\ x_0^{(m)} &x_1^{(m)} &x_2^{(m)} &\cdots &x_n^{(m)}\end{bmatrix}$  
## Gradient Descent
**gradient descent algorithm:**  
$repeat ~ until ~ convergence$  
$\lbrace$  
$~ ~ ~ ~ θ_j:=θ_j-\alpha\frac{\partial}{\partial{θ_j}}J(θ)$  
$~ ~ ~ ~ j=0,\cdots,n$  
$\rbrace$   
  
$\frac{\partial}{\partial{θ_j}}J(θ)=\frac{\partial}{\partial{θ_j}}\frac{-1}{m} \sum\limits_{i=1}\limits^{m} [y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}\frac{\partial}{\partial{θ_j}}[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[\frac{\partial}{\partial{θ_j}}y^{(i)}log(h_\theta(x^{(i)}))+\frac{\partial}{\partial{θ_j}}(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[y^{(i)}\frac{\partial}{\partial{θ_j}}log(h_\theta(x^{(i)}))+(1-y^{(i)})\frac{\partial}{\partial{θ_j}}log(1-h_\theta(x^{(i)}))]$   
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[y^{(i)}\frac{\frac{\partial}{\partial{θ_j}}h_\theta(x^{(i)})}{h_\theta(x^{(i)})}+(1-y^{(i)})\frac{\frac{\partial}{\partial{θ_j}}(1-h_\theta(x^{(i)}))}{1-h_\theta(x^{(i)})}]$   
  
Now we need to solve $\frac{\partial}{\partial{θ_j}}h_\theta(x^{(i)}),\frac{\partial}{\partial{θ_j}}(1-h_\theta(x^{(i)}))$   
$h_\theta(x^{(i)})=g(x^{(i)}\Theta)=\frac{1}{1+e^{-x^{(i)}\Theta}}$  
  
$\frac{\partial}{\partial{θ_j}}h_\theta(x^{(i)})=\frac{\partial g(x^{(i)}\Theta)}{\partial{(x^{(i)}\Theta)}} \cdot\frac{\partial{(x^{(i)}\Theta)}}{\partial \theta_j}$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =\frac{e^{-x^{(i)}\Theta}}{(1+e^{-x^{(i)}\Theta})^2}\cdot x_j^{(i)}$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =[\frac{1}{1+e^{-x^{(i)}\Theta}} \cdot\frac{e^{-x^{(i)}\Theta}}{1+e^{-x^{(i)}\Theta}}]\cdot x_j^{(i)}$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =[\frac{1}{1+e^{-x^{(i)}\Theta}} \cdot(1-\frac{1}{1+e^{-x^{(i)}\Theta}})]\cdot x_j^{(i)}$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =[g(x^{(i)}\Theta) \cdot(1-g(x^{(i)}\Theta))]\cdot x_j^{(i)}$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =h_\theta(x^{(i)}) \cdot (1-h_\theta(x^{(i)}))\cdot x_j^{(i)}$  
  
The same goes for $\frac{\partial}{\partial{θ_j}}(1-h_\theta(x^{(i)}))$.  
$\frac{\partial}{\partial{θ_j}}(1-h_\theta(x^{(i)}))=-h_\theta(x^{(i)}) \cdot (1-h_\theta(x^{(i)}))\cdot x_j^{(i)}$   

So $\frac{\partial}{\partial{θ_j}}J(θ)=-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[y^{(i)}\frac{\frac{\partial}{\partial{θ_j}}h_\theta(x^{(i)})}{h_\theta(x^{(i)})}+(1-y^{(i)})\frac{\frac{\partial}{\partial{θ_j}}(1-h_\theta(x^{(i)}))}{1-h_\theta(x^{(i)})}]$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[y^{(i)}\frac{h_\theta(x^{(i)}) \cdot (1-h_\theta(x^{(i)}))\cdot x_j^{(i)}}{h_\theta(x^{(i)})}+(1-y^{(i)})\frac{-h_\theta(x^{(i)}) \cdot (1-h_\theta(x^{(i)}))\cdot x_j^{(i)}}{1-h_\theta(x^{(i)})}]$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[y^{(i)}\cdot (1-h_\theta(x^{(i)}))\cdot x_j^{(i)}+(1-y^{(i)})(-h_\theta(x^{(i)})) \cdot x_j^{(i)}]$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[y^{(i)}\cdot (1-h_\theta(x^{(i)}))\cdot x_j^{(i)}-(1-y^{(i)})h_\theta(x^{(i)}) \cdot x_j^{(i)}]$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[(y^{(i)}-y^{(i)}h_\theta(x^{(i)}))\cdot x_j^{(i)}-(h_\theta(x^{(i)})-y^{(i)}h_\theta(x^{(i)})) \cdot x_j^{(i)}]$  
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[y^{(i)}-y^{(i)}h_\theta(x^{(i)})-h_\theta(x^{(i)})+y^{(i)}h_\theta(x^{(i)})] \cdot x_j^{(i)}$   
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =-\frac{1}{m} \sum\limits_{i=1}\limits^{m}[y^{(i)}-h_\theta(x^{(i)})] \cdot x_j^{(i)}$   
$~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ =\frac{1}{m} \sum\limits_{i=1}\limits^{m}[h_\theta(x^{(i)})-y^{(i)}] \cdot x_j^{(i)}$   

**So gradient descent algorithm:**  
$repeat ~ until ~ convergence$  
$\lbrace$  
$~ ~ ~ ~ θ_j:=θ_j-\frac{\alpha}{m} \sum\limits_{i=1}\limits^{m}[h_\theta(x^{(i)})-y^{(i)}] \cdot x_j^{(i)}$  
$~ ~ ~ ~ j=0,\cdots,n$  
$\rbrace$   
The gradient descent algorithm above looks the same as linear regression, but is actually quite different. Because$~ h(x)$ was a linear function before,but here $h(x)$ in logistic regression is defined as follows:  
$h_\theta(x)=\frac{1}{1+e^{-x ~ \Theta}}$  
### A vector implementation of Gradient Descent
$\Theta:=\Theta-\frac{\alpha}{m} X^T(g(X\Theta)-Y)$  
$\Theta=\begin{bmatrix}\theta_0\\\ \theta_1\\\ \vdots\\\ \Theta_n\end{bmatrix},Y=\begin{bmatrix} y^{(1)}\\\ y^{(2)}\\\ \vdots\\\ y^{(m)}\end{bmatrix},X=\begin{bmatrix} x_0^{(1)} &x_1^{(1)} &x_2^{(1)} &\cdots &x_n^{(1)} \\\ x_0^{(2)} &x_1^{(2)} &x_2^{(2)} &\cdots &x_n^{(2)}\\\ \vdots &\vdots &\vdots &\ddots &\vdots\\\ x_0^{(m)} &x_1^{(m)} &x_2^{(m)} &\cdots &x_n^{(m)}\end{bmatrix}$  
## Advanced Optimization
In addition to gradient descent algorithms, there are some algorithms that are often used to minimize the cost function. These algorithms are more complex and superior, often don't require manual selection of learning rates, and are faster than gradient descent algorithms.  
These are: Conjugate Gradient(共轭梯度), Local Optimization (Broyden fletcher goldfarb shann, BFGS) and Limited Memory Local Optimization (LBFGS).  
These algorithms have an intelligent inner loop, called a line search algorithm, that automatically tries different learning rates. Just provide these algorithms with a way to compute the derivative term and the cost function,then it will return the result.Suitable for large machine learning problems.  
They are too complex and should not be implemented by ourselves, but instead call **Python** methods. For example an unconstrained minimum function [fmin_tnc(func, x0, fprime=None, args=(),* args)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_tnc.html). It uses one of many advanced optimization algorithms, like an enhanced version of gradient descent, to automatically choose the learning rate and find the best value for $\Theta$.  
>$scipy.optimize.fmin\underline{ ~ }tnc(func=CostFunction, x0=Theta, fprime=GetPartial, args=(X,Y))$  
>"""  
>func：function_name(return value)  
>x0：array  
>fprime:function_name(return array)  
>args:tuple(array)  
>"""
## Multiclass Classification:One-vs-all
In a multi-classification problem, $y = 0,1,\cdots,n$.  
The method how to classify is as follows:  
(1) Split into $n+1$ binary classification problems.  
(2) For each Classification, predict$~ h(x)$ value which represents the likelihood that $y$ is of this type.  
(3) The final result is the most likely type.  
![contents](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/Logistic%20Regression/7.png)  
The method is Mathematically expressed as:  
$Prediction=\underset {i=0,\cdots,n}{max} ~ \Big(h_\theta^{(i)}(x)=P(y=i|x;\Theta)\Big)$
