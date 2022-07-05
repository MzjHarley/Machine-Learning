# Neural Networks
## Non-linear Classification
The disadvantages of linear regression and logical regression: When there are too many data features input, the calculation load is large.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/1.png)  
In computer vision, a picture is represented by a matrix of pixels. Suppose an image is 50×50px and its feature count is 2500 (grayscale, 7500 if it’s an RGB image). If the combination of two features will reach a million level (choose two combinations from 2500, 2500 * 2499 / 2 ≈ 3 * 10^6), logistic regression will not be applicable.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/2.png)  
## Model Representation I
To mimic(模仿) the way the brain works, neural networks can be similarly divided into: input data features, intermediate data processing layers, and final output.  
Neural network models are built on many neurons, each of which is a learning model. These neurons (also called activation units) take some features and provide an output based on their own model.  
The figure below is an example of a neuron that uses a logistic regression model as its own learning model. The parameter $\Theta$ can also be called weights.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/3.png)  
The neural network model is organized by many logical units according to different levels, and each output variable is the input variable of the next layer.  
Logic unit: input vector$ ~ X$ (input layer), intermediate layer $a^{(j)}$ (hidden layer), output layer $h(x)$.  
A bias unit can be added to the input of each layer, usually with a value of 1.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/4.png)  
|parameter|meaning|
|---|---|
|$j$ |represents the number of layers.|
|$i$ |represents the number of elements from top to bottom.|
|$a_i^{(j)}$ |the$ ~ i^{th}$ activation unit of the $j^{th}$ layer.|
|$θ^{(j)}$ |the weight matrix that maps the$ ~ j^{th}$ layer to the $(j+1)^{th}$ layer.| 
  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/5.png)  
## Model Representation II
We call this left-to-right (input$ ~ \rightarrow$ activation $\rightarrow$ output) algorithm forward propagation.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/22.png)  
If the first few layers are covered, the neural network is like logistic regression, except that we change the input vector $x_1\sim x_3$ in logistic regression into $a_1^{(2)}\sim a_3^{(2)}$ of the middle layer, that is  
  
$h_\theta(x)=g(\theta_0^{(2)}a_0^{(2)}+\theta_1^{(2)}a_1^{(2)}+\theta_2^{(2)}a_2^{(2)}+\theta_3^{(2)}a_3^{(2)})$   
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/23.png)  
## Multi-class classification
One-Vs-All method is a generalization of the two-class classification problem to multi-class classification.   
Multi-classification with neural networks:  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/20.png)  
## Cost Function
Suppose there are$ ~ m$ training samples, each containing a set of input $x$ and a set of output signals $y$.  
$L$ represents the number of layers of the neural network.  
$S_l$ represents the number of neurons in each layer.  
$S_L$ represents the number of processing units in the last layer.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/21.png)  
The Cost Function previously defined for logistic regression is as follows (the first half represents the distance between the hypothesis and the true value, and the second half is the bias term for regularization of parameters):  
$J(\Theta) = -\frac{1}{m} \sum\limits_{i=1}\limits^{m} [y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum\limits_{j=1}\limits^{n}\theta_j^2$   
The cost function of the neural network is the same:  
$J(\Theta) = -\frac{1}{m} \sum\limits_{i=1}\limits^{m} \sum\limits_{k=1}\limits^{S_L}[y_k^{(i)}log({h_\theta(x^{(i)})}_ k)+(1-y_k^{(i)})log(1-h_\theta(x^{(i)})_ k)]+\frac{\lambda}{2m}\sum\limits_{l=1}\limits^{L-1}\sum\limits_{j=1}\limits^{S_l}\sum\limits_{i=1}\limits^{S_{l+1}}(\theta_{ij}^{(l)})^2$  
$\Theta^{(l)}$'s size is $S_{l+1}\times (S_{l}+1),i\in\lbrace 1,2,\cdots,S_{l+1}\rbrace,j\in\lbrace 0,1,\cdots,S_{l}\rbrace$  
Here we don't normalize bias unit's parameter $\theta_{i0}$.    
## Back-propagation Algorithm
To minimize $J(Θ),\frac{\partial}{\partial \theta_{ij}^{(l)}} J(\theta)$ are required.  
A backpropagation algorithm is used: first calculate the error$ ~ \delta^{(L)}$ of the last layer, and then reverse (to the left) to find the error $\delta^{(l)}$ of each layer until the penultimate(倒数第二的) layer (the first layer is the input variable, which doesn't exist error).  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/18.png)  
For one training example$(x,y)$,the cost function is as follows:  
$J(\Theta)'=-\sum\limits_{k=1}\limits^{S_L}[y_k^{ ~ } ~ log(a^{(4)}_ k)+(1-y_k^{ ~ })log(1-a_k^{(4)})]$   
Here we don't need to normalize the $J(\theta)'$,because regularization is the last things to do,but here we just compute one training example's cost not all examples.  
For one training example$(x,y)$'s each output unit(Layer=4),the cost function is as follows:  
$J(\Theta)''=-[y_{k}^{ ~ } ~ log(a^{(4)}_ k)+(1-y_k^{ ~ })log(1-a_k^{(4)})],k\in\lbrace 1,\cdots,S_{L}\rbrace$  
  
$\delta_k^{(4)}=\frac{\partial J(\theta)''}{\partial z_k^{(4)}}=\frac{\partial  J(\theta)''}{\partial a_k^{(4)}}\bullet \frac{\partial a_k^{(4)}}{\partial z_k^{(4)}}=(\frac{-y_k^{ ~ }}{a_k^{(4)}}+\frac{1-y_k^{ ~ }}{1-a_k^{(4)}}) * a_k^{(4)}(1-a_k^{(4)})=\frac{a_k^{(4)}-y_k^{ ~ }}{a_k^{(4)}(1-a_k^{(4)})} * a_k^{(4)}(1-a_k^{(4)})=a_k^{(4)}-y_k^{ ~ },k\in\lbrace 1,\cdots,S_{L}\rbrace$  
  
So $\delta^{(4)}=a^{(4)}-y.$  
  
$\delta_j^{(3)}=\frac{\partial J(\theta)''}{\partial z_j^{(3)}}=\sum\limits_{k=1}\limits^{S_L}\frac{\partial  J(\theta)''}{\partial a_k^{(4)}}\bullet \frac{\partial a_k^{(4)}}{\partial z_k^{(4)}}\bullet \frac{\partial  z_k^{(4)}}{\partial a_j^{(3)}}\bullet \frac{\partial a_j^{(3)}}{\partial z_j^{(3)}}=\sum\limits_{k=1}\limits^{S_L}\frac{\partial J(\theta)''}{\partial z_k^{(4)}}\bullet \frac{\partial  z_k^{(4)}}{\partial a_j^{(3)}}\bullet \frac{\partial a_j^{(3)}}{\partial z_j^{(3)}}=\sum\limits_{k=1}\limits^{S_L}\delta_k^{(4)} * \Theta_{kj}^{(3)} * a_j^{(3)}(1-a_j^{(3)})=(\Theta_{:j}^{(3)})^{T}  \times\delta^{(4)} * a_j^{(3)}(1-a_j^{(3)}),j\in\lbrace 0,\cdots,S_{3}\rbrace$  
  
So $\delta^{(3)}=(\Theta^{(3)})^{T} \times\delta^{(4)} * a^{(3)} * (1-a^{(3)}).$   
  
$\delta_i^{(2)}=\frac{\partial J(\theta)''}{\partial z_i^{(2)}}=\sum\limits_{j=1}\limits^{S_3}\frac{\partial J(\theta)''}{\partial z_j^{(3)}}\bullet \frac{\partial  z_j^{(3)}}{\partial a_i^{(2)}}\bullet \frac{\partial a_i^{(2)}}{\partial z_i^{(2)}}=\sum\limits_{j=1}\limits^{S_3}\delta_j^{(3)} * \Theta_{ji}^{(2)} * a_i^{(2)}(1-a_i^{(2)})=(\Theta_{:i}^{(2)})^{T} \times\delta^{(3)} * a_i^{(2)}(1-a_i^{(2)}),i\in\lbrace 0,\cdots,S_{2}\rbrace$   
  
So $\delta^{(2)}=(\Theta^{(2)})^{T} \times\delta^{(3)} * a^{(2)} * (1-a^{(2)}).$   
  
With the error, start calculating $\frac{\partial}{\partial \Theta_{ij}^{(l)}}J(\theta)''$.   
$l$: current layer index.  
$j$: the index of activation unit of current layer.   
$i$: the index of activation unit of next layer.  
$\frac{\partial}{\partial \Theta_{ij}^{(l)}}J(\theta)''=\frac{\partial J(\theta)''}{\partial z_{i}^{(l+1)}}\bullet \frac{\partial z_{i}^{(l+1)}}{\partial \theta_{ij}^{(l)}}=\delta_i^{(l+1)} * a_j^{(l)}$  

So the back-propagation algorithm is as follows:  
$Training ~ set \lbrace (x^{(1)},y^{(1)}),cdots,(x^{(m)},y^{(m)})\rbrace$    
$Set ~ \Delta_{ij}^{(l)}=0 ~ for ~ all ~ i,j,l$  
$For ~ i = 1 ~ to ~ m:$  
$ ~ ~ ~ ~ Set ~ a^{(1)}=x^{(i)}$  
$ ~ ~ ~ ~ Perform ~ Forward Propagation ~ to ~ compute ~ a^{(l)} ~ for ~ l=2,3,\cdots,L$  
$ ~ ~ ~ ~ Compute ~ \delta^{(L)}=a^{(L)}-y^{(i)}$  
$ ~ ~ ~ ~ Compute ~ \delta^{(L-1)},\delta^{(L-2)},\cdots,\delta^{(2)}$  
$ ~ ~ ~ ~ \Delta_{ij}^{(l)}=\Delta_{ij}^{(l)}+\delta_i^{(l+1)} * a_j^{(l)}$  

Then calculate $\frac{\partial}{\partial \theta_{ij}^{(l)}} J(\theta)$, the formula is as follows:  
$\frac{\partial}{\partial \theta_{ij}^{(l)}} J(\theta)=D_{ij}^{(l)}=\begin{cases}\frac{1}{m}\Delta_{ij}^{(l)}+\frac{\lambda}{m}\theta_{ij}^{(l)} ~ ~ ~ &,j \neq0\\\  \frac{1}{m}\Delta_{ij}^{(l)} ~ ~ ~ &,j =0\end{cases} $
## Gradient Test
In order to verify whether the complex model is running properly or not, we use a method called Numerical gradient checking to verify whether the gradient is decreasing or not.  
For the following$ ~ J(θ)$ graph, take one point on the left and right of the$ ~ θ$ point:$ ~ (θ+ε), (θ-ε)$, then the derivative (gradient) of the point$ ~ θ$ is approximately equal to$ ~ \frac{J(Θ+ε)-J(θ-ε)}{2ε}$.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/15.png)  
Therefore, for each $θ$, its derivative can be approximated as:  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/16.png)  
Compare this approximation with the derivative of $J(θ)$ obtained at each step in the back-propagation algorithm. If the two results are close, the code is correct, otherwise it is wrong.  
## Random Initialization θ
Previously for logistic regression, we initialized all parameters $θ$ to 0.  
However, for neural networks, this method is not feasible: if the first layer parameter $θ$ is the same (whether it is 0 or not), it means that the value of all activation units in the second layer will be the same.  
Usually the initial parameters$ ~ \Theta$ are random values between positive and negative $ε$.  
## Put It Together
$\bullet$ Choose a neural network(usually, the more neurons in the hidden layer, the better.)  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/17.png)  
$\bullet$ Train the neural network:  
+ 1.Random initialization of parameters $\Theta$.  
+ 2.Calculate all $ℎ_\theta(x)$ using the forward propagation method.  
+ 3.Write the code to compute the cost function $J(\Theta)$.  
+ 4.Compute all partial derivatives using back-propagation.  
+ 5.Using numerical methods to test these partial derivatives.  
+ 6.Use an optimization algorithm to minimize the cost function.  

The intuitive representation of the neural network is as follows. Since $J(\Theta)$ is not a convex function, we can reach a local minimum.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/NeuralNetworks/19.png)  
