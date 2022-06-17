# Machine Learning definition
A computer program is said to learn from experience E with respect to some task T and some performance measure(性能度量)P,if its performance on T,as measured by P,improves with experience E.

-------------------------------------------------------------
# Machine learning algorithms
## The main two types of Machine learning algorithms
+ Supervised Learning: The dataset used for learning is labeled.  
+ Unsupervised Learning：The dataset used for learning doesn't have any labels or has the same labels。Known datasets, not how to process them, and no telling what each data point is.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/ML%20Introduction%20and%20Basic%20Concepts/1.png)![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/ML%20Introduction%20and%20Basic%20Concepts/8.png)    
(the right example is unsupervised learning,which divides the data into two data collections. Also it's called clustering algorithm.)
### Supervised learning
The term supervised learning refers to the fact that we give the algorithm a data set in which the "right answer" are given, the task of algorithm is to just produce more of the right answers.  
+ regression problem(回归问题):we're trying to predict a continuous valued output.  
  predict house prices based on historical house price data.You can use a straight line (pink) or  a quadratic curve to fit (blue).  
  ![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/ML%20Introduction%20and%20Basic%20Concepts/2.png)
+ classification problem(分类问题):we're trying to predict a discrete valued output.  
  ![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/ML%20Introduction%20and%20Basic%20Concepts/3.png)  
  The following image is an example of making predictions based on two features (two dimensions), with other possible dimensions on the right (dimensions may be infinitely numerous)  
  ![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/ML%20Introduction%20and%20Basic%20Concepts/4.png)
### Unsupervised learning
The dataset used for learning doesn't have any labels or has the same labels.  
Known datasets, don't know how to process them, and no telling what each data point is.  
Given the dataset,a unsupervised learning algorithm will find some structure in the data to decide that the data lives in different clusters,then break these data into different separate clusters.  
####  Clustering algorithm in real-life applications
+ example 1:Every day, Google News divides the crawling URLs into news features.  
+ example 2:Gene information grouping.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/ML%20Introduction%20and%20Basic%20Concepts/9.png)  
+ example 3:Organize a large cluster of computers/Analysis of social networks/Market segmentation/Astronomical data analysis
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/ML%20Introduction%20and%20Basic%20Concepts/10.png)  
+ example 4:Cocktail party issues that take apart multiple audio sources that are mixed together.  
![content](https://github.com/MzjHarley/Machine-Learning/blob/main/IMG/ML%20Introduction%20and%20Basic%20Concepts/11.png) 
## others
+ Reinforcement learning(强化学习) 
+ recommender systems(推荐系统)
