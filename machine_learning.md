# Machine Learning
The list of questions are from: https://github.com/alexeygrigorev/data-science-interviews/blob/master/theory.md and the original source is this blog post: https://medium.com/data-science-insider/160-data-science-interview-questions-14dbd8bf0a08


> What is supervised machine learning?

We have a machine learning task and the dataset contains desired outputs. If it is a classification problem then we have the 'correct' labels or if it is a regression problem we have the 'correct' values.

> What is regression? What is linear regression?

Regression is a supervised machine learning task. The difference between classification and regression is that in the former we try to predict a categorical variable and in the latter a continuous variable. Linear regression is solving a regression problem with a linear model which is predicting the outcome using a linear function of explanatory variables.

> (revisit) What are the different types of regression models?

Linear, logistics (for binary values), ridge&lasso (for regularization), quantile (median response instead of mean)

> (revisit) What are the main assumptions of linear regression?

The variance of the residuals is constant and the residuals are uncorrelated. No measurement error.

> What’s the normal distribution? Why do we care about it?

It is a bell shaped probabiltiy distribution and it is characterized by two parameters the mean ($\mu$) and the variance ($\sigma^2$).
The importance of this distribution is mainly due to the *Central Limit Theorem*. 
This theorem states that the mean of samples of **any** random variable converges to a normal distribution as the number of samples increases.

> (revisit) How do we check if a variable follows the normal distribution?

Graphical methods: histogram, qq-plot. Frequentist tests: Kolmogorov-Smirnov, Shapiro-Wilk, Pearson's chi-squared, Bayesian tests: KL-divergence

> (revisit) What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices?

The normal distribution may not be appropriate model to use: non-constant varience and fat tails. 

### Solving Linear Regression
> What methods for solving linear regression do you know?

There are two main methods to solve linear regression: (1) least square method / normal equations, (2) gradient descent. 

Normal equations can be derived from gemoetrical arguments (orthoganal projection) or from maximum likelihood estimation. For the simple case ($y = \beta x + \alpha + \epsilon$) the formula is $\hat{\beta} = \frac{(x-\bar{x})(y-\bar{y})}{(x-\bar{x})^2}$, $\hat{\alpha} = \bar{y} - \hat{\beta} \bar{x} $. SVD???

In the gradient descent method, we iteratively calculate the parameters with the following formula: $\theta_{new} = \theta_{old} - \eta \nabla_{\theta} J(\theta_{old})$. Gradient descent will converge to a (local) minimum but if loss function is convex then it is not an issue since there is a single global minimum point. The downside is that we need to choose the step size $\eta$ which determines the convergence speed (if we choose a big step size value then the estimated parameter values would oscilate and never converge, reach to the optimum). We can also solve other loss functions, like for example we can add regularization term to the loss function. Another advantage is that we can learn the parameters incrementally instead of full batch. In many applications we could regularly receive new data, so we can just use this new data to update the parameters instead of full calculation. *Can we do this with normal equations as well?*

> What is SGD  —  stochastic gradient descent? What’s the difference with the usual gradient descent?

It is an approximation for the gradient descent method. In the gradient descent there is a summation term, we sum the gradients. Usually it is not a problem but for high-dimensional dataset it could be computationally expensive to do all the summation in every iteration. In stochastic gradient descent, the true gradient of is approximated by a gradient at a single sample.

### Evaluation
> (revisit) Which metrics for evaluating regression models do you know?

We can define the residual as $y - \hat{y}$ and calculate them at each point. There are different ways to turn these residuals into a error metric, we would like to assess the *deviation* from the true value and both overforecasting and underforecasting should be penalized. We have different options, we can take the absolute value or take the square. (What is the difference?)

Finally we can turn all these error numbers calculated at each point to a single value. Here again we have option. The simplest option is to take the simple mean: mean absolute error, mean squared error. We can also calculate a weighted version: sum all the absolute errors and divide by the actuals. In this case the contribution of higher errors will be higher (compared to the simple mean, why?)

In the r-squared metric, we compute the sum of squared errors of *the* benchmark (which is the straight line) and compare agaisnt the models squared error to assess how good the model is fitting to the data. 

> (revisit) What is the bias-variance trade-off?

Let's say we have a dataset on hand, with a flexible model model we can get a good fit (low bias) on that fixed dataset. But now consider all possible datasets that we may encounter, with a flexible model we will estimate completely different parameters (high variance) for each dataset. *(Why this is a problem?)*

> What is overfitting?

We can always use a complex (flexible) model to have a better fit in the training set. But then the error metrics could be worst in the unseen data.

> How to validate your models?

The objective of a machine learning task is to make predictions (for unseen data). We would like quantify the accuracy of the prediction of the model on *unseen* data. This is important because we would like to know how much we can rely on this model to achieve our business objectives.

> (revisit) Why do we need to split our data into three parts: train, validation, and test?

We have a machine learning task and some data. Why don't we use all the data that we have to train the model? We would like to *validate* our model by simulating the real use case of the model: predicting unseen data. So we spare some part of the dataset as test to validate our model. We use the rest of the dataset to train the model we *learn* from the training dataset. 

For some models may have some hyperparameters to tune (e.g. k-neighours algorithm, $k$ is the number of neihgbours to look). We can dedicate a part of the training to tune the hyperparamters. Why we should not use the same dataset to tune the parameters and train the model? *"your evaluation results may not accurately reflect the model's performance on unseen data"* why? Fine-tuning on the same dataset increases the risk of overfitting, where the model learns to memorize the training data rather than capturing the underlying patterns. This can lead to poor performance on new data and biased evaluation results. 

> (revisit) Can you explain how cross-validation works?

We use different portions of the data to test and train a model on different iterations. 

> What is K-fold cross-validation?

The dataset is partitioned into K equal sized "folds". We select one of the folds as test set and we use the rest as training set and assess the accuracy. We repeat this process and each iteration we select a different "fold" as testing data.

> How do we choose K in K-fold cross-validation? What’s your favorite K?

Higher the K we will have a better accuracy estimation, the extreme is called leave-one-cross-validation (where K equals the number of samples in the dataset). On the other increases K also increases the need for computational resources. We need to find a balance between resources and the quality of the accuracy estimation.

### Feature Selection
> What is feature selection? Why do we need it?

> Is feature selection important for linear models?

> Which feature selection techniques do you know?

> Can we use L1 regularization for feature selection?

> Can we use L2 regularization for feature selection?

### Regularization
> What is regularization? Why do we need it?

> Which regularization techniques do you know?

> What kind of regularization techniques are applicable to linear models?

> How does L2 regularization look like in a linear model?

> How do we select the right regularization parameters?

> What’s the effect of L2 regularization on the weights of a linear model?

> How L1 regularization looks like in a linear model?

> What’s the difference between L2 and L1 regularization?

> Can we have both L1 and L2 regularization components in a linear model?

> When do we need to perform feature normalization for linear models? When it’s okay not to do it?

### Multi-collinearity
> What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y?

> What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise?

### Interpretation
> What’s the interpretation of the bias term in linear models?

> How do we interpret weights in linear models?

> If a weight for one variable is higher than for another  —  can we say that this variable is more important?

### Classification
> What is classification? Which models would you use to solve a classification problem?

> What is logistic regression? When do we need to use it?

> Is logistic regression a linear model? Why?

> What is sigmoid? What does it do?

> How do we evaluate classification models?

> What do we do with categorical variables?

> Why do we need one-hot encoding?

> What is "curse of dimensionality"?

### Clustering
> What is unsupervised learning?

> What is clustering? When do we need it?

> Do you know how K-means works?

> How to select K for K-means?

> What are the other clustering algorithms do you know?

### Dimensionality Reduction
> What is the curse of dimensionality? Why do we care about it?

> Do you know any dimensionality reduction techniques?

> What’s singular value decomposition? How is it typically used for machine learning?

### Time Series
> What is a time series?

> How is time series different from the usual regression problem?

> Which models do you know for solving time series problems?

> If there’s a trend in our series, how we can remove it? And why would we want to do it?

> You have a series with only one variable “y” measured at time t. How do predict “y” at time t+1? Which approaches would you use?

> You have a series with a variable “y” and a set of features. How do you predict “y” at t+1? Which approaches would you use?

> What are the problems with using trees for solving time series problems?
