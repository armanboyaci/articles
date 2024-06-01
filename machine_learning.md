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
> Which metrics for evaluating regression models do you know?

> What is the bias-variance trade-off?

> What is overfitting?

> How to validate your models?

> Why do we need to split our data into three parts: train, validation, and test?

> Can you explain how cross-validation works?

> What is K-fold cross-validation?

> How do we choose K in K-fold cross-validation? What’s your favorite K?

### Classification
> What is classification? Which models would you use to solve a classification problem?

> What is logistic regression? When do we need to use it?

> Is logistic regression a linear model? Why?

> What is sigmoid? What does it do?

> How do we evaluate classification models?

> What do we do with categorical variables?

> Why do we need one-hot encoding?

> What is "curse of dimensionality"?

### Multi-collinearity
> What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y?

> What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise?

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

### Interpretation
> What’s the interpretation of the bias term in linear models?

> How do we interpret weights in linear models?

> If a weight for one variable is higher than for another  —  can we say that this variable is more important?

### Feature Selection
> What is feature selection? Why do we need it?

> Is feature selection important for linear models?

> Which feature selection techniques do you know?

> Can we use L1 regularization for feature selection?

> Can we use L2 regularization for feature selection?

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
