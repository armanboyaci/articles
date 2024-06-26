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

Usually a model with more features is more complex (flexible) and there is a risk of overfitting. We find the most relevant features and get rid of the rest; this lighter model can generalize better to unseen data, improving its predictive performance. In addition, the lighter model will be easier to debug, faster to train.

> Is feature selection important for linear models?

> Which feature selection techniques do you know?

1. Removing features with low variance
2. Univariate feature selection: some statistical test? (e.g. f-test, pearson's correlation ...)
3. Recursive feature elimination: the least important features are pruned from current set of features until the desired number of features to select is eventually reached.
4. Feature selection using SelectFromModel
5. Sequential Feature Selection (backward, forward)

> Can we use L1 regularization for feature selection?

> Can we use L2 regularization for feature selection?

### Regularization
> What is regularization? Why do we need it?

Regularization is about controling the complexity of the model to avoid overfitting. We trade a decrease in the fittness in the training for an increase in generalizability.

> Which regularization techniques do you know? What kind of regularization techniques are applicable to linear models? How does L2 regularization look like in a linear model?

The common trick that we use to add a penalizing term to the objective function that we try to minimize. In Lasso (resp. Ridge) the penalizing term is sum of absolute (resp. squared) of the coefficients.

> How do we select the right regularization parameters?

We use cross-validation.

> What’s the effect of L2 regularization on the weights of a linear model?

It is penalizing.

> How L1 regularization looks like in a linear model?

> What’s the difference between L2 and L1 regularization?

The L1 regularization solution is sparse. L2 regularization doesn’t perform feature selection, since weights are only reduced to values near 0 instead of 0. L1 regularization has built-in feature selection. L1 regularization is robust to outliers, L2 regularization is not. 
[Why is L2 preferred over L1 Regularization?](https://www.reddit.com/r/MachineLearning/comments/dgog2h/d_why_is_l2_preferred_over_l1_regularization/)

> Can we have both L1 and L2 regularization components in a linear model?

Elasticnet do that.

> When do we need to perform feature normalization for linear models? When it’s okay not to do it?

### Multi-collinearity
> (revisit) What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y?

It is called perfect multicollinearity and in this case the input matrix $X$ will contain linearly dependent columns corresponding to x and y. And so X is not invertable, we are not going to able to solve the "normal equations". *But what about lasso or ridge regession?*

> (revisit) What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise?

In this case we have nearly exact linear relationship. We don't need to drop any variable. Since including collinear variables does not reduce the predictive power or reliability of the model as a whole and does not reduce the accuracy of coefficient estimates.

### Interpretation
> (revisit) What’s the interpretation of the bias term in linear models?

The bias term (intercept) is the value of the output $y$ when all the input values $x$s are zero. Sometimes to have a better interpretation we can change the variables. Let's say we fitted a regression line for world record for mile run. Assume it is $y = 1007 - 0.393x$. $x=0$ means the year 0 which is meaningless. Instead we can rewrite the same formula as $y = 241 - 0.393(x-1950)$. Let $x'=x-1950$ then we have $y = 241 - 0.393x'$ which is more meaningful.

> (revisit) How do we interpret weights in linear models?

The safest interpretation of a regression is as a comparison. Let's say we have the following model: earnings = −26.0 + 0.6 * height + 10.6 * male + error. It might seem natural to report that the estimated effect of height is $600 and the estimated effect of sex is $10 600. But the correct interpretation is (1) the average difference in earnings, comparing two people of the same sex but one inch different in height, is $600, (b) two people with the same height but different sex, the man’s earnings will be, on average, $10 600 more than the woman’s.

> (revisit) How to interpret a log-log model?

Elasticity tells the percentage change in dependent variable with respect to 1% change in the independent variable while holding all other variables constant.

> (revisit) If a weight for one variable is higher than for another  —  can we say that this variable is more important?

(CHATGPT) Not necessarily. While the weight assigned to a variable in linear regression does indicate its influence on the predicted outcome, it doesn't always directly translate to importance.

Variables with higher weights have a stronger linear relationship with the outcome variable within the confines of the model. However, the importance of a variable can also depend on other factors such as the scale and range of the variables, multicollinearity (correlation between predictor variables), and the specific context of the problem being analyzed. 

Moreover, the interpretation of importance can vary based on the goals of the analysis. Sometimes, variables with lower weights might be crucial for the model's performance or for understanding certain nuances in the data, even if they don't have as strong a linear relationship with the outcome.

Therefore, while weight magnitude can provide insights, it's essential to interpret it alongside other factors and domain knowledge before concluding a variable's importance.

### Classification
> (revisit) What is classification? Which models would you use to solve a classification problem?

Classification is a supervised machine learning tasks. For each sample, we are given a class label and we would like to predict the class labal of a unknown class label. There are many methods to solve a classification problem: nearest neighbor, logistic regression, decision trees, neural networks, gaussian process.

> What is logistic regression? When do we need to use it?

It can be used to predict binary variables.

> How we can use a binary classifier in the case of multiple classes? 

One vs One, One vs All

> Is logistic regression a linear model? Why?

It can be seen as a generalized linear model. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value. In logistic regression the link function is logit (log-odds): $ln(\frac{\mu}{1 - \mu})$.

> What is sigmoid? What does it do?

$\frac{1}{1 + \exp^{ax + b}}$. It can be used to project the output space of a linear model to numbers between 0 and 1.

> (revisit) How do we evaluate classification models?

Confusion matrix has two dimensions: actual and predicted. In the binary confusion matrix case, we have positives and negatives. We can then calculate the accuracy as $\frac{TP+TN}{P+N}$. But if the classess are imbalanced the accuracy may be misleading. We can calculate true positive rate (recall) $\frac{TP}{P}$ and positive predictive value (precisiton) $\frac{TP}{PP}$. We can combine these two numbers to get a single metric. Aritmetic mean (Balanced Accuracy), Harmonic mean (F-measure), Geometric mean (Fowlkes–Mallows index). 

*How about multi class case?*

### Imbalance

> (revisit) Why imbalance in the classes is problematic?

If we have few positives relative to negatives, the training model will spend most of its time on negative examples and not learn enough from positive ones.

> (revisit) How to handle imbalance in the class labels?

If you have an imbalanced data set, we can first check the accuracy on the true distribution. If the model works well and generalizes that is good. If not, we undersample and oversample to make the distribution of classes balanced.

> What is probability calibration?

### Clustering
> What is unsupervised learning?

It is a machine learning task, we are given an unlabeled data and we would like to discover patterns in the data without any explicit guidance. Three types of unsupervised learning tasks are clustering, association and dimensionality reduction. 

> What is clustering? When do we need it?

The goal of clustering is to group together similar data points into clusters, maximize intra-cluster similarity and minimize inter-cluster similarity. It can be helpful to explore data we don't have labels or categories. It can be also used for anomaliy detection, we can outliers in the data that don't fit well into any cluster.

> Do you know how K-means works?

It is an iterative algorithm. We initialize cluster centroids randomly. We assign each data point to the cluster whose centroid is the nearest. We update the centroids by taking the mean of the data points in each cluster. We repeat those steps until the assignments of data points to clusters stabilize. 

> How to select K for K-means?

The number of clusters may be chosen using our domain knowledge. We can try to choose with cross-validation.

> (revisit) What are the other clustering algorithms do you know?

There are some limitiations of k-means algorithm. For instance it assumes equivariance. We can use gaussian mixtures. Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively. The DBSCAN algorithm views clusters as areas of high density separated by areas of low density. Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped. 

### Dimensionality Reduction
> What is the curse of dimensionality? Why do we care about it?

The more dimensions we add, the more sparse the data becomes, the more samples an algorithm needs to see before it is able to generate acceptable predictions.

> (revisit) Do you know any dimensionality reduction techniques?

1. Feature Selection
2. Feature Projection: PCA, LDA, autoencoder, NNMF, t-SNE, UMAP

> (revisit) What’s singular value decomposition? How is it typically used for machine learning?

The singular value decomposition is a factorization of a real or complex matrix into a rotation, followed by a rescaling followed by another rotation. It generalizes the eigendecomposition of a square normal matrix with an orthonormal eigenbasis to any matrix.

It can be used for dimensionality reduction, data compression, noise reduction, feature extraction, and latent factor analysis.

### Preprocessing

> What do we do with categorical variables?

We need to convert them to numerical variables since machine learning algorithms require numerical input. If we use label encoding we can artificially introduce ordinal relationship. We can introduce a new binary column for each category. If the number of categories is large the final input will be very high dimensional. We can use a pre-trained embedding to reduce the dimension size.

> Why scaling features might be useful?

Convergence if we use gradient-based algorithm.  Without scaling, the gradients can differ significantly in magnitude, leading to slow and unstable convergence.
Distance-based algorithms (e.g. Knn, Kmeans), if not scaled, variables with larger ranges can dominate the distance calculations. 
Regularization, If features are on different scales, regularization may penalize some coefficients more than others, leading to biased results. Scaling ensures that regularization is applied uniformly across all features.
When features are scaled, the model coefficients can be more easily compared to assess the relative importance of each feature. This is particularly useful in linear models where coefficients indicate the weight or influence of each feature on the target variable.

> What are the different ways of scaling?

min-max scaling: $x' = \frac{x-min}{max-min}$ scales the data to a fixed range $\[0, 1\]$. We can use standardization: $z = \frac{x-\mu}{\sigma}$ if the data has a gaussian distribution. But if there are outliers in the data we can use robust scaling instead $x' = \frac{x-median}{IQR}$.  
 

### Time Series
> What is a time series?

A dataset may be considered as a time series if it is keyed by time. Most commonly equally spaced points. It is a sequence because the data objects can be ordered by this key. 

> How is time series different from the usual regression problem?

While time series models (e.g. ARIMA) assumes that the data is ordered, in the regression model data points are assumed to be independent (the dataset is treated as a set not an ordered list).

> (revisit) Which models do you know for solving time series problems?

Autoregressive models (e.g. MA, ARIMA); non-linear autoregressive conditional heteroskedasticity (ARCH) models (e.g. GARCH). State-space models (e.g.  hidden-markov model). Kalman filter?

> (revisit) If there’s a trend in our series, how we can remove it? And why would we want to do it?

There are different methods to achieve this: (1) differencing or moving average, (2) detrending by regression, (3) filtering. Time series models like ARIMA uses differencing. State space model: observational equation: $y_t = \mu_t + \epsilon_t$, state equations: $\mu_t = \mu_{t-1} + \eta_t, \beta_t = \beta_{t-1} + \xi_t$. We would want to do detrending to have a more accurate forecast. 

> You have a series with only one variable “y” measured at time t. How do predict “y” at time t+1? Which approaches would you use?

Naive forecast, moving-average, exponential smoothing, ARIMA

> You have a series with a variable “y” and a set of features. How do you predict “y” at t+1? Which approaches would you use?

Now you use regression methods as well.

> What are the problems with using trees for solving time series problems?
