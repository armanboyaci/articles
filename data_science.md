# Data Science Questions

> 1. There is a fair coin (one side heads, one side tails) and an unfair coin (both sides tails).
> You pick one at random, flip it 5 times, and observe that it comes up as tails all five times.
> What is the chance that you are flipping the unfair coin?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q1_datascience.ipynb

> 2. Assume we have a classifier that produces a score between 0 and 1 for the probability of a particular loan application being fraudulent.
> In this scenario:
> a) what are false positives, b) what are false negatives,
> and c) what are the trade-offs between them in terms of dollars and how should the model be weighted accordingly?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q2_datascience.ipynb

> 3. How would you design a metric to compare rankings of lists of shows for a given user?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q3_datascience.ipynb

> 4. A coin was flipped 1000 times, and 550 times it showed up heads. Do you think the coin is biased? Why or why not?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q4_datascience.ipynb

> 5. Assume you take have a stick of length 1 and you break it uniformly at random into three parts.
> What is the probability that the three parts form a triangle?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q5_datascience.ipynb

> 6. Assume you are given the below tables for trades and users.
> Write a query to list the top 3 cities which had the highest number of completed orders.

##### trades
| column_name  | type  | 
|---|---|
| order_id  | integer   |
| user_id   | integer   |
| symbol    | string (e.g. "NLFX")  |
| price     | float     | 
| quantity  | integer   |
| side      | string ("buy", "sell") |
| status    | string ("complete", "cancelled") |
| timestamp | datetime  |

##### users
| column_name | type |
|---|---|
| user_id	| integer |
| city	| string |
| email	| string | 
| signup_date |	datetime |

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q6_datascience.ipynb

> 7. Assume we have some credit model, which has accurately calibrated (up to some error) score of how credit-worthy any individual person is.
> For example, if the model’s estimate is 92% then we can assume the actual score is between 91 and 93.
> If we take 92 as a score cutoff and deem everyone above that score as credit-worthy,
> are we over-estimating or underestimating the actual population’s credit score?


> 8. What does it mean for an estimator to be unbiased?
> What about consistent? Give examples of an unbiased but not consistent estimator, as well as a biased but consistent estimator.

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q8_datascience.ipynb

> 9. Imagine the social graphs for both Facebook and Twitter.
> How do they differ? What metric would you use to measure how skewed the social graphs are?


> 10. Say you need to produce a binary classifier for fraud detection.
> What metrics would you look at, how is each defined, and what is the interpretation of each one?


> 11. Given a number n, return the number of lists of consecutive positive integers that sum up to n.
For example, for n = 9, you should return 3 since the lists are: [2, 3, 4], [4, 5], and [9]. Can you do it in linear time?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q11_datascience.ipynb

> 12. Assume you have the below tables on user actions. Write a query to get the active user retention by month.

| column_name | type |
|---|---|
| user_id | integer |
| event_id | string ("sign-in", "like", "comment") |
| timestampe | datetime |


> 13. Say we have X ~ Uniform(0, 1) and Y ~ Uniform(0, 1). What is the expected value of the minimum of X and Y?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q13_datascience.ipynb

> 14. Say we are given a list of several categories (for example, the strings: A, B, C, and D)
> and want to sample from a list of such categories according to a particular weighting scheme.
> Such an example would be: for 100 items total, we want to see A 20% of the time, B 15% of the time, C 35% of the time, and D 30% of the time.
> How do we simulate this? What if we care about an arbitrary number of categories and about memory usage?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q14_datascience.ipynb

> 15. How can you decide how long to run an experiment?
> What are some problems with just using a fixed p-value threshold and how do you work around them?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q15_datascience.ipynb

> 16. What is user churn and how can you build a model to predict whether a user will churn?
> What features would you include in the model and how do you assess importance?


> 17. You are drawing from a normally distributed random variable X ~ N(0, 1) once a day.
> What is the approximate expected number of days until you get a value of more than 2?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q17_datascience.ipynb

> 18. Say you are given an unfair coin, with an unknown bias towards heads or tails.
> How can you generate fair odds using this coin?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions%20/q18_datascience.ipynb

> 19. What is the expected number of coin flips needed to get two consecutive heads?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q19_datascience.ipynb

> 20. You’re on the data science team and are responsible for figuring out surge pricing.
> Why does it need to exist and what metrics and data should you track?

> 21. There are two games involving dice that you can play.
> In the first game, you roll two die at once and get the dollar amount equivalent to the product of the rolls.
> In the second game, you roll one die and get the dollar amount equivalent to the square of that value.
> Which has the higher expected value and why?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q21_datascience.ipynb

> 22. Say you are given a very large corpus of words. How would you identify synonyms?

> 23. Write a program to generate the partitions for a number n.
> A partition for n is a list of positive integers that sum up to n.
> For example: if n = 4, we want to return the following partitions: [1,1,1,1], [1,1,2], [2,2], [1,3], and [4].
> Note that a partition [1,3] is the same as [3,1] so only the former is included.

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q23_datascience.ipynb

> 24. Three ants are sitting at the corners of an equilateral triangle.
> Each ant randomly picks a direction and starts moving along the edge of the triangle.
> What is the probability that none of the ants collide?
> Now, what if it is k ants on all k corners of an equilateral polygon?



> 25. Say you are running a multiple linear regression and believe there are several predictors that are correlated.
> How will the results of the regression be affected if they are indeed correlated? How would you deal with this problem?


> 26. What are some factors that might make testing metrics on the Airbnb platform difficult?

> 27. Assume we have a classifier that produces a score between 0 and 1 for the probability of a particular loan application behind a fraud.
> Say that for each application’s score, we take the square root of that score.
> How would the ROC curve change? If it doesn’t change, what kinds of functions would change the curve?


> 28. What is the bias-variance tradeoff? How is it expressed using an equation?

> 29. How many cards would you expect to draw from a standard deck before seeing the first ace?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q29_datascience.ipynb

> 30. Assume you are given the below table for spending activity by product type.
> Write a query to calculate the cumulative spend for each product over time in chronological order.

| column_name | type |
|---|---|
| order_id   | integer |
| user_id    | integer |
| product_id | string |
| spend      | float  |
| date       | datetime |

> 31. Write a program to calculate correlation (without any libraries except for math) for two lists X and Y.

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q31_datascience.ipynb

> 32. Say you are modeling the yearly revenue of new listings. What kinds of features would you use?
> What data processing steps need to be taken, and what kind of model would run?


> 33. Assume you want to test whether a new feature increases signups to the site.
> How would you run this experiment? What statistical test(s) would you use?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q33_datascience.ipynb

> 34. A and B are playing the following game: a number k from 1-6 is chosen, and A and B will toss a die until the first person sees the side k,
> and that person gets $100. How much is A willing to pay to play first in this game?


> 35. Say we have two random variables X and Y. What does it mean for X and Y to be independent?
> What about uncorrelated? Give an example where X and Y are uncorrelated but not independent.

https://colab.research.google.com/drive/10UcUx0nC7uxIN-00xVZCGj-RtpxRAf4y#scrollTo=7JLJpktB_5Wy

> 36. You and your friend are playing a game. The two of you will continue to toss a coin until the sequence HH or TH shows up.
> If HH shows up first, you win. If TH shows up first, your friend wins. What is the probability of you winning?

> 37. Say you are tasked with producing a model that can recommend similar listings to an Airbnb user when they are looking at any given listing.
> What kind of model would you use, what data is needed for that model, and how would you evaluate the model?

> 38. In the streaming context, for A/B testing, what are some metrics and data to track, and what are some differences versus more traditional A/B testing?

> 39. Given a list of positive integers, return the maximum increasing subsequence,
> that is, the largest increasing subsequence within the array that has the maximum sum.
> Examples: if the input is [5, 4, 3, 2, 1] then return 5 (since no subsequence is increasing), if the input is [3, 2, 5, 7, 6] return 15 = 3 + 5 + 7, etc.

> 40. What are the assumptions behind linear regression? How do you diagnose if any of these assumptions are violated?

> 41. Say we have X ~ Uniform(-1, 1) and Y = X^2. What is the covariance of X and Y?

> 42. Facebook has a content team that labels pieces of content on the platform as spam or not spam.
> 90% of them are diligent raters and will label 20% of the content as spam and 80% as non-spam.
> The remaining 10% are non-diligent raters and will label 0% of the content as spam and 100% as non-spam.
> Assume the pieces of content are labeled independently from one another, for every
> rater. Given that a rater has labeled 4 pieces of content as good, what is the probability that they are a diligent rater? 

> 43. Say you have an unfair coin which will land on heads 60% of the time. How many coin flips are needed to detect that the coin is unfair?

> 44. Given a positive integer n, find the smallest number of perfect squares that sum up to n.
> For example, for n = 7, you should return 4 since 7 = 4 + 1 + 1 +1, and for n = 13, you should return 2 since 13 = 4 + 9.

> 45. Your team is trying to figure out whether a new driver app with extra UI features will increase the number of rides taken.
> How would you test whether the extra features in the app make it better than the original version?

> 46. Say we are running a probabilistic linear regression which does a good job modeling the underlying relationship between some y and x.
> Now assume all inputs have some noise ε added, which is independent of the training data. What is the new objective function? How do you compute it?

> 47. A fair die is rolled n times. What is the probability that the largest number rolled is r, for each r in 1..6?

> 48. You have the entire social graph of Facebook users, with nodes representing users and edges representing friendships between users.
> Given the edges of the graph and the number of nodes, write a function to return the smallest number of friendships in-between two users.
> For example, if the graph consists of 5 users A, B, C, D, E, and the friendship edges are: (A, B), (A, C), (B, D), (D, E) then the function should return 2 for the input A and E.

> 49. Assume you have the below tables on sessions that users have, and a users table.
> Write a query to get the active user count of daily cohorts, i.e. the counts of users registered each day.

##### sessions
| column_name |	type |
|---|---|
| session_id	| integer |
| user_id	| integer |
| date |	datetime |

##### users
| column_name |	type |
|---|---|
| user_id	| integer |
| email	| string |
| date	| datetime |

> 50. How would you build a model to calculate a customer's propensity to buy a particular item? What are some pros and cons of your approach?

> 51. Let’s say that you are the first person working on the Facebook News Feed. What metrics would you track and how would you improve those metrics?

> 52. Say you pick the radius of a circle from a uniform distribution between 0 and 1. What is the probability density of the area of the resulting circle?

> 53. Given a binary tree, write a function to determine whether the tree is a mirror image of itself. Two trees are a mirror image if their root values are the same and the left subtree is a mirror image of the right subtree.

> 54. There are two groups of n Snapchat users, A and B, and each user in A is friends with those in B and vice versa.
> Each user in A will randomly choose a user in B as their best friend and each user in B will randomly choose a user in A as their best friend.
> If two people have chosen each other, they are mutual best friends. What is the probability that there will be no mutual best friendships? 

> 55. Say you model the lifetime for a set of customers using an exponential distribution with parameter λ,
> and you have the lifetime history (in months) of n customers. What is the MLE for λ?

> 56. Assume you are given the below tables on users and user posts. Write a query to get the distribution of the number of posts per user.

#### users
| column_name	| type |
|---|---|
| user_id	| integer |
| date	| datetime | 

#### posts

| column_name	| type |
|---|---|
| post_id	 | integer | 
| user_id	| integer | 
| body	| string | 
| date	| datetime | 

> 57. What is the loss function used in k-means clustering for k clusters and n sample points?
> Compute the update formula using 1) batch gradient descent, 2) stochastic gradient descent for the cluster mean for cluster k using a learning rate ε.

> 58. Dropbox has just started and there are two servers that service users: a faster server and a slower server.
> When a user is on the website, they are routed to either server randomly,
> and the wait time is exponentially distributed with two different parameters.
> What is the probability density of a random user's waiting time?

> 59. Estimate π using a Monte Carlo method. Hint: think about throwing darts on a square and seeing where they land within a circle.

> 60. How would you improve product engagement on Twitter?

> 61. Say you roll three dice, one by one. What is the probability that you obtain 3 numbers in a strictly increasing order?

> 62. Say you draw n samples from a uniform distribution U(a, b). What is the MLE estimate of a and b?

> 63. Given a stream of elements (of arbitrary size), write a class to find the median at any given time.
> Your class should have a function to add numbers to the stream and a function to calculate the median.

> 64. Assume you are given the below tables for the session activity of users.
> Write a query to assign ranks to users by the total session duration for the different session types
> they have had between a start date (2020-01-01) and an end date (2020-02-01).

#### Sessions

| column_name	| type |
|---|---|
| session_id	| integer  | 
|user_id	    | integer  |
|session_type	| string   |
|duration	    | integer  |
|start_time	  | datetime |

> 65. Say you are deciding whether to implement two-step authentication when users log in. What data would you look at and how can you make your decision?

> 66. Say you are given a random Bernoulli trial generator. How would you generate values from a standard normal distribution?

> 67. What are MLE and MAP? What is the difference between the two?

> 68. A and B are playing a game where A has n+1 coins, B has n coins, and they each flip all of their coins. What is the probability that A will have more heads than B?

> 69. A fair coin is tossed n times. Given that there were k heads in the n tosses, what is the probability that the first toss was heads?

> 70. Say you have two countries of interest and want to compare variances of clicking behavior from users (i.e. total distribution of clicks).
> How would you do this comparison, and what assumptions need to be met?

> 71. Given an array A of positive integers, a peak element is defined as an element that is greater than its neighbors.
> For example, if `A = [3, 5, 2, 4, 1]` you should return either 1 or 3 since index 1 is 5, and index 3 is 4, and both are peak elements. Find the index of any peak elements.

> 72. You're working with several sensors that are designed to predict a particular energy consumption metric on a vehicle.
> Using the outputs of the sensors, you build a linear regression model to make the prediction.
> There are many sensors, and several of the sensors are prone to complete failure.
> What are some cost functions you might consider, and which would you decide to minimize in this scenario?

> 73. Assume you are given the below tables on users and their time spent on sending and opening Snaps.
> Write a query to get the breakdown for each age breakdown of the percentage of time spent on sending versus opening snaps.

#### activities

| column_name	| type |
|---|---|
| activity_id |	integer |
| user_id |	integer |
| type |	string ("send", "open") |
| time_spent |	float |
| activity_date |	datetime |

#### age_breakdown

| column_name	| type |
|---|---|
| user_id	| integer |
| age_bucket	| string |

> 74. Let's say Facebook has expanded into a previously untapped geographical region.
> Looking at weekly metrics, you see a slow decrease in the average number of comments per user over several months.
> You also know that new users have been growing at a steady linear rate in the area for the same time frame.
> What are some reasons why the average number of comments per user would be decreasing and what metrics would you look into?

> 75. Suppose we have two random variables X and Y. Under what condition are X+Y and X-Y uncorrelated?

> 76. A biased coin, with probability p of landing on heads, is tossed n times.
> Write a recurrence relation for the probability that the total number of heads after n tosses is even.

> 77. Say that there are n topics on Twitter and there is a notion of topics being related.
> Specifically, if topic A is related to topic B, and topic B is related to topic C, then topic A is indirectly related to topic C.
> Define a topic group to be any group of topics that either directly or indirectly related.
> Given an n by n adjacency matrix N, where N[i][j] = 1 if topic i and topic are j related and 0 otherwise, write a function to determine how many topic groups are there.

> 78. Assume you are given the below table on sessions from users, with a given start and end time. A session is concurrent with another session if they overlap in their start and end times. Write a query to output the session that is concurrent with the largest number of other sessions.

#### sessions
| column_name	| type |
|---|---|
| session_id	| integer |
| user_id	| integer |
| session_type	| string |
| duration	| integer |
| start_time	| datetime | 

> 79. Let’s say that the usage of a Facebook posting tool dropped from 2% posts per user last month to 1% post per user today.
> What might be some potential causes and how would you go about understanding this drop?

> 80. Consider a Bernoulli random variable with parameter p. Say you observe the following samples: [1, 0, 1, 1, 1].
> What is the log likelihood function for p and what is the MLE of p?

> 81. What is Expectation-Maximization and when is it useful? Describe the setup algorithmically with formulas.

> 82. Say that you are pushing a new feature X out. You have 1000 users and each user is either a fan or not a fan of X, at random.
> There are 50 users out of 1000 that do not like X. You will decide whether to ship the feature or not, based on sampling 5 users independently
> and if they all like the feature, you will ship it. What is the probability that you will ship the feature?

> 83. Given n distinct integers, write a function to generate all permutations of those integers.

> 84. Assume you are given the below table on purchases from users. Write a query to get the number of people that purchased at least one product on multiple days.

#### purchases
| column_name	| type |
|---|---|
| purchase_id	| integer |
| user_id |	integer |
| product_id |	integer |
| quantity |	integer |
| price	| float |
| purchase_time	| datetime |

> 85. A fair die is rolled n times. What is the expected value and the standard deviation of the smallest number rolled?

> 86. Disney+ offers a 7-day free trial period. After 7 days, customers are charged whatever package they chose.
> Assume that there are customers who commit to Disney+ right away and therefore don't end up having a 7-day free trial period.
> Disney wants to measure the success of the free trial.
> What metrics and analysis might you do to determine whether or not the free trial successfully acquires new customers?

> 87. Alice and Bob are playing a new Facebook game together. They play a series of rounds until one of them wins two more rounds than the other.
> With probability p, Alice will win each round. What is the probability that Bob wins the overall series?

> 88. Assume you are given the below table on reviews from users. Define a top-rated place as a business whose reviews only consist of 4 or 5 stars. Write a query to get the number and percentage of businesses that are top-rated places.

#### reviews
| column_name	| type |
|---|---|
| business_id	| integer |
| user_id	| integer |
| review_text |	string |
| review_stars |	integer |
| review_date	| datetime |

> 89. Say you have n integers 1...n and take a random permutation.
> For any integers i, j let a swap be defined as when the integer i is in the jth position, and vice versa.
> What is the expected value of the total number of swaps?

> 90. Given two arrays, write a function to get the intersection of the two.
> For example, if A = [2, 4, 1, 5, 0], and B = [3, 4, 5] then you should return [4, 5].

> 91. Say we are using a Gaussian Mixture Model (GMM) for anomaly detection on fraudulent transactions to classify incoming transactions into K classes.
> Describe the model setup formulaically and how to evaluate the posterior probabilities and log likelihood.
> How can we determine if a new transaction should be deemed fraudulent?

> 92. LinkedIn recently launched a "status" feature where you can now see if a LinkedIn connection is online,
> symbolized by a green dot, idle, symbolized by an orange dot, or offline (grey dot) status that says how long ago the user was active.
> Assume that this feature has been around for a few months. What metrics would you look at to assess the success of this feature?

> 93. Suppose we have two random variables, X and Y, which are bivariate normal.
> The correlation between them is -0.2. Let A = cX + Y and B = X + cY. For what values of c are A and B independent?

> 94. If you are selling a product and want to decrease the shipment time from 2 days to 1 day to increase your amount of customers,
> what are 2 ways to quantify it and 2 risks of this idea?

> 95. You are analyzing the probability of failure or success of a small social media app competitor.
> Using some initial data, you estimate that any step, if there is 1 user then after a day there is a 1/12 chance there will be 0 users,
> 5/12 chance there will be 1 user, and 1/2 chance there will be 2 users.
> Say the app starts with one user on day 1. What is the probability that the app will eventually have no users?

> 96. Assume you are given the below table of measurement values from a sensor for several days.
> Each measurement can happen several times in a given day.
> Write a query to output the sum of values for every odd measurement and the sum of values for every even measurement by date.

### measurements
| column_name |	type |
|---|---|
| measurement_id	  | integer  |
| measurement_value |	integer  | 
| measurement_time  |	datetime |


> 97. Given a number x, define a palindromic subset as any subsequence within x that is a palindrome.
> Write a function that returns the number of digits of the longest palindromic subset.
> For example, if x is 93567619 then you should return 5 since the longest subset would be 96769, which is a 5 digit number.

> 98. Suppose there is a new vehicle launch upcoming.
> Initial data suggests that any given day there is either a malfunction with some part of the vehicle or possibility of a crash,
> with probability p which then requires a replacement. Additionally, each vehicle that has been around for n days must be replaced.
> What is the long-term frequency of vehicle replacements?

> 99. Explain what Information Gain and Entropy are in a Decision Tree.

> 100. Say you have a large amount of user data that measures the lifetime of each user.
> Assume you model each lifetime as exponentially distributed random variables.
> What is the likelihood ratio for assessing two potential λ values, one from the null hypothesis and the other from the alternative hypothesis?

> 101. You are a data scientist who works directly with the CEO. Your boss says she is incredibly ecstatic because the average cost of acquiring a customer is a lot
> lower than the expected value of a customer. She thinks that the acquisition cost has been minimized and the value of a customer has almost been maximized.
> Help her interpret the metric and give a suggestion about how she should use it to try and maximize revenue.

> 102. Given a string with lowercase characters and left and right parentheses, remove the minimum number of parentheses so that the string is valid.
> For example, if the string is ")a(b((cd)e(f)g)" then return "ab((cd)e(f)g)".

> 103. Describe the idea and mathematical formulation of kernel smoothing. How do you compute the kernel regression estimator?

> 104. What are some metrics you would use to measure user engagement at Slack? How would you be able to tell early whether or not user engagement is declining?

> 105. Alice and Bob are choosing their top 3 shows from a list of 50 shows. Assume that they choose independently of one another.
> Being relatively new to Hulu, assume also that they choose randomly within the 50 shows.
> What is the expected number of shows they have in common, and what is the probability that they do not have any shows in common?

> 106. Assume you are given the below table on transactions from various product search results from users on Etsy.
> For every given product keyword, there are multiple positions that being A/B > tested, and user feedback is collected on the relevance of results (from 1-5).
> There are many displays for each position of every product, each of which is captured by a display_id.
> Define a highly relevant display as one whereby the corresponding relevance score is at least 4.
> Write a query to get all products having at least one position with > 80% highly relevant displays.

### product_searches
| column_name |	type |
|---|---|
| product	| string |
| position |	integer |
| display_id |	integer |
| relevance |	integer |
| submit_time |	datetime |

> 107. Say that the lifetime of electric vehicles are modeled using a Gaussian distribution.
> Each type of electric vehicle has an expected lifetime and a lifetime variance.
> Say you chose two different types of electric vehicles at random. What is the probability that the two lifetimes will be within n time units?

> 108. Given an arbitrary array of positive integers, find the smallest missing positive integer. Can you do it with O(1) space?
> For example, if A = [1, 3, 6, 2, 7] and then you should return 4.

> 109. Say you roll three dice and observe the sum of the three rolls. What is the probability that the sum of the outcomes is 12, given that the three rolls are different?

> 110. Say we have N observations for some variable which we model as being drawn from a Gaussian distribution.
> What are your best guesses for the parameters of the distribution? Derive it mathematically.
