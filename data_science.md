# Data Science Questions

> 1. There is a fair coin (one side heads, one side tails) and an unfair coin (both sides tails).
> You pick one at random, flip it 5 times, and observe that it comes up as tails all five times.
> What is the chance that you are flipping the unfair coin?

https://github.com/armanboyaci/notebooks/blob/master/datascience_questions/q1_datascience.ipynb

> 2. Assume we have a classifier that produces a score between 0 and 1 for the probability of a particular loan application being fraudulent.
> In this scenario:
> a) what are false positives, b) what are false negatives,
> and c) what are the trade-offs between them in terms of dollars and how should the model be weighted accordingly?

> 3. How would you design a metric to compare rankings of lists of shows for a given user?

> 4. A coin was flipped 1000 times, and 550 times it showed up heads. Do you think the coin is biased? Why or why not?

> 5. Assume you take have a stick of length 1 and you break it uniformly at random into three parts.
> What is the probability that the three parts form a triangle?


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

> 7. Assume we have some credit model, which has accurately calibrated (up to some error) score of how credit-worthy any individual person is.
> For example, if the model’s estimate is 92% then we can assume the actual score is between 91 and 93.
> If we take 92 as a score cutoff and deem everyone above that score as credit-worthy,
> are we over-estimating or underestimating the actual population’s credit score?


> 8. What does it mean for an estimator to be unbiased?
> What about consistent? Give examples of an unbiased but not consistent estimator, as well as a biased but consistent estimator.


> 9. Imagine the social graphs for both Facebook and Twitter.
> How do they differ? What metric would you use to measure how skewed the social graphs are?


> 10. Say you need to produce a binary classifier for fraud detection.
> What metrics would you look at, how is each defined, and what is the interpretation of each one?


> 11. Given a number n, return the number of lists of consecutive positive integers that sum up to n.
For example, for n = 9, you should return 3 since the lists are: [2, 3, 4], [4, 5], and [9]. Can you do it in linear time?


> 12. Assume you have the below tables on user actions. Write a query to get the active user retention by month.

| column_name | type |
|---|---|
| user_id | integer |
| event_id | string ("sign-in", "like", "comment") |
| timestampe | datetime |


> 13. Say we have X ~ Uniform(0, 1) and Y ~ Uniform(0, 1). What is the expected value of the minimum of X and Y?

> 14. Say we are given a list of several categories (for example, the strings: A, B, C, and D)
> and want to sample from a list of such categories according to a particular weighting scheme.
> Such an example would be: for 100 items total, we want to see A 20% of the time, B 15% of the time, C 35% of the time, and D 30% of the time.
> How do we simulate this? What if we care about an arbitrary number of categories and about memory usage?

> 15. How can you decide how long to run an experiment?
> What are some problems with just using a fixed p-value threshold and how do you work around them?

> 16. What is user churn and how can you build a model to predict whether a user will churn?
> What features would you include in the model and how do you assess importance?


> 17. You are drawing from a normally distributed random variable X ~ N(0, 1) once a day.
> What is the approximate expected number of days until you get a value of more than 2?


> 18. Say you are given an unfair coin, with an unknown bias towards heads or tails.
> How can you generate fair odds using this coin?


> 19. What is the expected number of coin flips needed to get two consecutive heads?


> 20. You’re on the data science team and are responsible for figuring out surge pricing.
> Why does it need to exist and what metrics and data should you track?

> 21. There are two games involving dice that you can play.
> In the first game, you roll two die at once and get the dollar amount equivalent to the product of the rolls.
> In the second game, you roll one die and get the dollar amount equivalent to the square of that value.
> Which has the higher expected value and why?


> 22. Say you are given a very large corpus of words. How would you identify synonyms?

> 23. Write a program to generate the partitions for a number n.
> A partition for n is a list of positive integers that sum up to n.
> For example: if n = 4, we want to return the following partitions: [1,1,1,1], [1,1,2], [2,2], [1,3], and [4].
> Note that a partition [1,3] is the same as [3,1] so only the former is included.


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

> 32. Say you are modeling the yearly revenue of new listings. What kinds of features would you use?
> What data processing steps need to be taken, and what kind of model would run?


> 33. Assume you want to test whether a new feature increases signups to the site.
> How would you run this experiment? What statistical test(s) would you use?

> 34. A and B are playing the following game: a number k from 1-6 is chosen, and A and B will toss a die until the first person sees the side k,
> and that person gets $100. How much is A willing to pay to play first in this game?


> 35. Say we have two random variables X and Y. What does it mean for X and Y to be independent?
> What about uncorrelated? Give an example where X and Y are uncorrelated but not independent.


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
