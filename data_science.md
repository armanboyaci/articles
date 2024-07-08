# Data Science Questions

> 1. There is a fair coin (one side heads, one side tails) and an unfair coin (both sides tails).
> You pick one at random, flip it 5 times, and observe that it comes up as tails all five times.
> What is the chance that you are flipping the unfair coin?

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















