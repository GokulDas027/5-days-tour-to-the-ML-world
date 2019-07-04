# Hands-on session.

## Linear Regression

Linear regression is a method for finding the straight line or hyperplane that best fits a set of points.

![dataset plot]()
![prediction plot]()

The line equation is,

> `y = mx + b` 

In machine learning we use this convention instead,

> `y' = b + w1x1`

Where,
**y'** is the label we are predicting.
**b** is the bias.(does the same thing what a constant c does to the line equation.Determines if to pass through origin or not)
**w1** is the weight of feature 1. Weight is the same concept as the "slope" in the traditional equation of a line.
**x1** is a feature (the input).

To predict, just substitute the x1 values to the trained model.

A sophisticated model can use more than one features.

> `y' = b + w1x1 + w2x2 + w3x3 + .... + wNxN`
> here, x1,x2,x3 are the different different features which predict for the label.

Loss function for Linear regression is Mean Squared Error.(Also known as L2 Loss.)

![L2 loss equation]()

Let's try implementing [Linear Regression in Scikit Learn.](https://colab.research.google.com/drive/1dbJr3bqCK8PIeavB0jHYWU0xZpkLm5Zj)

## Logistic Regression

Many problems require a probability estimate as output and here comes Logistic Regression.
* Logistic regression is an extremely efficient mechanism for calculating probabilities.

For example, consider that the probability of coconut falling on someone's head while walking through a field is 0.05. Then over the year, 18 accidents will happen in that field because of coconut!

> ``P(thenga|day) = 0.05
> coconut falling on head =
> 0.05*365 
> ~= 18``

A Sigmoid function, defined as follows, produces output that always falls between 0 and 1. So,the aim of logistic regression is to get a probabilistic value between 0 and 1, and you can see that in the graph

Sigmoid function
![sigmoid equation]()

Where,
> `y = w1x1 + w2x2 + ... wNxN`
> `and p is the predicted output.`

![sigmoid plot]()

Loss function for Logistic regression is Log Loss
![log loss equation]()

## Support Vector Machine

## k-Means Clustering
