# StockPredictorModel
The StockPredictorModel is an algorithm written in Scala using logistic regression to predict whether the stock index goes Up or Down.  

The algorithm is based on a lab exercise in R from the online course “Statistical Learning” taught by Trevor Hastie and Rob Tibshirani form the Stanford University. 

## Dataset 
The dataset consists of the percentage returns of the S&P stock index from 2001 to 2005 and the direction whether the market was Up or Down on this date.


## Method
In order to predict future directions, a logistic regression model is trained with 80% of the dataset and tested by the remaining 20%.

## Accuracy 
The result is an overall accuracy rate of 56% and on days when logistic regression predicts an increase in the market it has a 58% accuracy rate. 

## Getting Started
To run the program Apache Spark and Hadoop are necessary. You can either use your own local machine by installing a standalone mode of Spark or using a cloud service such as Databricks.  
