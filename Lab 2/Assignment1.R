#Lab 2 Assignment 1: Explicit Regularization


#Install necessary packages:
install.packages("dplyr")
install.packages("glmnet")


#-----Task 1-----

#First we make a dataframe
df_tecator = read.csv("tecator.csv")

#We use dplyr for select()
library(dplyr)

#Then we select Fat & Channels 1 to 100 (Absorbance characteristics)
#into a new dataframe, excluding Protein & Moisture
df = df_tecator %>% select(Fat, Channel1:Channel100)

#After that we do partitioning code into train/test
#50-50 from lecture 1a slide 34
n = dim(df)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.5))
train = df[id,]
test = df[-id,]

#The underlying probabilistic model
#with Fat = Y and Channels = X_j
#can be reported as the following:

#Y_i = B0 + sum(Bj * X_i,j) for j=1 to 100 + E_i

#Where we have:
# Y_i = Fat measurement for sample i
# B0 = Intercept coefficient
# Bj = Regression coefficient for Channel j
# X_i,j = Absorbance value or Channel j in sample i
# E_i = Error term for sample i

#Then we compute the linear regression
#model with Fat as Target
#and Channels 1 to 100 as features,
#and fit the model to the training data
m1 = lm(Fat~., data = train)

#Estimation of training error:

#Create predictions using the model m1 on the
#training data
preds_train = predict(m1, newdata = train)

#Calculate the squared error for each sample,
#with (actual-predicted)^2
SE_train = (train$Fat - preds_train)^2

#Calculate the mean squared error for training:
MSE_train = mean(SE_train)

#Print the MSE_train:
MSE_train # = 0.005709117 ≈ 0.57%

#Estimation of test error:

preds_test = predict(m1, newdata = test)

#Calculate the squared error for each sample,
#with (actual-predicted)^2
SE_test = (test$Fat - preds_test)^2

#Calculate the mean squared error for test:
MSE_test = mean(SE_test)

#Print the MSE_test:
MSE_test # = 722.4294 ≈ 7220.43% <- very bad!

#Comment on the quality of fit and prediction
#and therefore on the quality of model:

#The MSE for training data is 0.57%, which is quite
#low and therefore a good fit. The model was able to
#capture the training data well.
#But the MSE for test data is 7220.43%, which
#results in bad predictions, as the model
#fails hard on unseen data.
#So the model has high variance which leads to
#high levels of overfitting.

#-----Task 2-----

#The cost function to be optimized (minimized) for the LASSO
#regression with Fat = Y and Channels = X_j is:

#J(B) = Loss term + Penalty term

#J(B) = (1 / 2*n) * sum((Y_i - (B0 + sum(Bj * X_i,j) for j=1 to 100))^2)
# + lambda * sum(|Bj| for j=1 to 100)

#Where:
# Loss term = (1 / 2*n) * sum((Y_i) - Prediction_i)^2)
# This measures the quality of the fit with MSE.
# Penalty term = lambda * sum(|Bj| for j=1 to 100)
# This penalizes large coefficients to prevent overfitting
# and forces weak coefficients to zero, called feature selection.

#Where we have:
# Y_i = Actual Fat measurement for sample i
# B0 = Intercept coefficient
# Bj = Regression coefficient for Channel j
# X_i,j = Absorbance value for Channel j in sample i
# n = Number of training samples
# lambda = Tuning parameter, which controls
# the strength of the penalty.


#-----Task 3-----

#Select Fat & all Channels into a new dataframe
df_2 = train%>%select(Fat, Channel1:Channel100)

#Use the glmnet library for Lasso/Ridge regression
library(glmnet)

#Convert all Channels into a matrix x (features)
x = as.matrix(df_2%>%select(-Fat))

#Convert Fat into a matrix y (target)
y = as.matrix(df_2%>%select(Fat))

#Fit the Lasso regression model
#No lambda=1, which allows glmnet to fit
#the model across the entire sequence of lambdas
#alpha = 1 to use lasso (ridge if alpha = 0)
mB_lasso = glmnet(x, y, family = "gaussian", alpha = 1)

#Plot:
plot(mB_lasso, xvar = "lambda", label = TRUE)

#Plot interpretation:

#-Log(lambda) is used for aesthetics
#so it shows the simplest model (high penalty)
#to the left and moving right showing the most
#complex model (lowest penalty).
#High penalty (lambda) = simple
#Low penalty (lambda) = complex

#Numbers on top:
#The numbers on top of the plot show the number
#of non-zero coefficients aka the number of
#selected features at different points along the
#penalty line.

#X-axis (-log(lambda)):
#Going from left to right the log(lambda) value
#aka penalty factor is decreasing.
#As the lambda (penalty factor) decreases,
#the model complexity increases, and more features
#enter the model when we go further right.

#Y-axis (coefficients):
#Shows the coefficient values, each colored line
#shows the coefficient (Bj) for one of the 100 channels.

#Feature selection:
#The lasso (alpha=1) forces the coefficients to
#be exactly zero where the lines start on the far left.
#When going further right, lines separate from
#the Y = 0 line, which means those features
#have been selected (aka coefficient is not zero).

#What value of the penalty factor can be chosen if
#we want to select a model with only 3 features?

#The top of the plot shows number of selected features,
#to get 3 features we must find a value between 1 and 9 features,
#where we start seeing 3 different colored lines leaving the Y = 0 line.
#This is where the red and green lines emerge, together with the very first red one.
#We estimate this to happen at -Log(lambda) = 0.2.
#So we solve that equation:
#   -Log(lambda) = 0.2
#<=>e^-log(lambda) = e^0.2
#<=>e^log(lambda^-1) = e^0.2
#<=>lambda^-1 = e^0.2
#<=>1/lambda = e^0.2
#<=>lambda = 1/(e^0.2) ≈ 0.818731 ≈ 0.82

#So the penalty factor (lambda) should be chosen
#to around 0.82 to select a model with only
#3 features.


#-----Task 4-----

#Same as in Task 3 but using Ridge regression
#instead of Lasso, which is done by changing
#alpha from 1 to 0, so alpha=0 in the glmnet() function.
mB_ridge = glmnet(x, y, family = "gaussian", alpha = 0)

#Plot:
plot(mB_ridge, xvar = "lambda", label = TRUE)

#Compare the plots from steps 3 and 4. Conclusions?

#Comparison of regularization:

#Lasso (feature selection):
# The lasso plot is a more aggressive model,
# because it forces the coefficients of less
# important channels to become exactly zero
# (lines hit the x-axis) going from right to left
# increasing the lambda (penalty factor).
# This lets us know
# which specific few channels are most important,
# but it let the rest of the chosen coefficients
# get very large, up to 100.

#Ridge (feature shrinkage):
# The ridge plot is a safer model, because it
# keeps all 100 channels in the model going from
# right to left, but
# forces all 100 coefficients to be tiny,
# max around 8. This evenly distributes the
# prediction work and drastically reduces the
# risk of any single channel's coefficient
# becoming too large and capturing noise.

#Conclusion of model choice:
# For this data with 100 features, ridge regression
# is a better choice for prediction quality. By
# shrinking all 100 coefficients to small values,
# it effectively controls the overfitting (variance)
# problem, resulting in a model that generalizes
# better to new data than the lasso model in the beginning.


#-----Task 5-----

#Select Fat & all Channels into a new dataframe
df_3 = train%>%select(Fat, Channel1:Channel100)

#Convert all Channels into a matrix x (features)
x = as.matrix(df_3%>%select(-Fat))

#Convert Fat into a matrix y (target)
y = as.matrix(df_3%>%select(Fat))

#Create the cross validation lasso model
mB_cv_lasso = cv.glmnet(x, y, alpha=1, family="gaussian")

#Plot the cv lasso model showing the
#dependence of the CV score on log(lambda):
plot(mB_cv_lasso)

#Comment how the CV score (Mean CV error) changes with log(lambda):
# High penalty aka right side:
# As log(lambda) gets very large (high penalty)
# the model is too simple (underfit). The CV score is high
# because the model has high bias.

# Optimal penalty aka bottom of curve:
# As log(lambda) decreases from the right, the CV
# score decreases too, reaching its minimum lambda.
# This is the optimal point where bias and variance
# are balanced.

# Low penalty aka left side:
# As log(lambda) gets smaller (low penalty, more complex model),
# the CV score increases again. The model is too complex
# and overfits the training data, leading to high variance
# and high CV error.

#The optimal penalty (lambda)
mB_cv_lasso$lambda.min # = 0.05744535

#How many variables were chosen in this model:
# 29 variables were chosen according to the plot.
# But for the optimal parameter 8 were enough.

#Does the information displayed in the plot
#suggests that the optimal 𝜆 value results
#in a statistically significantly better
#prediction than log𝜆 = −4?
# -log(0.05744535) = 2.85692,
# aka log(𝜆_optimal) = -2.85692
# If we look at the plot, we see that
# -4 is very close to the bottom of the curve,
# and within the standard error range, so
# the difference in prediction error is not
# statistically significant. So we can choose
# log(𝜆) = -4 as well without greater loss.

#Scatter plot of original test versus
#predicted test values for the model corresponding
#to optimal lambda:

#preds_test = original test from Task 1

#Select Fat & all Channels into a new dataframe
df_4 = test%>%select(Fat, Channel1:Channel100)

#Convert all test Channels into a matrix x (features)
x = as.matrix(df_4%>%select(-Fat))

#Create the preds for test with cross validation lasso model
preds_test_optimal = predict(mB_cv_lasso, s = mB_cv_lasso$lambda.min, newx = x)

#Create the plot using optimal predictions
plot(
  test$Fat, #Actual values on x
  preds_test_optimal, #Optimal predictions on y
  main = "Comparison of prediction models on test data",
  xlab = "Actual fat content (test set)",
  ylab = "Predicted fat content",
  pch = 19,
  col = "darkblue", #Dark blue for optimal predictions
  xlim = range(test$Fat),
  ylim = range(c(preds_test, preds_test_optimal))
)

#Add original predictions to the same plot
points(
  test$Fat,
  preds_test,
  pch = 17,
  col = "red" #Red for original predictions
)

#Add ideal fit Line (y=x)
abline(a = 0, b = 1, col = "black", lty = 2, lwd = 2)

#Add legend to clarify the models
legend("topleft",
       legend = c("Optimal Lasso-CV preds", "Initial test preds", "Ideal fit Line"),
       col = c("darkblue", "red", "black"),
       pch = c(19, 17, NA),
       lty = c(NA, NA, 2),
       lwd = c(NA, NA, 2))

#Comment whether the model predictions are good:
# The predictions are much better now, as the
# dark blue points (Lasso-CV) cluster much
# more tightly around the ideal fit line compared
# to the red dots (original test),
# although not all the points are perfectly on
# the line, which indicates that the model
# still has some error. But overall it is a lot
# better than the original model.
# It can be said that the optimal model has
# much higher predictive reliability.
