#########################################
# Assignment 3. Principal components and implicit regularization
#########################################
# communities.csv contains results of crime level based on various characteristics of the given location.
# We are mainly looking at violent crimes per 100k.

# Libraries
library(caret)
library(dplyr)
library(ggplot2)
library(ggfortify)

#########################################
# Task 1
#########################################
# Scale all variables except of ViolentCrimesPerPop and implement PCA by using
# eigen(). Report how many components are needed to obtain at least 95% of
# variance in the data. What is the proportion of variation explained by each
# of the first two principal components?

# Reading and scaling data
data <- read.csv("communities.csv")
# View(data) # see data formatted
scaler <- preProcess(data %>% select(-ViolentCrimesPerPop))
data2 <- predict(scaler,data)

# PCA and eigen values
res <- prcomp(data2)
cov <- cov(data2)
eig <- eigen(cov)

# Components needed to obtain 95% of variance in data
cs <- cumsum(eig$values / sum(eig$values)*100)
which(cs >= 95)[1] # 35 components

# Proportion of variation by the two first principal components
sum(eig$values[1:2]) #41.97854

############### Task 2 #################
# Repeat PCA analysis by using princomp() function and make the trace plot of
# the first principle component. Do many features have a notable contribution
# to this component? report which 5 features contribute mostly (by the absolute
# value) to the first principle component. Comment whether these features have
# anything in common and whether they may have a logical relationship to the
# crime level. Also provide a plot of the PC scores in the coordinates (PC1,
# PC2) in which the color of the points is given by ViolentCrimesPerPop.
############### Task 2 #################

# Implement PCA using princomp
res2 <- princomp(data2)

# Trace plot of the first component
plot(res2[["loadings"]][,1], col="blue",pch=5, ylab="")

# Adding the 5 most contributing features by absolute value
top_5 <- head(sort(abs(res2[["loadings"]][, 1]), decreasing=TRUE), n=5)
top5_names <- names(res2$loadings[,1])[order(abs(res2$loadings[,1]), decreasing = TRUE)[1:5]]
print(top5_names)
index_top_5 <- which(abs(res2[["loadings"]][, 1]) %in% top_5)
points(index_top_5, res2[["loadings"]][index_top_5, 1], col = "red", pch = 5)

# Plot of the PC scores. Color of points is given by ViolentCrimesPerPop
autoplot(res2, colour = "ViolentCrimesPerPop") +
  labs(x = "PC1", y = "PC2", color = "Violent crimes per pop.")

# Results:
# Q: Report which 5 features
# contribute mostly (by the absolute value) to the first principal component.
# Comment whether these features have anything in common and whether
# they may have a logical relationship to the crime level.
# A: See top5_names (medFamInc, etc.)
# Q: Plot Analysis Scatter plot
# A: Points shift along PC1 as crime intensity increases. 
# Darker blue = higher crime. This suggests PC1 captures a 
# socio-economic gradient linked to crime.
# PC1 is dominant 
# Q: Trace plot analysis
# A: Most features have small loadings near 0,
# so they dont influence PC1 much.
# Red highlight five features with most impact (PC1), most variation.

# Terms:
# Principal components are orthogonal directions of maximum variance.
# PC1 is a weighted linear combination of original features.
# 35 PC's explain 95% of the variance? Means that the data has significant high-dimensional structure
# PC1-PC2 plot shows a correlation and the position along PC1.

############### Task 3 #################
# Split the original data into training and test (50/50) and scale both
# features and response appropriately, and estimate a linear regression model
# from training data in which ViolentCrimesPerPop is target and all other
# data columns are features. Compute training and test errors for these data
# and comment on the quality of model.
############### Task 3 #################

# Split data into train and test
n = dim(data)[1]
set.seed(12345)
id = sample(1:n, floor(n * 0.5))
train = data[id, ]
test = data[-id, ]

# Scale the data using preProcess, which will transform each variable in 'train'
scaler3 <- preProcess(train)
train_scaled <- predict(scaler3, train)
test_scaled <- predict(scaler3, test)

# Define feature columns (inputs X).
# We use all columns except the target column.
feature_column_names <- setdiff(colnames(train_scaled), "ViolentCrimesPerPop")

# Extract the feature matrix X and target vector y for training data
X_train <- train_scaled[, feature_column_names]
y_train <- train_scaled$ViolentCrimesPerPop

# Extract the feature matrix X and target vector y for test data
X_test <- test_scaled[, feature_column_names]
y_test <- test_scaled$ViolentCrimesPerPop

# Fit the linear regression model.
# ViolentCrimesPerPop is the target variable,
# "." means use all other columns as predictors.
model <- lm(ViolentCrimesPerPop ~ . - 1, data = train_scaled)

summary(model)

# Function for the average of (true-predicted value)^2
MSE <- function(y, y_hat) {
  return(mean((y - y_hat) ^ 2))
}

# Use the fitted to model to predict data
y_hat_train <- predict(model, X_train) # predict on the training data
y_hat_test <- predict(model, X_test) # predict on the test data
MSE(y_train, y_hat_train) 
MSE(y_test, y_hat_test) 

# Notes:
# Lower MSE means better predictive performance.

############### Task 4 #################
# Implements a cost function for linear regression.
# Uses BFGS for optimization.
############### Task 4 #################

# Vectors to store training and test errors at each function evaluation
train_errors <- c()
test_errors <- c()

# Cost function for linear regression parameters 'theta'
# theta is a vector of regression coefficients.
# The function returns the training MSE (which we minimize),
# but also records training and test MSE at each call.
cost_function <- function(theta) {
  train_error <- MSE(y_train, as.matrix(X_train) %*% theta)
  test_error <- MSE(y_test, as.matrix(X_test) %*% theta)
  
  train_errors <<- c(train_errors, train_error)
  test_errors <<- c(test_errors, test_error)
  
  return(train_error)
}

# Initial parameter vector (all zeros), one coefficient per feature
theta <- rep(0, ncol(X_train))

# Run BFGS optimization to minimize the training MSE
res <- optim(
  par = theta, # initial guess for parameters
  fn = cost_function, 
  method = "BFGS", # optimization method
  control = list(trace = 1, maxit = 200),
)

# Plot
plot(
  train_errors,
  type = "l",
  col = "blue",
  ylab = "Error",
  xlab = "Number of iterations",
  main = "Training and Test Errors over Iterations",
  ylim = c(0.25, 0.75),
  xlim = c(501, 10000),
)

# Mark the best test error with a green dot on the plot
points(test_errors, type = "l", col = "red", )

# Add a legend explaining the curves and the optimal point
legend(
  "topright",
  legend = c("Training Error", "Test Error", "Optimal Test Error"),
  lty = c(1, 1, 0),
  col = c("blue", "red", "green"),
  pch = c(NA, NA, 19),
)

# Print the optimal iteration and the corresponding errors
optimal_iteration <- which.min(test_errors)
optimal_iteration

train_errors[optimal_iteration]
test_errors[optimal_iteration]

points(optimal_iteration,
       test_errors[optimal_iteration],
       col = "green",
       pch = 19)
