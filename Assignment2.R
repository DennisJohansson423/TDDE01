# Assignment 2. Linear regression and ridge regression.
library(caret)

# ----1.----

data = read.csv("parkinsons.csv")

# Split data training/test data 60/40
n = dim(data)[1]
set.seed(12345)
id = sample(1:n, floor(n * 0.6))
train = data[id, ]
test = data[-id, ]

# Scale data using preProcess so all features are comparable.
scaler <- preProcess(train)
train_scaled <- predict(scaler, train)
test_scaled <- predict(scaler, test)

# Define training columns and test columns
y_train <- train_scaled[, 5] # set column 5 as training column
X_train <- train_scaled[, -(1:6)] # drop columns 1-6
y_test <- test_scaled[, 5]
X_test <- test_scaled[, -(1:6)]

# ----2.----

# Define the MSE (difference between actual and predicted output)
mse <- function(y, y_hat)
  mean((y_hat - y) ^ 2)

# Define predictor (x cols).
feature_cols <- colnames(X_train)

# Target column (motor_UPDRS)
target_col <- colnames(train_scaled[5])

# Build formula. -1 to remove the intercept. because we scaled the data, so would be ~0.
formula <- as.formula(paste(target_col, "~", paste(feature_cols, collapse = " + "), "-1"))

# Fit Ordinary Least Squares on training data.
model <- lm(formula, data = train_scaled)

# Run summary to get estimates, standard errors, t-stats and p-values.
model_summary <- summary(model)
print(model_summary)

# Extract the estimated coefficient vector (theta)
theta <- model_summary$coefficients[, 1]

# Extract the residual standard deviation estimate
sigma = model_summary$sigma

# Make in-sample and out-of-sample predictions on train and test data.
train_pred <- predict(model, train_scaled)
test_pred <- predict(model, test_scaled)

# Compute and print MSE for train and test sets.
# Test data tests unseen data.
print(paste(
  "MSE on the training data:",
  mse(train_scaled$motor_UPDRS, train_pred)
))
print(paste("MSE on the test data:", mse(test_scaled$motor_UPDRS, test_pred)))

# ----3.----

# Log likelihood
# Using the Gaussian linear model without intercept
# sigma is standard deviation, hence log(sigma).
log_likelihood <- function(theta, sigma) {
  n <- length(y_train) # number of training observations
  
  # compute residuals r = y-x0 and plug into Gaussian log likelihood formula.
  ll <- -n / 2 * log(2 * pi) - n * log(sigma) - (1 / (2 * sigma ^ 2)) * sum((y_train -
                                                                               as.matrix(X_train) %*% theta) ^
                                                                              2)
  return(ll)
}

# Ridge. Negative log likelihood. Minimize the function.
# 
ridge <- function(theta, sigma, lambda) {
  ridge <- -log_likelihood(theta, sigma) + lambda * sum(theta ^ 2)
  return(ridge)
}

# Results interpretation/notes for oral defense:
# Model good? It's an okay baseline. R^2=0.12 (variation). 
# Train/test MSE are relatively close (0.88 vs 0.94) so little overfitting.
# Interpreting the table: estimate = slope. SE=uncertainty. t=Estimate/SE.
# p tests hypothesis test. Large t + small p = evidence the feature matters
# R² Close value means little overfitting.
# Ridge adds L2 Penalty to shrink coefficients.
# A small p-value means data provide strong evidence against 

# Notes after oral defense:
# Hyper parameter lambda is used in the next function
# Using Gaussian model is an arbitrary choice

# Ridge opt.
ridge_opt <- function(lambda) { # make lambda an input (hyperparameter)
  objective_function <- function(params) { 
    # theta is the coefficients
    theta <- params[1:(length(params) - 1)]
    
    # log_sigma optimizes over log sigma.
    # using log to avoid negative sigma.
    log_sigma <- params[length(params)]
    sigma <- exp(log_sigma)
    
    return(ridge(theta, sigma, lambda))
  }
  
  # run BFGS optimization. 16 parameters+1 element for log(sigma)
  return(optim(
    par=rep(0,17),
    fn=objective_function,
    method="BFGS"
  ))
}

# DF (calculate effective degrees of freedom using the ridge smoother)
df = function(lambda) {
  # X is a numeric matrix
  X <- as.matrix(X_train)
  
  # build the ridge smoother matrix
  P <- X %*% solve(t(X) %*% X + lambda * diag(ncol(X))) %*% t(X)
  
  # return the effective degrees of freedom of the ridge.
  return(sum(diag(P)))
}

# ----4---
# Task: Compute optimal theta parameters for lambda=1,100,1000.
for (lambda in c(1,100,1000))
{
  # calls the optimizer for a given lambda.
  # -17 to drop the last element (our log(sigma))
  theta_opti <- unlist(ridge_opt(lambda)$par[-17])
  
  # perform matrix multiplication to get ŷ on train and test.
  y_hat_train <- as.matrix(X_train) %*% theta_opti
  y_hat_test <- as.matrix(X_test) %*% theta_opti
  
  print(paste("MSE for lambda =", lambda, "on training data:", mse(y_train, y_hat_train)))
  print(paste("MSE for lambda =", lambda, "on test data:",    mse(y_test,  y_hat_test)))
}

# print the results
for (lambda in c(1, 100, 1000)) {
  print(paste("DF for lambda =", lambda, ":", df(lambda)))
}

# Results for lambda=1,100,1000 is 13.8, 9.9, and 5.6 (decreasing.
# Interpretation of results: effective degrees of freedom decrease as lambda grows.
