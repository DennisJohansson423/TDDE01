library(kknn)

data <- read.csv("optdigits.csv")
str(data)

n <- nrow(data)

## Task 1
# Split 50/25/25
set.seed(12345)
id <- sample(1:n, floor(n*0.5))
train <- data[id, ]

id1 <- setdiff(1:n, id)
set.seed(12345)
id2 <- sample(id1, floor(n*0.25))
valid <- data[id2, ]

id3 <- setdiff(id1, id2)
test <- data[id3, ]

## Task 2
# Identify label (last col)
names(train)[ncol(train)] <- "label"
names(valid)[ncol(valid)] <- "label"
names(test)[ncol(test)] <- "label"

train$label <- factor(train$label)
valid$label <- factor(valid$label)
test$label <- factor(test$label)

form <- label~.

# Train
m1 <- kknn(form, train, train, k=30, kernel="rectangular")
pred_train <- m1$fitted.values

# Test
m2 <- kknn(form, train, test, k=30, kernel="rectangular")
pred_test <- m2$fitted.values

# Confusion matrices
cm_train <- table(True = train$label, Pred = pred_train)
cm_test <- table(True = test$label, Pred = pred_test)

# Misclassification error
err_train <- mean(pred_train != train$label)
err_test <- mean(pred_test != test$label)
acc_train <- 1 - err_train
acc_test <- 1 - err_test

cm_test
round(prop.table(cm_test, 1), 2)
"
Test accuracy about 94.1% and train accuracy about 95.8%. 
0 is classified as perfect, 6 and 7 are almost perfect.
The digits that are most often confused are 4-7, 5-9.
"

## Task 3
probs_train <- m1$prob

# Find the 8:s
is8 <- train$label == 8
train8 <- train[is8, ]
probs8 <- probs_train[is8, ]

# Confidence for 8
conf8 <- probs8[, "8"]

# The 2 easiest and 3 hardest 
easy_idx <- order(conf8, decreasing = TRUE)[1:2]
hard_idx <- order(conf8, decreasing = FALSE)[1:3]

# Plot the heatmap
plot_digit <- function(rowvec, main = "") {
  x <- as.numeric(rowvec[1:64])
  m <- matrix(x, nrow = 8, ncol = 8, byrow = TRUE)
  m <- m[8:1, ]
  pal <- heat.colors(20)
  heatmap(m,
          Rowv = NA, Colv = NA, dendrogram = "none", scale = "none",
          col = pal, labRow = 1:8, labCol = 1:8, main = main, margins = c(3,3))
}

# Plot 2 easiest
par(mfrow = c(1, 2))
plot_digit(train8[easy_idx[1], ], main = sprintf("Easiest 8 (1)", conf8[easy_idx[1]]))
plot_digit(train8[easy_idx[2], ], main = sprintf("Easiest 8 (2)", conf8[easy_idx[2]]))
"
1 is not that easy to recognize visualy but 2 is easy to recognize visualy.
"

# Plot 3 hardest
par(mfrow = c(1, 3))
plot_digit(train8[hard_idx[1], ], main = sprintf("Hardest 8 (1)", conf8[hard_idx[1]]))
plot_digit(train8[hard_idx[2], ], main = sprintf("Hardest 8 (2)", conf8[hard_idx[2]]))
plot_digit(train8[hard_idx[3], ], main = sprintf("Hardest 8 (3)", conf8[hard_idx[3]]))
"
All 3 are hard to recognize visualy.
"

## Task 4
k_values <- 1:30
train_err <- numeric(length(k_values))
valid_err <- numeric(length(k_values))

for(i in seq_along(k_values)){
  k <- k_values[i]
  
  # Train predictions
  m_train_k <- kknn(form, train, train, k=k, kernel = "rectangular")
  train_err[i] <- mean(m_train_k$fitted.values != train$label)
  
  # Validation predictions
  m_valid_k <- kknn(form, train, valid, k=k, kernel = "rectangular")
  valid_err[i] <- mean(m_valid_k$fitted.values != valid$label)
}

# Plot training and validation errors
plot(k_values, train_err, type="b", pch=19, col="blue",
     ylim = range(c(train_err, valid_err)),
     xlab = "K number of neighbours",
     ylab = "Missclassification error",
     main = "Training and validation errors vs K")
lines(k_values, valid_err, type="b", pch=19, col="red")
legend("topright", inset = c(-0.25, 0),  # negative x inset pushes it outside
       legend = c("Training error", "Validation error"),
       col = c("blue", "red"), pch = 19, lty = 1, bty = "n")

# Find the optimal K
opt_k <- k_values[which.min(valid_err)]
opt_k

# Calculate the error with optimal K
m_train_opt <- kknn(form, train, train, k = opt_k, kernel = "rectangular")
m_valid_opt <- kknn(form, train, valid, k = opt_k, kernel = "rectangular")
m_test_opt <- kknn(form, train, test, k = opt_k, kernel = "rectangular")

err_train_opt <- mean(m_train_opt$fitted.values != train$label)
err_valid_opt <- mean(m_valid_opt$fitted.values != valid$label)
err_test_opt <- mean(m_test_opt$fitted.values != test$label)

round(c(K = opt_k,
        train_error = 1 - err_train_opt,
        valid_error = 1 - err_valid_opt,
        test_error = 1 - err_test_opt), 4)
"
The optimal K is 7.
train_error = 98.22%
valid_error = 97.28%
test_error = 96.13%
The training error is lowest, as expected, 
but the validation and test errors are close in magnitude, 
indicating that the model generalizes well to unseen data.
The small gap between training and validation errors suggests only mild overfitting.
"

## Task 5
cross_ent <- numeric(length(k_values))

for(i in seq_along(k_values)){
  k <- k_values[i]
  
  # Predict valid
  m_valid <- kknn(form, train, valid, k=k, kernel = "rectangular")
  probs_valid <- m_valid$prob
  y_true <- valid$label
  eps <- 1e-15
  
  Y <- model.matrix(~y_true - 1)
  
  # Cross-entropy
  ce <- -mean(rowSums(Y*log(probs_valid + eps)))
  cross_ent[i] <- ce
}

# Plot cross-entropy vs K
plot(k_values, cross_ent, type="b", pch=19, col="blue",
     xlab="K number of neighbours",
     ylab="Cross-entropy (validation set)",
     main="Validation cross-entropy vs K")

# Find optimal K
opt_k_ce <- k_values[which.min(cross_ent)]
opt_k_ce
"
Optimal k is 8.
Because the response is multinomial, 
cross-entropy aligns with the negative log-likelihood of that model. 
Unlike misclassification error, it also accounts for prediction confidence 
and penalizes overconfident mistakes, making it a more appropriate measure 
for probabilistic classifiers.
"







