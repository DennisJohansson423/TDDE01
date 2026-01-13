library(tree)

### Task 1

# Load data
bank <- read.csv2("bank-full.csv")
bank$duration <- NULL
bank <- data.frame(lapply(bank, function(x)
  if (is.character(x)) factor(x) else x))

# Split train/valid/test into 40/30/30
n <-dim(bank)[1]
set.seed(12345)
id <- sample(1:n, floor(n*0.4))
train <- bank[id,]

id1 <- setdiff(1:n, id)
set.seed(12345)
id2 <- sample(id1, floor(n*0.3))
valid <- bank[id2,]

id3 <- setdiff(id1,id2)
test <- bank[id3,]

### Task 2

## a)

# Default decision tree
treeA <- tree(y~., data = train)

# Predictions
predA_train <- predict(treeA, newdata = train, type = "class")
predA_valid <- predict(treeA, newdata = valid, type = "class")

# Miscalssifications
errA_train <- mean(predA_train != train$y)
errA_valid <- mean(predA_valid != valid$y)
errA_train
errA_valid

## b)

# Smallest allowed node size equal to 7000
treeB <- tree(y~., data = train,
              control = tree.control(n = nrow(train), minsize = 7000))

# Predictions
predB_train <- predict(treeB, newdata = train, type = "class")
predB_valid <- predict(treeB, newdata = valid, type = "class")

# Miscalssifications
errB_train <- mean(predB_train != train$y)
errB_valid <- mean(predB_valid != valid$y)
errB_train
errB_valid

## c)

# Minimum deviance to 0.0005
treeC <- tree(y~., data = train,
              control = tree.control(n = nrow(train), mindev = 0.0005))

# Predictions
predC_train <- predict(treeC, newdata = train, type = "class")
predC_valid <- predict(treeC, newdata = valid, type = "class")

# Miscalssifications
errC_train <- mean(predC_train != train$y)
errC_valid <- mean(predC_valid != valid$y)
errC_train
errC_valid

# Plot the trees
plot(treeA)
text(treeA, pretty = 0, label = "yprob", digits = 2)
treeA         
summary(treeA)

plot(treeB)
text(treeB, pretty = 0, label = "yprob", digits = 2)
treeB         
summary(treeB)

plot(treeC)
text(treeC, pretty = 0, label = "yprob", cex = 0.4, digits = 2)
treeC         
summary(treeC)

"
Tree C fits the training data a bit better, but the validation error 
is essentially the same as for A and B. Since Tree B is the simplest 
model with the same validation performance, it is the best choice.

Increasing the minimum node size (Tree B) restricts how finely the 
tree can split the data, which leads to a smaller tree with similar 
performance. Decreasing the minimum deviance (Tree C) lets the 
algorithm keep splitting even when the improvement is very small, 
producing a much larger tree that fits the training data better but 
does not generalise better to the validation set.
"

### Task 3

# Define number of leaves
sizes <- 2:50
train_dev <- numeric(length(sizes))
valid_dev <- numeric(length(sizes))

# Prune treeC and compute deviance
for(i in seq_along(sizes)){
  k <- sizes[i]
  
  # Prune treeC
  pruned_k <- prune.tree(treeC, best = k)
  
  train_dev[i] <- deviance(pruned_k)
  
  pruned_valid <- predict(pruned_k, newdata = valid, type = "tree")
  
  valid_dev[i] <- deviance(pruned_valid)
}

# Plot training vs validation deviance
ylim_all <- range(c(train_dev, valid_dev), na.rm = TRUE)

plot(sizes, train_dev, type = "l",
     xlab = "Number of leaves",
     ylab = "Deviance",
     main = "Training and validation deviance vs tree size",
     ylim = ylim_all,
     col = "red")
lines(sizes, valid_dev, col = "blue")
legend("topright", legend = c("Training", "Validation"), 
       col = c("red", "blue"), lty = 1)

# Find optimal amount of leaves
opt_k <- sizes[which.min(valid_dev)]
opt_k

# Plot the optimal tree
opt_tree <- prune.tree(treeC, best = opt_k)
summary(opt_tree)
plot(opt_tree)
text(opt_tree, pretty = 0, label = "yprob", cex = 0.4, digits = 2)

"
The optimal tree has 22 leaves. The root split is on poutcome (previous
campaign outcome), so this is the most important variable: customers
with failure/other/unknown outcomes behave differently from those with
successful outcomes. For the non-successful group, the next key splits
are on month and contact type, showing that timing of the call and
whether the contact channel is unknown strongly affect the subscription
probability.

Deeper in the tree, pdays (days since last contact), age, day of month,
balance, housing loan status and job type refine the predictions. In
general, clients contacted long after the previous campaign, with higher
balances and no housing loan, tend to have higher subscription
probabilities than recently contacted clients with low balances and a
housing loan.
"

### Task 4

# Predict test using opt_tree
pred_test <- predict(opt_tree, newdata = test, type = "class")

#Confusion matrix
cm <- table(Actual = test$y, Predicted = pred_test)
cm

# Accuracy for the prediction
accuracy <- mean(pred_test == test$y)
accuracy

# Extract TP, FP, FN, TN
TP <- cm["yes", "yes"]
FP <- cm["no", "yes"]
FN <- cm["yes", "no"]
TN <- cm["no", "no"]

# Calculate F1-score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
F1 <- 2 * precision * recall / (precision + recall)
F1

"
The test confusion matrix shows that the model achieves an accuracy 
of ≈0.891, which is only slightly better than a trivial 
classifier that always predicts “no” (≈0.883). Thus, in terms of 
accuracy, the model does not give a huge improvement over the baseline.
At the same time, the model detects some positive cases: 
the precision for the class “yes” is fairly high (≈0.67), 
but the recall is low (≈0.14), giving an F1-score of ≈0.22. 
This means the model finds only a small fraction of the actual 
subscribers and therefore has limited predictive power for the 
positive class. Because the data set is highly imbalanced, 
accuracy is not very informative here; the F1-score is more 
appropriate, since it focuses on the minority “yes” class 
and combines both precision and recall.
"

### Task 5

# Define the loss matrix
lvl <- levels(train$y)

loss <- matrix(0, nrow = 2, ncol = 2,
               dimnames = list(lvl, lvl))

loss["yes", "no"] <- 5
loss["no", "yes"] <- 1
loss

# Fit a new tree with loss
tree_loss <- tree(y~., data = train, 
                  control = tree.control(n = nrow(train), mindev = 0.0005),
                  loss = loss)

# Prediction
pred_test_loss <- predict(tree_loss, newdata = test, type = "class")

#Confusion matrix
cm_loss <- table(Actual = test$y, Predicted = pred_test_loss)
cm_loss

"
With the loss matrix that penalises false negatives more heavily 
than false positives, the tree is encouraged to avoid missing “yes” 
cases. As a consequence, it predicts “yes” more often, which 
increases the number of true positives and reduces the number of 
false negatives, but also increases the number of false positives. 
The overall accuracy stays almost the same, but the F1-score 
increases noticeably, which means the model is better at finding 
the actual subscribers when we take both precision and recall 
into account.
"

### Task 6

# Logistic regression on training
logistic_model <- glm(y~., data = train, family = binomial)

# Prob of "yes" from optimal tree
p_tree_test <- predict(opt_tree, newdata = test, type = "vector")[, "yes"]


# Prob of "yes" from logistic regression
p_logi_test <- predict(logistic_model, newdata = test, type = "response")

y_true <- ifelse(test$y == "yes", 1, 0)

# Create TPR/FPR with pi
pi_vals <- seq(0.05, 0.95, by = .05)
TPR_tree <- numeric(length(pi_vals))
FPR_tree <- numeric(length(pi_vals))
TPR_logi <- numeric(length(pi_vals))
FPR_logi <- numeric(length(pi_vals))

# Loop over thresholds pi to compute TPR/FPR
for(i in seq_along(pi_vals)){
  pi <- pi_vals[i]
  
  ## Tree
  
  # Pred for tree
  pred_tree <- ifelse(p_tree_test > pi, "yes", "no")
  pred_tree <- factor(pred_tree, levels = levels(test$y))
  
  # Confusion matrix
  cm_tree <- table(Actual = test$y, Predicted = pred_tree)
 
  # Extract TP, FP, FN, TN
  TP <- cm_tree["yes", "yes"]
  FP <- cm_tree["no", "yes"]
  FN <- cm_tree["yes", "no"]
  TN <- cm_tree["no", "no"]
  
  # Compute TPR/FPR for tree
  TPR_tree[i] <- TP / (TP + FN)
  FPR_tree[i] <- FP / (FP + TN)
  
  ## Logistic
  
  # Pred for logi
  pred_logi <- ifelse(p_logi_test > pi, "yes", "no")
  pred_logi <- factor(pred_logi, levels = levels(test$y))
  
  # Confusion matrix
  cm_logi <- table(Actual = test$y, Predicted = pred_logi)
  
  # Extract TP, FP, FN, TN
  TP <- cm_logi["yes", "yes"]
  FP <- cm_logi["no", "yes"]
  FN <- cm_logi["yes", "no"]
  TN <- cm_logi["no", "no"]
  
  # Compute TPR/FPR for logi
  TPR_logi[i] <- TP / (TP + FN)
  FPR_logi[i] <- FP / (FP + TN)
}

# Plot the ROC curves
plot(FPR_tree, TPR_tree, type = "b",
     xlab = "False positive rate",
     ylab = "True positive rate",
     main = "ROC curves: Tree vs Logistic",
     col = "red")
lines(FPR_logi, TPR_logi, type = "b", col = "blue")
abline(0, 1, lty = 2)
legend("bottomright", legend = c("Tree", "Logistic", "Random guessing"),
       col = c("red", "blue", "black"), lty = c(1, 1, 2), pch = c(1, 1, NA))

"
The ROC curves for both models lie clearly above the diagonal,
so both the decision tree and the logistic regression are better
than random guessing. The ROC curve for the decision tree is
mostly above the curve for the logistic regression, which means
that for the same FPR the tree generally achieves a higher TPR.
So the tree has slightly better discriminative ability on this
data set, although the difference is not huge.

In this problem the positive class (“yes”) is rare, most
observations are “no”. With such class imbalance, the ROC curve
can look optimistic because the FPR uses the large number of
negatives in the denominator, so even many false positives can
correspond to a small FPR. What we really care about here is how
well the model finds the few “yes” cases. A precision–recall
curve focuses on the positive class by plotting precision versus
recall and is therefore more sensitive to changes in performance
on the minority class. For this kind of imbalanced data, a
precision–recall curve usually gives a more informative picture
than the ROC curve.
"

