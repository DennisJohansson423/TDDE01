library(neuralnet)

### Task 1
set.seed(1234567890)

# Generate random x-values
Var <- runif(500, 0, 10)

# Compute sin
Sin <- sin(Var)
mydata <- data.frame(Var, Sin)

# Split i first 25 to training and the rest to test
tr <- mydata[1:25, ] # Training
te <- mydata[26:500, ] # Test

# Random initialization of the weights in the interval [-1, 1]
winit <- runif(31, -1, 1)

# Train neural network
nn <- neuralnet(Sin ~ Var, data = tr, hidden = 10,
                linear.output = TRUE,
                startweights = winit)

# Plot of the training data (black), test data (blue), and predictions (red)
plot(tr, cex = 2, main = "Default logistic activation")
points(te, col = "blue", cex = 1)
points(te[,1], predict(nn, te), col = "red", cex = 1)
legend("bottomleft", legend = c("Training", "Test", "Prediction"),
       col = c("black", "blue", "red"), pch = 19)

"
The neural network with one hidden layer of 10 hidden units gives a 
very good approximation of sin(x) on the interval [0,10]. 
In the plot we see that the red prediction points lie almost on top 
of the blue test points along the whole curve, even though the 
network was trained on only 25 observations (black points). 
This shows that the model has successfully captured the periodic 
pattern of the sine function and generalises well within the 
training interval, with only small local deviations that are expected 
given the limited training data and random weight initialisation.
"


### Task 2

# Linear
act_linear <- function(x) x

# ReLU
act_relu <- function(x) x * (1 / (1 + exp(-10 * x))) # Crashes with pmax(0, x)

# Softplus
act_softplus <- function(x) log(1 + exp(x))

# Help function for the plots
plot_nn_result <- function(nn, tr, te, main_title){
  plot(tr, cex = 2, main = main_title, xlab = "Var", ylab = "Sin")
  points(te, col = "blue", cex = 1)
  points(te[,1], predict(nn, te), col = "red", cex = 1)
  legend("bottomleft",
         legend = c("Training", "Test", "Prediction"),
         col = c("black", "blue", "red"), pch = 19)
}

## Train all 3 networks

# Linear
nn_lin <- neuralnet(Sin ~ Var, data = tr, hidden = 10,
                    act.fct = act_linear,
                    linear.output = TRUE)
plot_nn_result(nn_lin, tr, te, "Hidden activation: Linear")

# ReLU
nn_relu <- neuralnet(Sin ~ Var, data = tr, hidden = 10,
                    act.fct = act_relu,
                    linear.output = TRUE)
plot_nn_result(nn_relu, tr, te, "Hidden activation: ReLU")

# Softplus
nn_softplus <- neuralnet(Sin ~ Var, data = tr, hidden = 10,
                         act.fct = act_softplus,
                         linear.output = TRUE)
plot_nn_result(nn_softplus, tr, te, "Hidden activation: Softplus")

"
Linear activation cannot capture the nonlinear sine wave and 
produces an almost constant prediction. ReLU captures the overall 
shape much better, but the resulting curve is piecewise linear with 
some sharp corners. Softplus gives a smooth approximation that 
follows the sine function very closely, indicating that smooth 
nonlinear activations (logistic/softplus) work best for this problem.
"


### Task 3

# Generate new random x-values
set.seed(1234567890)
Var_big <- runif(500, 0, 50)

# Compute new sin
Sin_big <- sin(Var_big)
data_big <- data.frame(Var = Var_big, Sin = Sin_big)

# Prediction
pred_big <- as.vector(predict(nn, data_big))

# Plot the prediction
y_min <- min(data_big$Sin, pred_big)
y_max <- max(data_big$Sin, pred_big)

plot(data_big$Var, data_big$Sin,
     col = "blue", pch = 19, cex = 0.7,
     xlab = "Var", ylab = "Sin",
     main = "NN from Task 1 applied on [0, 50]")
points(data_big$Var, pred_big,
       col = "red", pch = 19, cex = 0.7)
legend("topright", 
       legend = c("True sin(x)", "Prediction"),
       col = c("blue", "red"), pch = 19)

plot(data_big$Var, data_big$Sin,
     col = "blue", pch = 19, cex = 0.7,
     xlab = "Var", ylab = "Sin",
     main = "NN from Task 1 applied on [0, 50]",
     ylim = c(y_min, y_max))
points(data_big$Var, pred_big,
       col = "red", pch = 19, cex = 0.7)
legend("bottomleft", 
       legend = c("True sin(x)", "Prediction"),
       col = c("blue", "red"), pch = 19)

"
In Task 3 we used the network from Task 1, which was trained only 
on inputs in interval [0,10], to predict sin(x) for new inputs in the 
interval [0,50]. In the plot we see that the predictions follow the 
true sine values reasonably well for x between 0 and 10, i.e. inside 
the training range. For larger x the predictions no longer track the 
oscillating sine function and instead stay close to an almost 
constant value. This shows that the network generalises well within 
the range where it has seen data, but it fails to extrapolate outside 
this interval and therefore gives poor predictions for x > 10.
"


### Task 4

nn$weights

"
In Task 4 we inspected nn$weights. The input–hidden weights [[1]][[1]]
show that each hidden unit computes a linear function of the input, 
z_i(x) = b_i + w_i x, followed by a logistic activation. Several of 
the weights w_i have fairly large magnitude (for example 4.04) and 
the corresponding biases are also large (e.g. –11.87). This means 
that for inputs x larger than the training range (x > 10), the 
pre-activations z_i(x) become very large positive or negative numbers, 
so the logistic units saturate and their outputs are almost exactly 
0 or 1. The hidden–output weights [[1]][[2]] then form a linear 
combination of these saturated hidden units. Since the hidden layer 
is essentially constant for large x, the output of the network also 
becomes almost constant. This explains why in Task 3 the predictions 
follow the sine curve reasonably well on [0,10], but for larger inputs 
they stop oscillating and instead converge to a nearly constant value.
"


### Task 5

# Generate random x-values
set.seed(1234567890)
Var2 <- runif(500, 0, 10)

# Compute sin
Sin2 <- sin(Var2)
mydata2 <- data.frame(Var = Var2, Sin = Sin2)

# Training
tr2 <- mydata2

# Train neural network
nn_inv <- neuralnet(Var ~ Sin, data = tr2, hidden = 10,
                linear.output = TRUE,
                threshold = 0.1)

# Prediction
pred_inv <- as.vector(predict(nn_inv, tr2))

# Plot of the training data (black), test data (blue), and predictions (red)
plot(tr2$Sin, tr2$Var, 
     col = "blue", pch = 19, cex = 0.7,
     xlab = "sin(x)", ylab = "x", 
     main = "Trying to learn x from sin(x)")
points(tr2$Sin, pred_inv, col = "red", pch = 19, cex = 0.7)
legend("bottomleft", legend = c("True x", "Predicted x"),
       col = c("blue", "red"), pch = 19)

"
In Task 5 we reversed the roles of input and output and trained a 
neural network to predict x from sin(x) on [0,10]. The blue points 
form several vertical branches for the same sin(x)-values, which 
shows that the mapping x -> sin(x) is not one-to-one on this interval: 
one sin(x) value corresponds to several different x. The network, 
however, must represent a single-valued function x = f(sin(x)). 
As a result, the red predictions form one continuous curve that cuts 
through the blue branches and often lies somewhere in between them. 
The model therefore gives a poor fit and effectively learns something 
like an “average” x for each sin(x), illustrating that we cannot 
reliably learn the inverse of sin(x) on this interval with this 
kind of neural network.
"
