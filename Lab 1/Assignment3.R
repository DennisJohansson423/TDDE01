#1

#Install packages & Use libraries
install.packages("ggplot2")
install.packages('RcolorBrewer')
library(ggplot2)
library(RColorBrewer)

#Make csv to a data frame
pima_data <- read.csv("pima-indians-diabetes.csv", header = FALSE)

#Name columns
colnames(pima_data)[1:9] <- c("Pregnancies", "PlasmaGlucose",
                              "BloodPressure", "TricepThickness",
                              "Insulin", "BMI", "DiabetesPedigree",
                              "Age", "Diabetes")

#Extract needed column data
age <- pima_data[,8]
plasma_glucose <- pima_data[,2]
data_to_color <- pima_data[,7]

#Assign colors & different shades to dots
num_shades <- 10
color_groups <- cut(data_to_color, breaks = 20)
color_palette <- brewer.pal(num_shades, "Reds")

#Plot
plot(x = age, y = plasma_glucose, xlab = "Age",
     ylab = "Plasma glucose concentration",
     main = "Age vs Plasma glucose concentration",
     sub = "Color: Darker means higher diabetes degree",
     col = color_palette[color_groups], pch = 16)

#Question:
# Do you think that Diabetes is easy 
# to classify by a standard logistic regression
# model that uses these two variables as 
# features?
#Answer:
# Partially, the only main indication I can see is that
# Plasma glucose levels over around "75" start having higher
# diabetes degrees, but there are all kinds of cases for all
# plasma levels and ages so it's still hard to classify.



#2

#Use libraries
library(tidyr)
library(dplyr)

#Create training set
train=pima_data%>%select(PlasmaGlucose, Age, Diabetes)

#Train a logistic regression model
m1=glm(as.factor(Diabetes)~., train, family = "binomial")

#Get summary with info about the model
summary(m1)

#Equation of the probabilistic model:
# P(Diabetes = 1) = 1 / (1+e^(-((Intercept=-5.912449)+
#      0.035644*PlasmaGlucose+0.024778*Age)))
# ->
# P(Diabetes = 1 ) = 1 / (1+e^(5.912449
#      -0.035644*PlasmaGlucose-0.024778*Age))

#Training misclassification error calcs:

#Create prob
Prob=predict(m1, type="response")

#Create pred
Pred=ifelse(Prob>0.5, "Diabetes", "No diabetes")

#Create confusion matrix
confusion_matrix <- table(train$Diabetes, Pred)

#Incorrect predictions (from table above,
#anti-diagonal elements):

#False positives:
false_positives <- confusion_matrix["0","Diabetes"]
false_negatives <- confusion_matrix["1","No diabetes"]

#Total incorrect predictions:
total_incor_preds <- false_positives + false_negatives

#Total observations:
total_obs <- sum(confusion_matrix)

#Misclassification error:
misclass_error <- (total_incor_preds/total_obs)

#Show the misclassification error:
misclass_error #=0.2630208 = 26.3%

#Scatter plot:

#Color according to pred (Diabetes or not):
data_to_color2 <- Pred
data_to_color2 <- factor(data_to_color2,
    levels = c("No diabetes", "Diabetes") )

#Assign colors & different shades to dots
color_palette <- c("lightcoral", "darkred")

#Plot
plot(x = age, y = plasma_glucose, xlab =
       "Age", 
     ylab = "Plasma Glucose concentration",
     main = "Age vs Plasma glucose concentration",
     sub = "Color: Darker means diabetes, lighter means no diabetes.",
     col = color_palette[data_to_color2], pch = 16)

#Quality of the new classification:
# Now it classifies 50% as diabetes (darkred) and 50% as no diabetes (lightred).
# The dominant factor is plasma glucose concentration as it's
# coefficient has a higher value than age.
# High PlasmaGlucose levels are more associated with diabetes
# according to the model regardless of age, but there
# is some overlap for medium PlasmaGlucose levels.



#3

#Coefficients:
coefficients <- coef(m1)

#Extract coefficients:
#Intercept:
beta0 <- coefficients[1]
#Coef for Feature 1 = PlasmaGlucose
beta1 <- coefficients[2]
#Coef for Feature 2 = Age
beta2 <- coefficients[3]

#Intercept
line_intercept <- -beta0 / beta1

#Slope
line_slope <- -beta2 / beta1

#Plot
plot(x = age, y = plasma_glucose, xlab =
       "Age", 
     ylab = "Plasma Glucose concentration",
     main = "Age vs Plasma glucose concentration",
     sub = "Color: Darker means diabetes, lighter means no diabetes.",
     col = color_palette[data_to_color2], pch = 16)

#Show decision boundary line
abline(a = line_intercept,
       b = line_slope,
       col = "black",
       lwd = 2,
       lty = 2)

print("Decision boundary parts below:")
beta0
beta1
beta2

#Function for the decision boundary
#(solving for Age aka X2):
# y = - (beta0/beta1) - (beta2/beta1) * x
# y = - (Intercept/PlasmaGlucose) - (Age/PlasmaGlucose) * x

#Comment on if the boundary seems to catch the
#data distribution well:
# Yes, I'd say it catches well, as there are no full
# circles of different colors on both sides of the line,
# just some of both on the line which makes it a bit
# sketchy.



#4

#0.2 case:
#Create pred with r = 0.2
Pred=ifelse(Prob>0.2, "Diabetes", "No diabetes")

#0.8 case:
#Create pred with r=0.8
#Pred=ifelse(Prob>0.8, "Diabetes", "No diabetes")

#Create confusion matrix
confusion_matrix <- table(train$Diabetes, Pred)

#Incorrect predictions (from table above,
#anti-diagonal elements):

#False positives:
false_positives <- confusion_matrix["0","Diabetes"]
false_negatives <- confusion_matrix["1","No diabetes"]

#Total incorrect predictions:
total_incor_preds <- false_positives + false_negatives

#Total observations:
total_obs <- sum(confusion_matrix)

#Scatter plot:

#Color according to pred (Diabetes or not):
data_to_color2 <- Pred
data_to_color2 <- factor(data_to_color2,
                         levels = c("No diabetes", "Diabetes") )

#Assign colors & different shades to dots
color_palette <- c("lightcoral", "darkred")

#Plot
plot(x = age, y = plasma_glucose, xlab =
       "Age", 
     ylab = "Plasma Glucose concentration",
     main = "Age vs Plasma glucose concentration",
     sub = "Color: Darker means diabetes, lighter means no diabetes.",
     col = color_palette[data_to_color2], pch = 16)

#Comment on prediction when r value changes:
# If the r-value changes, the amount of
# dots of both colors in the scatterplot
# changes. If r=0.8, the first 80% (from left
# to right) will have diabetes, if r=0.2,
# the first 20% will have diabetes, etc.



#5

x1 <- train$PlasmaGlucose
x2 <- train$Age
y <- train$Diabetes

train$z1 <- x1^4
train$z2 <- x1^3 * x2
train$z3 <- x1^2 * x2^2
train$z4 <- x1 * x2^3
train$z5 <- x2^4

#Train a new logistic regression model
m2=glm(as.factor(y)~ x1 + x2 + z1 + z2 + z3 + z4 + z5, train, family = "binomial")

#Create prob
Prob=predict(m2, type="response")

#Create pred
Pred=ifelse(Prob>0.5, "Diabetes", "No diabetes")

#Create confusion matrix
confusion_matrix <- table(train$Diabetes, Pred)

#Incorrect predictions (from table above,
#anti-diagonal elements):

#False positives:
false_positives <- confusion_matrix["0","Diabetes"]
false_negatives <- confusion_matrix["1","No diabetes"]

#Total incorrect predictions:
total_incor_preds <- false_positives + false_negatives

#Total observations:
total_obs <- sum(confusion_matrix)

#Misclassification error:
misclass_error <- (total_incor_preds/total_obs)

#Show the misclassification error:
misclass_error #=0.2447917 = around 24.5%

#Scatter plot:

#Color according to pred (Diabetes or not):
data_to_color2 <- Pred
data_to_color2 <- factor(data_to_color2,
                         levels = c("No diabetes", "Diabetes") )

#Assign colors & different shades to dots
color_palette <- c("lightcoral", "darkred")

#Plot
plot(x = age, y = plasma_glucose, xlab =
       "Age", 
     ylab = "Plasma Glucose concentration",
     main = "Age vs Plasma glucose concentration",
     sub = "Color: Darker means diabetes, lighter means no diabetes.",
     col = color_palette[data_to_color2], pch = 16)

#Question:
# What can you say about the quality of this
# model compared to the previous logistic 
# regression model?
#Answer:
# This model is more complex, and seems to color
# the data in a more realistic way, by taking
# into account old age more (60+ in this case),
# as people of older age have higher plasma glucose
# levels due to age, and it doesnt have to mean they
# have diabetes. But people that are around 35-55 years
# old with high levels are more likely to actually
# have diabetes, which makes sense as they should
# function better normally.

#Question:
# How have the basis
# expansion trick affected the shape of 
# the decision boundary and the prediction
# accuracy?
#Answer:
# The shape has changed from a linear first degree function
# separation, to a second degree function separation, with
# the key difference being that older age with no diabetes
# has taken over more dots in the mid-right part of the plot.
# The misclassification rate has gone down about 1.8% (0.18),
# so it classifies about 75.5% correct now, as opposed to
# 73.7% before.

