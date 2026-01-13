#Lab3 Assignment 3. Support Vector Machines

#-----Task 1-----

#Question:
#Which filter do you return to the user?
#filter0, filter1, filter2 or filter3? Why?

#Answer:
#We return filter2 to the user, as it is the only
#filter trained on combined training and validation
#data sets (trva). In this way we give our model
#as much data as possible, before using the sacred
#test set (te).


#-----Task 2-----

#Question:
#What is the estimate of the generalization
#error of the filter returned to the user?
#err0, err1, err2 or err3? Why?

#Answer:
#filter2 has generalization error: err2 = 0.1498127
#It is estimated on the test set (te), as te has
#not been used before, so it becomes more
#unbiased in that way. The test set cant be
#used in training, or in hyperparam selection.

#Why the other errors are bad:
#err0 is trained on tr and tested on va, best
#validation error, not generalization error.

#err1 is trained only on tr aka not the final
#model trva, and tested on te, so it
#underestimates and is worse than err2.

#err3 is biased and bad, filter3 is trained on
#spam = tr + va + te, which is the whole dataset.
#This is called data leakage, as the model is
#tested on data it was trained on. False performance.


#-----Task 3-----

library(kernlab)
set.seed(1234567890)

data(spam)
foo <- sample(nrow(spam))
spam <- spam[foo,]
tr <- spam[1:3000, ]
va <- spam[3001:3800, ]
trva <- spam[1:3800, ]
te <- spam[3801:4601, ] 

by <- 0.3
err_va <- NULL
for(i in seq(by,5,by)){
  filter <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=i,scaled=FALSE)
  mailtype <- predict(filter,va[,-58])
  t <- table(mailtype,va[,58])
  err_va <-c(err_va,(t[1,2]+t[2,1])/sum(t))
}

filter0 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter0,va[,-58])
t <- table(mailtype,va[,58])
err0 <- (t[1,2]+t[2,1])/sum(t)
err0

filter1 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter1,te[,-58])
t <- table(mailtype,te[,58])
err1 <- (t[1,2]+t[2,1])/sum(t)
err1

filter2 <- ksvm(type~.,data=trva,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter2,te[,-58])
t <- table(mailtype,te[,58])
err2 <- (t[1,2]+t[2,1])/sum(t)
err2

filter3 <- ksvm(type~.,data=spam,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter3,te[,-58])
t <- table(mailtype,te[,58])
err3 <- (t[1,2]+t[2,1])/sum(t)
err3

#Implementation of SVM predictions:

#Get the RBF kernel function from the fitted model
rbf_kernel <- rbfdot(sigma = 0.05)

sv<-alphaindex(filter3)[[1]]
co<-coef(filter3)[[1]]
inte<- - b(filter3)
k<-NULL
for(i in 1:10){ # We produce predictions for just the first 10 points in the dataset.
  k2<-NULL
  for(j in 1:length(sv)){
    x_sv <- as.numeric(spam[sv[j], -58]) #get the j:th SV features as a numeric vector
    x_new <- as.numeric(spam[i, -58])    #get the i:th new point features as a numeric vector
    k2<- c(k2, co[j] * rbf_kernel(x_sv, x_new)) #My code here, this calculates alpha_i*y_i * K(x_i x))
  }
  k<-c(k, sum(k2) + inte) #My code here, which adds the intercept to the weighted sum
}
k
predict(filter3,spam[1:10,-58], type = "decision")

#Correct output:
#-1.0702965  1.0003450  0.9995908 -0.9999648 -0.9995379
# 1.0000612 -0.8585873 -0.9997047  0.9998209 -1.0000973

#My output:
#-1.0702965  1.0003450  0.9995908 -0.9999648 -0.9995379
# 1.0000612 -0.8585873 -0.9997047  0.9998209 -1.0000973

#Conclusion:
#They are now identical after the code modifications were made.¨
