# 
# Digit Recognizer: Kaggle Competition
# 
# The data directory contains 3 files:
#   - test.csv
#   - train.csv
#   - sample_submission.csv

##### Clean Up Workspace #####
rm(list = ls()) # Remove Previous Workspace
gc(reset = TRUE) # Garbage Collection

##### Install and/or Load Packages #####
packages <- function(x, repos = "http://cran.r-project.org", ...) {
  x <- deparse(substitute(x))
  if (!require(x, character.only = TRUE)) {
    install.packages(pkgs = x, dependencies = TRUE, repos = repos, ...)
    library(x, character.only = TRUE)
  }
}

# Load libraries
packages(doParallel) # Parallel Computing
packages(foreach) # Parallel Computing
packages(reshape2) # Manipulate Datasets
packages(dplyr) # Splitting, applying, and combining data
packages(boot) # Contains cv.glm
packages(leaps) # For regsubsets
packages(ggplot2)
packages(glmnet)
packages(forcats)
packages(caret)
packages(randomForest)
packages(e1071)
packages(h2o)


##### DEFINE FUNCTIONS #####

plotDigit <- function(x, zlim = c(-1, 1)) {
  cols <- gray.colors(100)[100:1]
  image(matrix(x, nrow = 28)[, 28:1], col = cols,
        zlim = zlim, axes = FALSE)  
}

miss.class <- function (pred.class, true.class, produceOutput=F) {
  confusion.mat <- table(pred.class, true.class)
  if (produceOutput) {
    return(1 - sum(diag(confusion.mat)) / sum(confusion.mat))
  } else {
    print('miss-class')
    print(1 - sum(diag(confusion.mat)) / sum(confusion.mat))
    print('confusion mat')
    print(confusion.mat)
  }
}


##### SET UP DATA #####

# Import train and test data
train <- read.csv("train.csv")
test  <- read.csv("test.csv")

# Convert label to factor for classification
train[, 1] <- as.factor(train[, 1])

# Create matrices
X <- as.matrix(train[, -1])
Y <- train[, 1]
X_0 <- as.matrix(test)

# Start a local h2o cluster
localH2O <- h2o.init(max_mem_size = '6g', nthreads = -1)

# Create H2O df
train.h2o <- as.h2o(train)
test.h2o  <- as.h2o(test)

##### DATA EXPLORATION #####

# Plot an image
plotDigit(X[1, ])    # from training set
plotDigit(X_0[1, ])  # from test set

for (i in 1:nrow())
  
  # Missing values, duplicate data, etc.
  str(train) # Structure of the df: # of obs, # of variables, types of variables
any(is.na(train)) # TRUE if missing values exist, FALSE otherwise
colSums(sapply(train, is.na)) # Number of missing values per column
sum(is.na(train)) / (nrow(train) * ncol(train)) # Percentage of values that are missing
nrow(train) - nrow(unique(train)) # Number of duplicate rows


##### MODELS #####

# Set timer
s <- proc.time()

# random forest (h2o)
start <- proc.time()[3]
rf.h2o <- h2o.randomForest(x = names(train.h2o[, -1]), y = "label", train.h2o, ntree = 50)
stop <- proc.time()[3]
print(stop-start)
yhat.rf <- as.numeric(h2o.predict(rf.h2o, newdata=test.h2o, type='class'))

# deep learning (h2o)
start <- proc.time()[3]
nn.h2o <- h2o.deeplearning(x = 2:785, y = 1, training_frame = train.h2o,
                           activation = "RectifierWithDropout", input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5, 0.5), balance_classes = T,
                           hidden = c(100, 100), momentum_stable = 0.99,
                           nesterov_accelerated_gradient = T, epochs = 15)
stop <- proc.time()[3]
print(stop-start)
h2o.confusionMatrix(nn.h2o)
yhat.nn <- h2o.predict(nn.h2o, test.h2o)
nn.pred <- as.data.frame(yhat.nn)
nn.pred <- data.frame(ImageId = seq(1,length(nn.pred$predict)),
                      Label = nn.pred$predict)
write.csv(nn.pred, "nn_predictions.csv", row.names = F, quote = F)


### Random Forest

require(randomForest)
start  <- proc.time()[3]
rf.out <- randomForest(x = X, y = as.factor(Y), ntree = 50)
stop   <- proc.time()[3]
print(stop-start)
y.hat  <- as.numeric(predict(rf.out, newdata=X_0, type='class')) - 1

# Create submission files
rf.pred <- data.frame(seq(1:nrow(test)), y.hat)
names(rf.pred) <- c("ImageId", "Label")
write.csv(rf.pred, "rf_predictions.csv", row.names = F, quote = F)

### Support Vector Machines

# Fit SVMs with linear, radial, and polynomial kernels
svm.lin <- tune(svm, as.factor(label) ~ ., data=train, kernel="linear", type='C',
                ranges=list(0.01, 0.1, 1))
yhat.svm <- predict(svm.lin$best.model, test)


### Linear Discriminant Analysis
packages(MASS)

# Remove constant variables
train.nzv <- train[, -nearZeroVar(train, freqCut = 99999, uniqueCut = 0.000001)]

# Scale data by max value
train.s <- train.nzv[, -1] / max(train.nzv[, -1])
train.s <- cbind(Y = train$label, train.s)

# Fit an LDA model
lda.out <- lda(Y ~ ., data=train.s)
yhat.lda <- predict(lda.out, test)

# Create submission files
lda.pred <- data.frame(ImageId = seq(1:nrow(test)), Label = yhat.lda$class)
write.csv(lda.pred, "lda_predictions.csv", row.names = F, quote = F)

### Neural Net

install.packages("drat", repos = "https://cran.rstudio.com")
drat::addRepo("dmlc")
install.packages("mxnet")
library(mxnet)


### Boosting trees
require(gbm)

lambdas      <- c(0.01, 0.1)
depths       <- c(1, 2)
iterations   <- 50
distribution <- "multinomial"

for (lambda in lambdas) {
  for (depth in depths) {
    for (iteration in iterations) {
      gbm.fit <- gbm(Y ~ ., data = train.s, distribution = distribution,
                     n.trees = iterations, shrinkage = lambda,
                     interaction.depth = depth, verbose = T)
      
      yhat.gbm <- predict.gbm(gbm.fit, test, n.trees = iteration, type="response")
      yhat.gbm[yhat.gbm >= 0.5]
    }
  }
}

yhat.gbm <- predict.gbm(gbm.fit, test, n.trees = 50, type="response")



