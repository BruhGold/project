# install.packages("tidyverse")
# install.packages("glmnet")

library(tidyverse)
library(glmnet)

forest <- read.csv("data/ForestFire.csv")

# Relevant data visualizations of features of this dataset
str(forest)
summary(forest)

# Check distribution of burned area
hist(forest$area, breaks = 50, main = "Distribution of Burned Area", xlab = "Area")

# We can see that most are close to zero but some are huge surpassing 1000 area
# use log_area instead cuz most forest fire are small but some are huge which make it skew
# toward the right
forest$log_area <- log1p(forest$area)
model <- lm(log_area ~ ., data = forest %>% select(-area))

hist(forest$log_area, 
     breaks = 50, 
     main = "Distribution of Burned Area (Log Transformed)", 
     xlab = "log(1 + Area)")

# Cleanup dataset. Are there any outliers ? Are there any missing values in any of the features ?
# Explain how you handle categorial features in the dataset.
# Check for missing values
colSums(is.na(forest))

# remove outliers with cook distance
# Calculate cook distance
cooksd <- cooks.distance(model)

# Plot cook distance
plot(cooksd, type = "h", main = "Cook's Distance", ylab = "Distance")
abline(h = 4/length(cooksd), col = "red", lty = 2)

# Identify influential points
threshold <- 4 / length(cooksd)
influential <- as.numeric(names(cooksd)[(cooksd > threshold)])
influential <- influential[!is.na(influential)]
influential
# Show how many are removed
cat("Number of influential outliers removed:", length(influential), "\n")

# Remove influential observations
forest_clean <- forest[-influential, ]

# Handle categorial features
# https://www.r-bloggers.com/2022/01/handling-categorical-data-in-r-part-1/
# We convert it to factor so when needed, R can create dummy tables
forest_clean$month <- factor(forest_clean$month, 
                       levels = c("jan", "feb", "mar", "apr", "may", "jun", 
                                  "jul", "aug", "sep", "oct", "nov", "dec"))

forest_clean$day <- factor(forest_clean$day, 
                     levels = c("mon", "tue", "wed", "thu", "fri", "sat", "sun"))

str(forest_clean)
# Apply appropriate transformations of the features and/or output.
trainTestSplit = function(data, seed, trainRatio = 0.8){
  set.seed(seed)
  dataSize = nrow(data)
  trainSize = round(trainRatio * dataSize)
  trainIndex = sample(1:dataSize, replace = FALSE, size = trainSize)
  
  trainData = data[trainIndex,]
  testData = data[-trainIndex,]
  return(list(train = trainData, test = testData))
}
fire.split = trainTestSplit(forest_clean, 0, trainRatio = 0.8)
fire.train = fire.split$train
fire.test = fire.split$test

# Use appropriate feature selection methods. Explain how you setup Ridge/Lasso/Elastic net 
# regularization method and interpret the result.
x_train <- model.matrix(log_area ~ . - area, data = fire.train)[, -1]
y_train <- fire.train$log_area

set.seed(123)

# Ridge Regression=
cv.ridge <- cv.glmnet(x_train, y_train, alpha = 0)
ridge.model <- glmnet(x_train, y_train, lambda = cv.ridge$lambda.min, alpha = 0)

# Lasso Regression
cv.lasso <- cv.glmnet(x_train, y_train, alpha = 1)
lasso.model <- glmnet(x_train, y_train, lambda = cv.lasso$lambda.min, alpha = 1)

# Elastic Net
cv.elastic <- cv.glmnet(x_train, y_train, alpha = 0.5)
elastic.model <- glmnet(x_train, y_train, lambda = cv.elastic$lambda.min, alpha = 0.5)

# Coefficients from best models
coef(ridge.model)
coef(lasso.model)
# Certain months like September, december have high influence on the burned area
# The DMC value and temp are also picked through the lasso model
coef(elastic.model)
# Elastic model select the same predictor as lasso meaning that they are important across models
# These results suggest that fire seasonality (December and September, etc) 
# and dryness indicators (DMC) 
# play key roles in explaining fire area variation

# Cross validation
# Prepare test data
x_test <- model.matrix(log_area ~ . - area, data = fire.test)[, -1]
y_test <- fire.test$log_area

# Predictions
y_pred <- predict(lasso.model, s = cv.lasso$lambda.min, newx = x_test)

# Evaluation
rmse <- sqrt(mean((y_test - y_pred)^2))
mae <- mean(abs(y_test - y_pred))
r2 <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)

cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("R²:", r2, "\n")

# Classification task
# We should use Recall score as classification metric because:
# we want to focus on avoiding false negative cases when we predict that 
# there is no fire but a fire occur
forest_clean$target <- ifelse(forest_clean$area > 0, 1, 0)

# Split train 80%, test 20%
fire.split <- trainTestSplit(forest_clean, seed = 0, trainRatio = 0.8)
fire.train <- fire.split$train
fire.test <- fire.split$test

set.seed(123)
fire.logit <- glm(target ~ ., data = fire.train, family = "binomial")
summary(fire.logit)

# i remove "nov" from test because it doesn't exist in train
# that cause this error "factor month has new levels nov"
fire.test <- fire.test[fire.test$month != "nov", ]
test.prod <- predict(fire.logit, newdata = fire.test, type = "response")
test.label <- as.vector(ifelse(test.prod > 0.5, 1, 0))

target = fire.test$target
TP = sum((test.label == target) & (test.label == 1))
FN = sum((test.label != target) & (test.label == 0))
recall_score = TP / (TP + FN)
recall_score
# the recall score is 1
# This means this model can correctly detect burned areas while avoiding cases where it burn
# and we say it is not
# Of course there might be a lot of true negative cases meaning that we deploy the troops for nothing
# We might need precision score as well

# Final models
# Regression model we use the lasso one since we compare it with elastic and the features
# that they selected are familiar
# A lot of predictors are also removed in the process which help us narrow down to the important ones
# This will help us predict the burned area
lasso.model

# Classification model with a perfect recall score of "1" 
# can help us predict if a fire happen or not while avoiding false negative cases well
fire.logit