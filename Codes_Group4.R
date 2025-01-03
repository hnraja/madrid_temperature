

library('tidyr')
library("readxl")
library('dplyr')
library('xgboost')
library('MASS')
library('car')
library('glmnet')
library('lars')

set.seed(1234)

######################################################################################################
                          ##############Load Data From Excel#################
######################################################################################################


excel <- read_excel("Data_Group4.xlsx", sheet = "weather_features")


######################################################################################################
                               ##############Regression#################
######################################################################################################

# Load Madrid data
data <- excel[35147:71413,]

# Reduce data to only necessary/relevant ones
# Note: (1) Picking 12AM data only to see some graphs/trends more clear
#       (2) Dropped categorical variables as they lead to
#           enormous number of indicators for linear model
indices <- seq(1, nrow(data), by = 24)
data <- data[indices,c("pressure", "humidity", "wind_speed", "wind_deg", "rain_1h",
                       "rain_3h", "clouds_all", "weather_id", "temp")]


# Linear model
linear_model <- lm(temp ~ ., data = data)
summary(linear_model)


# Linear model residual diagnostics
par(mfrow = c(2, 2))
plot(linear_model$fitted, linear_model$residuals, xlab = "Fitted", ylab = "Residuals",
     main = "Fitted v.s. Residuals", pch = 16, col = adjustcolor("black",0.7))
abline(h = 0, lty = 2, lwd = 2, col = "red")
car::qqPlot(linear_model$residuals, xlab = "Quantiles", ylab = "Residuals",
            main = "QQ-Plot", pch = 16)
residualsTS <- ts(linear_model$residuals, start = c(2015, 1), end = c(2018, 12), frequency = 24)
plot(residualsTS, xlab = "Year", ylab = "Residuals", main = "Time v.s. Residuals",
     pch = 16, col = adjustcolor("black",0.7))
abline(h = 0, lty = 2, lwd = 2, col = "red")
acf(linear_model$residuals, main = "ACF")


# AIC and BIC
stepAIC(linear_model)        # AIC
stepAIC(linear_model, direction = "both", k = log(nrow(data)))        # BIC


# Updated linear model from the best BIC
linear_model_update <- lm(temp ~ pressure + humidity + clouds_all + weather_id, data = data)
summary(linear_model_update)


# Updated linear model residual diagnostics
par(mfrow = c(2, 2))
plot(linear_model_update$fitted, linear_model_update$residuals, xlab = "Fitted", ylab = "Residuals",
     main = "Fitted v.s. Residuals", pch = 16, col = adjustcolor("black",0.7))
abline(h = 0, lty = 2, lwd = 2, col = "red")
car::qqPlot(linear_model_update$residuals, xlab = "Quantiles", ylab = "Residuals",
            main = "QQ-Plot", pch = 16)
residualsTS <- ts(linear_model_update$residuals, start = c(2015, 1), end = c(2018, 12), frequency = 24)
plot(residualsTS, xlab = "Year", ylab = "Residuals", main = "Time v.s. Residuals",
     pch = 16, col = adjustcolor("black",0.7))
abline(h = 0, lty = 2, lwd = 2, col = "red")
acf(linear_model_update$residuals, main = "ACF")


# VIF comparison
par(mfrow = c(1, 2))
vif(linear_model)
vif(linear_model_update)


# LASSO
X <- as.matrix(data[,1:8])
Y <- as.numeric(data[[9]])
fit.LASSO.range <- glmnet(X, Y, alpha = 1, lambda = seq(0, 10, length = 200),
                          standardize = TRUE, family = "gaussian")
coefs <- predict(fit.LASSO.range, type = "coef", s = fit.LASSO.range$lambda)
coefs <- as.matrix(coefs)
Y.lim <- range(coefs, na.rm = TRUE)
Y.lim <- extendrange(Y.lim, f = 0.1)
par(mfrow = c(1, 1))
plot(fit.LASSO.range$lambda, coefs[2, ], main = "Shrinkage of Parameters (LASSO)",
     xlab = "lambda", ylab = "beta", xlim = c(0, 0.06), ylim = Y.lim, type = "n")
for(i in 1:nrow(coefs)) {
  lines(fit.LASSO.range$lambda, coefs[i,], lty = i, col = i)
}
legend("bottomright", legend = paste0("beta", 1:(nrow(coefs)-1)), 
       lty = 1:(nrow(coefs)-1), col = 1:(nrow(coefs)-1), bty = "n")
data.lasso <- lars(X, Y, type = "lasso")
data.lasso
par(mfrow = c(1, 1))
plot(data.lasso)


# APSE of LASSO
set.seed(123)
training_i <- sample(1:nrow(data), size = 0.9 * nrow(data))
training <- data[training_i,]
X <- as.matrix(training[,1:8])
Y <- as.numeric(training[[9]])
fit <- glmnet(X, Y, alpha = 1, lambda = seq(0, 10, length = 200),
              standardize = TRUE, family = "gaussian")
optimal_lambda <- cv.glmnet(X, Y, alpha = 1)$lambda.min
test <- data[-training_i,]
X <- as.matrix(test[,1:8])
Y <- as.numeric(test[[9]])
APSE <- mean((Y - predict(fit, s = optimal_lambda, newx = X))^2)
APSE


######################################################################################################
                           ##############Regression END#################
######################################################################################################



######################################################################################################
                            ##############Holt-Winters#################
######################################################################################################

#Defining the training and test set
excel <- as.data.frame(excel, header=TRUE)
city <- excel[excel$city_name == "Madrid",]
temp <- ts(city$temp, start=c(2015,1), frequency=24)
train.length <- ceiling(0.9*length(temp))
train <- window(temp, end=c(2015+floor(train.length/24), train.length%%24))
test <- window(temp, start=c(2015+floor(train.length/24), train.length%%24+1))

# Simple exponential smoothing
es <- HoltWinters(train, gamma=FALSE, beta=FALSE)
es.pred = predict(es, n.ahead=length(test))
es.apse <- mean((test-es.pred[,"fit"])^2)
es.apse

# Double exponential smoothing
des <- HoltWinters(train, gamma=FALSE)
des.pred = predict(des, n.ahead=length(test))
des.apse <- mean((test-des.pred[,"fit"])^2)
des.apse

# Additive Holt Winters
hw.ad <- HoltWinters(train, seasonal = c("additive"))
pred.ad = predict(hw.ad, n.ahead=length(test), prediction.interval = TRUE , level=0.95)
hw.apse <- mean((test-pred.ad[,"fit"])^2)
hw.apse

# Plot Additive Holt-Winters
plot(hw.ad, pred.ad, main="Additive Holt-Winters Smoothing", xlab = "Year",xaxt="n")
lines(test, col = rgb(0.3, 0.3, 0.3, alpha = 0.5))
legend("bottomleft", legend = c("Prediction","Prediction interval", "Actual Values"), col = c("red","blue", "darkgrey"), lty=c(1,1), bty = "n")
axis(1, at=c(2000, 2500, 3000, 3500), labels = c("2015", "2016", "2017", "2018"))

# Forecasting with the best model (Additive HW)
hw.ad <- HoltWinters(temp, seasonal = c("additive"))
pred.ad = predict(hw.ad, n.ahead=24*7, prediction.interval = TRUE , level=0.95)
plot(hw.ad, pred.ad, main="Additive Holt-Winters Smoothing", xlim=c(2000,3500))
plot(hw.ad, pred.ad, main="Additive Holt-Winters Smoothing", xlim=c(3400,3530))

# Residuals
hw.resids = temp - fitted(hw.ad)[,1]
par(mfrow = c(2, 2)) 
plot(fitted(hw.ad)[,1], hw.resids, xlab = "Fitted", ylab = "Residuals", main = "Fitted vs. Residuals", type="p", pch=19)
abline(h = 0, lty = 2, col = "red")
car::qqPlot(hw.resids, col = adjustcolor("black", 0.7), xlab = "Theoretical Quantiles (Normal)", ylab = "Residual (Sample) Quantiles (r.hat)", main = "Normal Q-Q Plot", id = FALSE)
plot(hw.resids, xlab = "Time", ylab = "Residuals", main = "Time vs. Residuals")
abline(h = 0, lty = 2, col = "red") 
acf(hw.resids, main = "Sample ACF of Residuals")

######################################################################################################
                            ##############Holt-Winters END#################
######################################################################################################


######################################################################################################
                            ##############SARIMA#################
######################################################################################################


# Exploratory analysis
data <- read_excel("Data_Group4.xlsx")
data <- as.data.frame(data, header=TRUE)
city <- data[data$city_name == "Madrid",]
city$dt_iso <- as.POSIXct(city$dt_iso)
n <- ceiling(0.9*nrow(city))
trainDF <- city[c(1:n),]
testDF <- city[c((n+1):(nrow(city))), ]
plot(trainDF$dt_iso, trainDF$temp, ylab="Temperature (K)", xlab="Year", type="l",
     main="Hourly Temperature in Madrid For 2015 to 2018",  
     xlim=c(city[1,]$dt_iso,city[nrow(city),]$dt_iso))
points(testDF$dt_iso, testDF$temp, col="blue", type="l")
legend("bottomright", col=c("black","blue"), lty=c(1,1), 
       legend=c("Training set", "Test set"))
library(forecast)
mstl(city$temp, lambda="auto", iterate=5)%>% autoplot()

# Load data as time series
temp <- ts(city$temp, start=c(2015,1), frequency=24)
n <- ceiling(0.9*length(temp))
train <- window(temp, end=c(2015+floor(train.length/24), train.length%%24))
test <- window(temp, start=c(2015+floor(train.length/24), train.length%%24+1))

# Plot training data full ACF/PACF
acf(train, lag.max=n, main="ACF of Madrid Temperature Data")
pacf(train, lag.max=n, main="PACF of Madrid Temperature Data")

# Plot training data zoomed in ACF/PACF
acf(train, lag.max=3 * 24, main="ACF of Madrid Temperature Data")
pacf(train, lag.max=3 *  24, main="PACF of Madrid Temperature Data")

# Plot APSE for AR(p) models
p <- c()
APSE <- c()
for (i in c(1:100)) {
  fit <- ar(train, order.max = i)
  p[i] <- fit$order
  predicted <- as.numeric(predict(fit, n.ahead=length(test))$pred)
  APSE[i] <- mean((test - predicted)^2)
}
plot(p, APSE,main="APSE for AR(p)") # Best model is AR(19)

# Find best differencing
for (D in c(1:3)) {
  diff1 <- diff(train, differences = D, lag=24)
  acf(diff1, lag.max=24*15, main=paste("ACF of d=0 D=",D))
  diff1 <- diff(train, differences = D)
  acf(diff1, lag.max=24*15, main=paste("ACF of d=",D, "D=0"))
  for (d in c(1:3)) {
    diff1 <- diff(diff(train, differences = d), differences = D, lag=24)
    acf(diff1, lag.max=24*15, main=paste("ACF of d=",d, "D=",D))
  }
}

# d=1, D=1
diff1 <- diff(diff(train), lag=24)

# Plot final differenced data full ACF/PACF
acf(diff1, lag.max = n)
pacf(diff1, lag.max = n)

# Plot final differenced data zoomed in ACF/PACF
acf(diff1, lag.max = 3*24)
pacf(diff1, lag.max = 3*24)

# Residual diagnostics
# Commented out since some combinations are invalid can cause run time errors
# library(astsa)
# for (p in c(0:3)) {
#   for (P in c(0:3)) {
#     for (Q in c(0,1)) {
#       sarima(train, p=p,d=1,q=0,P=P,D=1,Q=Q,S=24)
#       sarima(exp(train), p=p,d=1,q=0,P=P,D=1,Q=Q,S=24)
#       sarima(log(train), p=p,d=1,q=0,P=P,D=1,Q=Q,S=24)
#       for (i in seq(-2,2,length.out=50)) {
#         sarima(train^i, p=p,d=1,q=0,P=P,D=1,Q=Q,S=24)
#       }
#     }
#   }
# }

# APSE
# Commented out since some combinations are invalid can cause run time errors
# for (p in c(0:3)) {
#   for (P in c(0:3)) {
#     for (Q in c(0,1)) {
#       predicted <- as.numeric(sarima.for(train,p=p,d=d,q=q,P=P,D=D,Q=Q,S=S, 
#                                          n.ahead=length(test))$pred)
#       APSE <- mean((test - predicted)^2)
#       paste("SARIMA(",p,",",d,",",q,")(",P,",",D,",",Q,")[24] - APSE=",APSE)
#     }
#   }
# }

# Final SARIMA 
y_hat <- as.numeric(sarima.for(train, p=0, d=1, q=0,P=3,D=1,Q=1,S=24, n.ahead=length(test))$pred)
mean((y_hat - test)^2) 
tim <- city[c((n+1):(nrow(city))), ]$dt_iso
plot(tim, as.numeric(test), main="SARIMA Test Set Performance", ylab="Temperature (K)", type="l", xlab="2019")
points(tim, y_hat, type="l", col="red")
legend("bottomleft", legend=c("Test data", "SARIMA(0,1,0)(3,1,1)[24]"), col=c("black","red"), lty=c(1,1))


######################################################################################################
                                ##############SARIMA END#################
######################################################################################################


tg_cols <- c("pressure", "humidity", "wind_speed", "wind_deg", "rain_1h", "rain_3h", "snow_3h", "clouds_all", "weather_id")


X <- excel[excel['city_name'] == 'Madrid', tg_cols]
Y <- excel[excel['city_name'] == 'Madrid', 'temp']


######################################################################################################
                             ##############XGBoost#################
######################################################################################################


# Adding lagged temp columns to X
lag_num <- 7*24

X_lag <- X %>%
  mutate(
    temp_lag1 = lag(Y$temp, lag_num), #lagged by 1 week
    temp_lag2 = lag(Y$temp, 2*lag_num), #lagged by 2 weeks
    humidity_lag1 = lag(X$humidity, lag_num),
    pressure_lag1 = lag(X$pressure, lag_num),
    snow_1h_lag1 = lag(X$snow_3h, lag_num),
    clouds_lag1 = lag(X$clouds_all, lag_num),
    wind_speed_lag1 = lag(X$wind_speed, lag_num),
    wind_deg_lag1 = lag(X$wind_deg, lag_num),
    rain_1h_lag1 = lag(X$rain_1h, lag_num),
    weather_id_lag1 = lag(X$weather_id, lag_num)
  )

X_lag <- X_lag[ , !(names(X_lag) %in% tg_cols)]

head(Y)
head(X)

data <- X_lag %>%
  mutate(temp = Y$temp)

test_num <- floor(0.1*nrow(data)) #7*24
train_data <- data[1:(nrow(data) - test_num), ]
test_data <- data[(nrow(data) - (test_num -1)):nrow(data), ]

train_X <- as.matrix(train_data %>% select(-temp))
train_Y <- train_data$temp
test_X <- as.matrix(test_data %>% select(-temp))
test_Y <- test_data$temp

grid <- expand.grid(
  max_depth = c(3, 5, 7, 9),
  learning_rate = c(0.01,0.05, 0.1, 0.2),
  n_estimators = c(25, 50, 100, 200,250)
)

results <- list()


for (i in 1:nrow(grid)) {
  params <- grid[i, ]
  
  # Fit the model
  model <- xgboost(
    data = train_X,
    label = train_Y,
    max_depth = params$max_depth,
    eta = params$learning_rate,
    nrounds = params$n_estimators,
    objective = "reg:squarederror",
    verbose = 0
  )
  
  # Predict on the test set
  predictions <- predict(model, test_X)
  
  # Calculate & save MSE
  mse <- mean((predictions - test_Y)^2)
  
  results[[i]] <- list(
    max_depth = params$max_depth,
    learning_rate = params$learning_rate,
    n_estimators = params$n_estimators,
    mse = mse
  )
}

results_df <- do.call(rbind, lapply(results, as.data.frame))

best_result <- results_df[which.min(results_df$mse),]
print(best_result)

#set test_num_fin to 7*24 to get the forecast chart and to floor(0.1*length(data)) for the Fitted plot

test_num_fin <- floor(0.1*nrow(data)) #7*24 #
train_fin <- data[1:(nrow(data) - test_num_fin), ]
test_fin <- data[(nrow(data) - (test_num_fin -1)):nrow(data), ]

train_X_fin <- as.matrix(train_fin %>% select(-temp))
train_Y_fin <- train_fin$temp
test_X_fin <- as.matrix(test_fin %>% select(-temp))
test_Y_fin <- test_fin$temp

#Fitting the optimal model from grid search
final_model <- xgboost(
  data = train_X_fin,
  label = train_Y_fin,
  max_depth = 3,
  eta = 0.2,
  nrounds = 25,
  objective = "reg:squarederror",
  verbose = 0
)

# Predict on the test set
predictions_fin <- predict(final_model, test_X_fin)
fitted_values <- predict(final_model, train_X_fin)

mean((predictions_fin - test_Y_fin)^2)

#plot with last week as the Test set
test_num_fin <- floor(0.1*nrow(data))
fitted_ts_7 <- ts(fitted_values, start = c(2015,1), frequency = 365*24, end = c(2018,365*24-test_num_fin))

pred_ts_7 <-  ts(predictions_fin, start = c(2018, 365*24-test_num_fin+1), frequency = frequency(fitted_ts_7))

y_ts <- ts(Y, start = c(2015,1), frequency = 365*24, end = c(2018,365*24))

train_y_ts <- window(y_ts,end = end(fitted_ts_7))
test_y_ts <- window(y_ts, start = start(pred_ts_7))

plot(train_y_ts,
     xlab = "Time", ylab = "Temperature",
     col = "lightblue", main = "Data, Fitted and Predicted Values", lty= 1, xlim = c(2015,2019)
)

lines(test_y_ts, col = "darkblue", lty = 1)
lines(fitted_ts_7, col = "red", lty = 1)
lines(pred_ts_7, col = "green", lty = 1)

legend("topleft",
       legend = c("Train Set", "Fitted","Forecast","Test Set"),
       col = c("lightblue", "red","green","darkblue"),lty = c(1,1,1,1)
)


#plot with 10% data as the Test set
#For test_num_fin <- 7*24
fitted_ts <- ts(fitted_values, start = c(2015,1), frequency = 365*24, end = c(2018,358*24))

pred_ts <-  ts(predictions_fin, start = c(2018, 8593), frequency = frequency(fitted_ts))

y_ts_2018 <- window(y_ts, start =c(2018,1),  end = c(2018,358*24))
fitted_ts_2018 <- window(fitted_ts, start =c(2018,1))

# Plot the data
plot(y_ts_2018,
     xlab = "Time", ylab = "Temperature",
     col = "darkblue", main = "Data, Fitted and Predicted Values", lty= 1
)

lines(fitted_ts_2018, col = "red", lty = 1)
lines(pred_ts, col = "green", lty = 1)

legend("topleft",
       legend = c("Data", "Fitted","Forecast"),
       col = c("darkblue", "red","green"),lty = c(1,1,1)
)


#Feature Importance plot
feature_importance <- xgb.importance(
  feature_names = colnames(train_X),  
  model = final_model
)


feature_importance <- feature_importance[order(-feature_importance$Gain),]

features <- feature_importance$Feature
gain <- feature_importance$Gain

barplot(
  height = gain,
  names.arg = features,
  horiz = TRUE,  # Horizontal bars
  las = 1,       # Rotate axis labels
  col = "#1f78b4", # A professional color choice
  border = NA,    # No border for cleaner look
  main = "Feature Importance",
  xlab = "Importance (Gain)"
)


######################################################################################################
                        ##############XGBoost END#################
######################################################################################################




