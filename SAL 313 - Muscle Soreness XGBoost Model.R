library(xgboost)
library(tidymodels)
library(MASS)
library(tidyverse)
library(readxl)
library(lubridate)
library(lmtest)
library(car)
library(plm)
library(sandwich)
library(cluster)
library(factoextra)
set.seed(101)
### all of data 2 without filtering by session type
xg_model_data <- data2 %>% dplyr::select(`Energy z-score`, `Leg Heaviness z-score`, 
                                  `Mental State z-score`, `Sleep Quality z-score`, `Distance (Last 28 Days)`, 
                                  `Distance (Last 28 Days)`, `Acute Player Load (Rolling Ave)`, `Chronic Player Load (Rolling Ave)`,
                                  Phase, `Session Player Load`, `Session Duration`, session_type, `Calf z-score`, `Foot z-score`,
                                  `General Soreness z-score`, `Glute z-score`, `Groin z-score`, `Hamstring z-score`, `Lower Back z-score`, 
                                  `Quad z-score`, `Total Muscle Soreness z-score`, `Session ID.y`, `Entity Id`)
# Changes foot z-score to a numeric value
xg_model_data$`Foot z-score` <- as.numeric(xg_model_data$`Foot z-score`)
# Creates Dummy Variables for Phase Variable
xg_model_data <- xg_model_data %>% mutate(Phase = ifelse(Phase == "In-Season", 0,
                                                         ifelse(Phase == "JLT Series", 1, 2)))

split <- initial_split(xg_model_data, strata = `Session Player Load`)
train <- training(split)
test <- testing(split)

dtrain <- xgb.DMatrix(as.matrix(dplyr::select(train,-`Session Player Load`, -`Session ID.y`, -`Entity Id`)),
                      label = train$`Session Player Load`)
dtest <- xgb.DMatrix(as.matrix(dplyr::select(test,-`Session Player Load`, -`Session ID.y`, -`Entity Id`)),
                     label = test$`Session Player Load`)

watchlist <- list(train = dtrain, eval = dtest)
# testing
param <- list(max_depth = 4, eta = 0.3, nthread = 2,
              objective = "reg:squarederror", eval_metric = "rmse")

xgb.train(param, dtrain, nrounds = 50, watchlist)

# finding appropriate hyperparameters

find_rounds <- function(rounds){
  rounds <- floor(rounds)
  xgb.train(
    params = list(
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = dtrain,
    nrounds = rounds,
    watchlist = list(train = dtrain, test = dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(find_rounds, c(2,500), tol = 1) #minimizes as 192

find_optimal_depth <- function(max_depth){
  max_depth <- floor(max_depth)
  xgb.train(
    params = list(
      max_depth = max_depth,
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = dtrain,
    nrounds = 192,
    watchlist = list(train = dtrain, test = dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(find_optimal_depth, c(2,100), tol = 1) # minimizes at 63

find_optimal_child_weight <- function(min_child_weight){
  weight <- floor(min_child_weight)
  xgb.train(
    params = list(
      max_depth = 63,
      min_child_weight = weight,
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = dtrain,
    nrounds = 192,
    watchlist = list(train = dtrain, test = dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(find_optimal_child_weight, c(2,1000), tol = 1) #minimizes at 77

learning_vals <- tibble(
  eta = 1 / (100 *(1:10)),
  nrounds = 129 * 10 * (1:10)
)



expand_mod <- function(eta, nrounds) {
  xgb.train(params = list(
    max_depth = 63,
    min_child_weight = 77,
    eta = eta,
    objective = "reg:squarederror",
    eval_metric = "rmse"
  ),
  data = dtrain, 
  nrounds = nrounds,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 1
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}


(new_performance <- pmap_dbl(learning_vals, expand_mod))
#Optimises on 9th iteration: rmse = 84.14
#eta = 1/900, rounds = 9040


##finding optimal colsample to try and reduce rmse further
find_optimal_colsample_level <- function(colsample){
  colsample <- colsample
  xgb.train(
    params = list(
      max_depth = 63,
      min_child_weight = 77,
      colsample_bylevel = colsample,
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = dtrain,
    nrounds = 192,
    watchlist = list(train = dtrain, test = dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}
optimize(find_optimal_colsample_level, c(0,1)) # minimized at 0.7770876
#tried for all colsamples, level yielded best results

learning_vals <- tibble(
  eta = 1 / (100 *(1:10)),
  nrounds = 129 * 10 * (1:10)
)



expand_mod <- function(eta, nrounds) {
  xgb.train(params = list(
    max_depth = 63,
    min_child_weight = 77,
    eta = eta,
    colsample_bylevel = 0.7770876,
    objective = "reg:squarederror",
    eval_metric = "rmse"
  ),
  data = dtrain, 
  nrounds = nrounds,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 1
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}


(new_performance <- pmap_dbl(learning_vals, expand_mod))
# 7th iteration was best: 83.55068
#eta = 1/700, nrounds = 9030

#Final Model
(bst <- xgb.train(params = list(
  max_depth = 63,
  min_child_weight = 77,
  eta = 1/700,
  colsample_bylevel = 0.7770876,
  objective = "reg:squarederror",
  eval_metric = "rmse"
),
data = dtrain, 
nrounds = 9030,
watchlist = list(train = dtrain, test = dtest),
verbose = 0
))

# Getting Summary Statistics for Model Predicted Session Player Load
test %>% dplyr::select(-`Session Player Load`) %>% 
  as.matrix() %>% 
  predict(object = bst) %>% 
  summary()

# Feature Importance from the Model
importance <- xgb.importance(model = bst)

# Releveling the Features for the Importance Plot
importance$Feature <-  factor(importance$Feature, levels = c('Calf z-score', 'Glute z-score', 'Total Muscle Soreness z-score', 'Quad z-score',
                                                              'Hamstring z-score', 'Groin z-score', 'Lower Back z-score', 'Leg Heaviness z-score',
                                                              'Energy z-score', 'Mental State z-score', 'General Soreness z-score', 'Sleep Quality z-score',
                                                              'Foot z-score', 'Phase', 'Distance (Last 28 Days)', 'Chronic Player Load (Rolling Ave)','Acute Player Load (Rolling Ave)',
                                                              'session_type', 'Session Duration'))
# Importance Plot w/ Session Duration and Session Type
importance %>% ggplot(aes(Gain, Feature)) +
  geom_col(fill = "#0E4D92")+
  theme_classic() +
  xlab("Importance") +
  ylab("Predictor") +
  ggtitle("Model 1 Variable Importance")

# Removing Features Session Duration and Session Type from the Importance Dataframe
# Gonna create another Importance Plot without Session Importance to encapsulate the other variables in the model
importance_wo_session_duration <- importance %>% filter(!Feature %in% c("Session Duration","session_type"))

# Creating new importance metric without session duration and session type
importance_wo_session_duration <- importance_wo_session_duration %>% mutate(new_importance = Gain / sum(importance_wo_session_duration$Gain))

#Releveling feature for the Importance Chart
importance_wo_session_duration$Feature <-  factor(importance_wo_session_duration$Feature, levels = c('Calf z-score', 'Glute z-score', 'Total Muscle Soreness z-score', 'Quad z-score',
                                                                                                     'Hamstring z-score', 'Groin z-score', 'Lower Back z-score', 'Leg Heaviness z-score',
                                                                                                     'Energy z-score', 'Mental State z-score', 'General Soreness z-score', 'Sleep Quality z-score',
                                                                                                     'Foot z-score', 'Phase', 'Distance (Last 28 Days)', 'Chronic Player Load (Rolling Ave)','Acute Player Load (Rolling Ave)'))

# Importance Plot w/o Session Duration and Sesion Type
importance_wo_session_duration %>% ggplot(aes(new_importance, Feature)) +
  geom_col(fill = "#0E4D92")+
  theme_classic() +
  xlab("Importance") +
  ylab("Predictor") +
  ggtitle("Model 1 Variable Importance", subtitle = "Without Session Type & Session Duration") +
  theme(plot.subtitle = element_text(size = 9))

# Adding predicted values to the dataframe
xg_model_data <- xg_model_data %>% mutate(pred_session_player_load = predict(bst,xgb.DMatrix(as.matrix(dplyr::select(xg_model_data,-`Session Player Load`, -`Session ID.y`, -`Entity Id`)),
                                                                             label = xg_model_data$`Session Player Load`)))
