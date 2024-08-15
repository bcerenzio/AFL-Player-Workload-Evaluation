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
library(ModelMetrics)
set.seed(101)
# collecting mean acute, chronic, and session workloads 
average_session_data <- data2 %>% group_by(`Session ID.y`, date) %>% 
  reframe(sess_id = `Session ID.y`,
          acwr_distance = mean(`ACWR Distance`, na.rm = TRUE),
          acwr_hsr_4 = mean(`ACWR HSR - Vel Zone 4 (18-24 km/h)`,na.rm = TRUE),
          acwr_hsr_5 = mean(`ACWR HSR - Vel Zone 5 (24-28 km/h)`, na.rm = TRUE),
          acwr_player_load = mean(`ACWR Player Load`, na.rm = TRUE),
          acute_dist_rolling = mean(`Acute Distance (Rolling Ave)`, na.rm = TRUE),
          acute_hsr_4_rolling = mean(`Acute HRS - Vel Zone 4 (Rolling Ave)`, na.rm = TRUE),
          acute_hsr_5_rolling = mean(`Acute HSR - Vel Zone 5 (Rolling Ave)`, na.rm = TRUE),
          acute_player_load = mean(`Acute Player Load (Rolling Ave)`, na.rm = TRUE),
          chronic_distance_rolling = mean(`Chronic Distance (Rolling Ave)`, na.rm = TRUE),
          chronic_hsr_4_rolling = mean(`Chronic HSR - Vel Zone 4 (Rolling Ave)`, na.rm = TRUE),
          chronic_hsr_5_rolling = mean(`Chronic HSR - Vel Zone 5 (Rolling Ave)`, na.rm = TRUE),
          chronic_player_load_rolling = mean(`Chronic Player Load (Rolling Ave)`, na.rm =TRUE),
          hsr_4_28days = mean(`HSR - Vel Zone 4 (Last 28 Days)`, na.rm = TRUE),
          hsr_4_7days = mean(`HSR - Vel Zone 4 (Last 7 Days)`, na.rm = TRUE),
          hsr_5_28days = mean(`HSR - Vel Zone 5 (Last 28 Days)`, na.rm = TRUE),
          hsr_5_7days = mean(`HSR - Vel Zone 5 (Last 7 Days)`, na.rm = TRUE),
          player_load_28days = mean(`Player Load (Last 28 Days)`, na.rm = TRUE),
          player_load_7days = mean(`Player Load (Last 7 Days)`, na.rm = TRUE),
          sess_duration = mean(`Session Duration`, na.rm = TRUE),
          session_hard_accels = mean(`Session Hard Accels`, na.rm = TRUE),
          session_hard_decels = mean(`Session Hard Decels`, na.rm = TRUE),
          session_mod_accels = mean(`Session Moderate Accels`, na.rm = TRUE),
          session_mod_decels = mean(`Session Moderate Decels`, na.rm = TRUE),
          session_odometer = mean(`Session Odometer`, na.rm = TRUE),
          sess_player_load = mean(`Session Player Load`, na.rm = TRUE),
          session_vel_4_dist = mean(`Session Vel Zone 4 Dist (18-24 km/h)`, na.rm = TRUE),
          session_vel_4_efforts = mean(`Session Vel Zone 4 Efforts`, na.rm = TRUE),
          session_vel_5_dist = mean(`Session Vel Zone 5 Dist (24-28 km/h)`, na.rm = TRUE),
          session_vel_5_efforts = mean(`Session Vel Zone 5 Efforts`, na.rm = TRUE),
          session_vel_6_dist = mean(`Session Vel Zone 6 Dist (28-36 km/h)`, na.rm = TRUE),
          sess_type = session_type,
          Phase = Phase
) %>% distinct()

##using previous practices to try and predict the following practice session intensity
## also removing stats from the present practice since these would be (somewhat) unknown before training
#still keeping session duration and session_type since those would be known beforehand as well as chronic and acute numbers

average_session_data <- average_session_data %>% arrange(date) %>% mutate(
  lag_session_hard_accels = lag(session_hard_accels),
  lag_session_hard_decels = lag(session_hard_decels),
  lag_session_mod_decels = lag(session_mod_decels),
  lag_session_mod_accels = lag(session_mod_accels),
  lag_session_odometer = lag(session_odometer),
  lag_session_player_load = lag(sess_player_load),
  lag_session_vel_4_dist = lag(session_vel_4_dist),
  lag_session_vel_4_efforts = lag(session_vel_4_efforts),
  lag_session_vel_5_dist = lag(session_vel_5_dist),
  lag_session_vel_5_efforts = lag(session_vel_5_efforts),
  lag_session_vel_6_dist = lag(session_vel_6_dist)
) %>% select(-starts_with("session"))
## duplicate row still in place despite the distinct function
average_session_data <- average_session_data[-c(161), ]

#renaming columns to match with previous model
average_session_data <- average_session_data %>% rename(session_player_load = sess_player_load,
                                                        session_duration = sess_duration,
                                                        session_type = sess_type,
                                                        session_id = sess_id)
# creating dummy variables for Phase
average_session_data <- average_session_data %>% mutate(Phase = ifelse(Phase == "In-Season", 0,
                                                         ifelse(Phase == "JLT Series", 1, 2)))


session_split <- initial_split(average_session_data, strata = session_player_load)
session_train <- training(session_split)
session_test <- testing(session_split)

session_dtrain <- xgb.DMatrix(as.matrix(dplyr::select(session_train,-session_id, -date, -session_player_load)),
                      label = session_train$session_player_load)
session_dtest <- xgb.DMatrix(as.matrix(dplyr::select(session_test,-session_id, -date, -session_player_load)),
                     label = session_test$session_player_load)

watchlist <- list(train = session_dtrain, eval = session_dtest)
#testing
param <- list(max_depth = 4, eta = 0.3, nthread = 2,
              objective = "reg:squarederror", eval_metric = "rmse")

xgb.train(param, session_dtrain, nrounds = 50, watchlist)

#finding appropriate hyperparameters

session_find_rounds <- function(rounds){
  rounds <- floor(rounds)
  xgb.train(
    params = list(
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = rounds,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_rounds, c(2,500), tol = 1) #minimizes at 160

session_find_optimal_depth <- function(max_depth){
  max_depth <- floor(max_depth)
  xgb.train(
    params = list(
      max_depth = max_depth,
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 160,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_optimal_depth, c(2,100), tol = 1) # minimizes at 99

session_find_optimal_child_weight <- function(min_child_weight){
  weight <- floor(min_child_weight)
  xgb.train(
    params = list(
      max_depth = 99,
      min_child_weight = weight,
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 160,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_optimal_child_weight, c(2,70), tol = 1) #minimizes at 52

session_learning_vals <- tibble(
  eta = 1 / (100 *(1:10)),
  nrounds = 160 * 10 * (1:10)
)



session_expand_mod <- function(eta, nrounds) {
  xgb.train(params = list(
    max_depth = 99,
    min_child_weight = 52,
    eta = eta,
    objective = "reg:squarederror",
    eval_metric = "rmse"
  ),
  data = session_dtrain, 
  nrounds = nrounds,
  watchlist = list(train = session_dtrain, test = session_dtest),
  verbose = 1
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}
#### PROBLEMS WITH OVERFITTING ####
(session_performance <- pmap_dbl(session_learning_vals, session_expand_mod))

#### Going to try to use lambda hyperparameter to try and ####
#### fix overfitting ####

session_find_optimal_lambda <- function(lambda){
  lambda <- floor(lambda)
  xgb.train(
    params = list(
      max_depth = 99,
      min_child_weight = 52,
      eta = 0.1,
      lambda = lambda,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 160,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}
optimize(session_find_optimal_lambda, c(1,200), tol = 1) #optimizes at 101



session_learning_vals <- tibble(
  eta = 1 / (100 *(1:10)),
  nrounds = 160 * 10 * (1:10)
)



session_expand_mod <- function(eta, nrounds) {
  xgb.train(params = list(
    max_depth = 99,
    min_child_weight = 52,
    eta = eta,
    lambda = 101,
    objective = "reg:squarederror",
    eval_metric = "rmse"
  ),
  data = session_dtrain, 
  nrounds = nrounds,
  watchlist = list(train = session_dtrain, test = session_dtest),
  verbose = 1
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}
#### STILL ISSUES WITH OVERFITTING BUT BETTER, GONNA TRY OPTIMIZING subsample  ####
(session_performance <- pmap_dbl(session_learning_vals, session_expand_mod))
##rmse minimizes on 1st iteration: rmse = 87.937
#eta = 0.01, rounds = 1600

session_find_optimal_subsample <- function(subsample){
  subsample <- subsample
  xgb.train(
    params = list(
      max_depth = 99,
      min_child_weight = 52,
      eta = 0.1,
      lambda = 101,
      subsample = subsample,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 160,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}
optimize(session_find_optimal_subsample, c(0,1), tol = 1) #optimizes at 0.38, 
                                                          #rmse was significantly worse


#### Rebuilding Model from scratch finding lamda and subsample first####
#using smaller eta to see if that helps results

session_find_optimal_lambda <- function(lambda){
  lambda <- floor(lambda)
  xgb.train(
    params = list(
      eta = 0.1,
      lambda = lambda,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 160,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}
optimize(session_find_optimal_lambda, c(1,200), tol = 1) # minimizes as 50

session_find_optimal_subsample <- function(subsample){
  subsample <- subsample
  xgb.train(
    params = list(
      eta = 0.1,
      lambda = 50,
      subsample = subsample,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 160,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}
optimize(session_find_optimal_subsample, c(0,1), tol = 1) #optimizes at 0.38, 
                                                          #rmse better this time around




session_find_optimal_depth <- function(max_depth){
  max_depth <- floor(max_depth)
  xgb.train(
    params = list(
      max_depth = max_depth,
      eta = 0.1,
      lambda = 50,
      subsample = 0.381996,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 160,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_optimal_depth, c(1,100), tol = 1) # minimizes at 82
#giving different values here on every run through, not sure why
# think it might have to do with subsample parameter

session_find_optimal_child_weight <- function(min_child_weight){
  weight <- floor(min_child_weight)
  xgb.train(
    params = list(
      max_depth = 82,
      min_child_weight = weight,
      eta = 0.1,
      lambda = 50,
      subsample = 0.38,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 160,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_optimal_child_weight, c(1,90), tol = 1) #minimizes at 23

session_learning_vals <- tibble(
  eta = 1 / (100 *(1:10)),
  nrounds = 160 * 10 * (1:10)
)



session_expand_mod <- function(eta, nrounds) {
  xgb.train(params = list(
    max_depth = 82,
    min_child_weight = 23,
    lambda = 50,
    subsample = 0.38,
    eta = eta,
    objective = "reg:squarederror",
    eval_metric = "rmse"
  ),
  data = session_dtrain, 
  nrounds = nrounds,
  watchlist = list(train = session_dtrain, test = session_dtest),
  verbose = 1
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

(session_performance <- pmap_dbl(session_learning_vals, session_expand_mod))
# minimizes at 1st iteration; rmse = 83.20438
#eta = 1/100, nrounds = 1600

#adding colsample_by* to see if that improves performance
session_find_optimal_colsample <- function(colsample){
  colsample <- colsample
  xgb.train(
    params = list(
      max_depth = 82,
      min_child_weight = 23,
      lambda = 50,
      subsample = 0.38,
      colsample_bynode = colsample,
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 160,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_optimal_colsample, c(0,1), tol = 1) #minimizes at 0.381966, rmse = 81.30609
                                                          #tested all colsamples, bynode was the best

session_learning_vals <- tibble(
  eta = 1 / (100 *(1:10)),
  nrounds = 160 * 10 * (1:10)
)



session_expand_mod <- function(eta, nrounds) {
  xgb.train(params = list(
    max_depth = 82,
    min_child_weight = 23,
    lambda = 50,
    subsample = 0.38,
    colsample_bynode = 0.381966,
    eta = eta,
    objective = "reg:squarederror",
    eval_metric = "rmse"
  ),
  data = session_dtrain, 
  nrounds = nrounds,
  watchlist = list(train = session_dtrain, test = session_dtest),
  verbose = 1
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

(session_performance <- pmap_dbl(session_learning_vals, session_expand_mod))
### minimizes on 1st iteration; rmse = 79.91883
## eta = 1/100, nrounds = 1600

#### Trying Without lagged values: only acute and chronic workloads ####
average_session_data <- data2 %>% group_by(`Session ID.y`, date) %>% 
  reframe(session_id = `Session ID.y`,
          acwr_distance = mean(`ACWR Distance`, na.rm = TRUE),
          acwr_hsr_4 = mean(`ACWR HSR - Vel Zone 4 (18-24 km/h)`,na.rm = TRUE),
          acwr_hsr_5 = mean(`ACWR HSR - Vel Zone 5 (24-28 km/h)`, na.rm = TRUE),
          acwr_player_load = mean(`ACWR Player Load`, na.rm = TRUE),
          acute_dist_rolling = mean(`Acute Distance (Rolling Ave)`, na.rm = TRUE),
          acute_hsr_4_rolling = mean(`Acute HRS - Vel Zone 4 (Rolling Ave)`, na.rm = TRUE),
          acute_hsr_5_rolling = mean(`Acute HSR - Vel Zone 5 (Rolling Ave)`, na.rm = TRUE),
          acute_player_load = mean(`Acute Player Load (Rolling Ave)`, na.rm = TRUE),
          chronic_distance_rolling = mean(`Chronic Distance (Rolling Ave)`, na.rm = TRUE),
          chronic_hsr_4_rolling = mean(`Chronic HSR - Vel Zone 4 (Rolling Ave)`, na.rm = TRUE),
          chronic_hsr_5_rolling = mean(`Chronic HSR - Vel Zone 5 (Rolling Ave)`, na.rm = TRUE),
          chronic_player_load_rolling = mean(`Chronic Player Load (Rolling Ave)`, na.rm =TRUE),
          hsr_4_28days = mean(`HSR - Vel Zone 4 (Last 28 Days)`, na.rm = TRUE),
          hsr_4_7days = mean(`HSR - Vel Zone 4 (Last 7 Days)`, na.rm = TRUE),
          hsr_5_28days = mean(`HSR - Vel Zone 5 (Last 28 Days)`, na.rm = TRUE),
          hsr_5_7days = mean(`HSR - Vel Zone 5 (Last 7 Days)`, na.rm = TRUE),
          player_load_28days = mean(`Player Load (Last 28 Days)`, na.rm = TRUE),
          player_load_7days = mean(`Player Load (Last 7 Days)`, na.rm = TRUE),
          session_duration = mean(`Session Duration`, na.rm = TRUE),
          session_player_load = mean(`Session Player Load`, na.rm = TRUE),
          session_type = session_type,
          Phase = Phase
  ) %>% distinct() %>% select(-`Session ID.y`)

#removing duplicate row
average_session_data <- average_session_data[-c(161), ]
#making phase dummy variables
average_session_data <- average_session_data %>% mutate(Phase = ifelse(Phase == "In-Season", 0,
                                                                       ifelse(Phase == "JLT Series", 1, 2)))


session_split <- initial_split(average_session_data, strata = session_player_load)
session_train <- training(session_split)
session_test <- testing(session_split)

session_dtrain <- xgb.DMatrix(as.matrix(dplyr::select(session_train,-session_id, -date, -session_player_load)),
                              label = session_train$session_player_load)
session_dtest <- xgb.DMatrix(as.matrix(dplyr::select(session_test,-session_id, -date, -session_player_load)),
                             label = session_test$session_player_load)
#finding appropriate hyperparameters

session_find_rounds <- function(rounds){
  rounds <- floor(rounds)
  xgb.train(
    params = list(
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = rounds,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_rounds, c(2,500), tol = 1) #minimizes at 40, rmse already looks better

session_find_optimal_depth <- function(max_depth){
  max_depth <- floor(max_depth)
  xgb.train(
    params = list(
      max_depth = max_depth,
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 40,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_optimal_depth, c(2,200), tol = 1) # minimizes at 199

session_find_optimal_child_weight <- function(min_child_weight){
  weight <- floor(min_child_weight)
  xgb.train(
    params = list(
      max_depth = 199,
      min_child_weight = weight,
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 40,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_optimal_child_weight, c(2,100), tol = 1) #minimizes at 32

session_learning_vals <- tibble(
  eta = 1 / (100 *(1:10)),
  nrounds = 40 * 10 * (1:10)
)



session_expand_mod <- function(eta, nrounds) {
  xgb.train(params = list(
    max_depth = 199,
    min_child_weight = 32,
    eta = eta,
    objective = "reg:squarederror",
    eval_metric = "rmse"
  ),
  data = session_dtrain, 
  nrounds = nrounds,
  watchlist = list(train = session_dtrain, test = session_dtest),
  verbose = 1
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}
(session_performance <- pmap_dbl(session_learning_vals, session_expand_mod))
##### model performed better without lagged values; rmse = 69.36804####
# minimized on 1st iteration
# eta = 0.01, rounds = 400
#no overfitting issues

#adding colsample_by* to improve performance
session_find_optimal_colsample <- function(colsample){
  colsample <- colsample
  xgb.train(
    params = list(
      max_depth = 199,
      min_child_weight = 32,
      colsample_bylevel = colsample,
      eta = 0.1,
      objective = "reg:squarederror",
      eval_metric = "rmse"
    ),
    data = session_dtrain,
    nrounds = 40,
    watchlist = list(train = session_dtrain, test = session_dtest),
    verbose = 0
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}

optimize(session_find_optimal_colsample, c(0,1), tol = 1) #minimizes at 0.381966, rmse = 69.71752
#tested all colsamples, bylevel was the best

session_learning_vals <- tibble(
  eta = 1 / (100 *(1:10)),
  nrounds = 40 * 10 * (1:10)
)



session_expand_mod <- function(eta, nrounds) {
  xgb.train(params = list(
    max_depth = 199,
    min_child_weight = 32,
    eta = eta,
    colsample_bylevel = 0.381966,
    objective = "reg:squarederror",
    eval_metric = "rmse"
  ),
  data = session_dtrain, 
  nrounds = nrounds,
  watchlist = list(train = session_dtrain, test = session_dtest),
  verbose = 1
  )$evaluation_log %>% 
    dplyr::select(test_rmse) %>% 
    slice_tail() %>% 
    flatten_dbl()
}
####model performed better without colsample####
(session_performance <- pmap_dbl(session_learning_vals, session_expand_mod))

#### Final Model####
(session_bst <- xgb.train(params = list(
  max_depth = 199,
  min_child_weight = 32,
  eta = 0.01,
  objective = "reg:squarederror",
  eval_metric = "rmse"
),
data = session_dtrain, 
nrounds = 400,
watchlist = list(train = session_dtrain, test = session_dtest),
verbose = 0
))

# Getting summary statistics for the predictions of the model
session_test %>% dplyr::select(-session_id, -session_player_load, -date) %>% 
  as.matrix() %>% 
  predict(object = session_bst) %>% 
  summary()

#Getting Feature Importance
session_importance <- xgb.importance(model = session_bst)

#releveling the features for the Importance Plot
session_importance$Feature <- factor(session_importance$Feature, levels = rev(session_importance$Feature))

# Importance Plot
# I know xgboost has a plotting function, but I wanted to customize it
session_importance %>% ggplot(aes(Gain, Feature)) +
  geom_col(fill = "#0E4D92")+
  theme_classic() +
  xlab("Importance") +
  ylab("Feature") +
  ggtitle("Model 2 Variable Importance")
xgb.plot.importance(xgb.importance(model = session_bst))

# Removing Session Duration and Session Type from the dataframe
# Going to create another importance plot without Session Duration and Session Type
session_importance_wo_session_duration_type <- session_importance %>% 
                                              filter(!Feature %in% c("session_duration", 'session_type'))

# Creating new importance with the remaining variables
session_importance_wo_session_duration_type <- session_importance_wo_session_duration_type %>% 
  mutate(new_importance = Gain / sum(session_importance_wo_session_duration_type$Gain))

# Releveling features for importance chart
session_importance_wo_session_duration_type$Feature <- factor(session_importance_wo_session_duration_type$Feature, 
                                                              levels = rev(session_importance_wo_session_duration_type$Feature))
# Importance Chart #2
session_importance_wo_session_duration_type %>% ggplot(aes(new_importance, Feature)) +
  geom_col(fill = "#0E4D92")+
  theme_classic() +
  xlab("Importance") +
  ylab("Feature") +
  ggtitle("Model 2 Variable Importance", subtitle = "Without Session Type & Session Duration") +
  theme(plot.subtitle = element_text(size = 9))

# Adding predicted session player load to the data frame
average_session_data <- average_session_data %>% 
  mutate(pred_session_average_player_load = predict(session_bst,
                                                    xgb.DMatrix(as.matrix(dplyr::select(average_session_data,-session_id, -session_player_load, -date)),
                                                                                                         label = average_session_data$session_player_load)))

# Adding column for residuals for my own analysis
average_session_data <- average_session_data %>% mutate(residuals = session_player_load - pred_session_average_player_load)

average_session_data <- average_session_data %>% select(session_id, pred_session_average_player_load)

# Joining Player Session Load Data with Mean Session Load Data
final_data <- left_join(xg_model_data, average_session_data, by = c("Session ID.y" = "session_id"))

#Creating Load Over Expected Statistic
final_data <- final_data %>% mutate(load_OE = pred_session_player_load - pred_session_average_player_load) %>%
                             mutate(load_OE_scaled = scale(load_OE))

# Density Plot of Load Over Expected (Scaled) Statistic
final_data %>% ggplot(aes(load_OE_scaled)) +
  geom_density(color = "#0E4D92") +
  theme_classic() +
  xlab("Load Over Expected (Scaled)") +
  ylab("") +
  ggtitle("Load Over Expected Density Plot")

# Removes any observations where Soreness or Wellness Data z-scores were 0
final_data_adjusted <- final_data %>% filter(`Energy z-score` != 0 |
                                               `Calf z-score` != 0 |
                                               `Leg Heaviness z-score` != 0 |
                                               `Mental State z-score` != 0 |
                                               `Sleep Quality z-score` != 0|
                                               `Calf z-score` != 0 |
                                               `General Soreness z-score` != 0 |
                                               `Glute z-score` != 0 |
                                               `Groin z-score` != 0 |
                                               `Hamstring z-score` != 0 |
                                               `Lower Back z-score` != 0 |
                                               `Quad z-score` != 0 |
                                               `Total Muscle Soreness z-score` != 0)

#Gives Season Level Data for Each Athlete
overworked_data <- final_data %>% group_by(`Entity Id`) %>% reframe(load_OE_scaled = mean(load_OE_scaled, na.rm = TRUE),
                                                                    energy_z_score = mean(`Energy z-score`, na.rm =TRUE),
                                                                    mental_z_score = mean(`Mental State z-score`, na.rm = TRUE),
                                                                    leg_heaviness_z_score = mean(`Leg Heaviness z-score`, na.rm = TRUE),
                                                                    sleep_quality_z_score =  mean(`Sleep Quality z-score`, na.rm = TRUE),
                                                                    chronic_player_load = mean(`Chronic Player Load (Rolling Ave)`, na.rm = TRUE),
                                                                    calf_z_score = mean(`Calf z-score`, na.rm = TRUE),
                                                                    foot_z_score = mean(`Foot z-score`, na.rm = TRUE),
                                                                    general_soreness_z_score = mean(`General Soreness z-score`, na.rm = TRUE),
                                                                    distance_28_days = mean(`Distance (Last 28 Days)`, na.rm. =TRUE),
                                                                    acute_player_load = mean(`Acute Player Load (Rolling Ave)`, na.rm = TRUE),
                                                                    glute_z_score = mean(`Glute z-score`, na.rm = TRUE),
                                                                    groin_z_score = mean(`Groin z-score`, na.rm = TRUE),
                                                                    hamstring_z_score = mean(`Hamstring z-score`, na.rm = TRUE),
                                                                    lower_back_z_score = mean(`Lower Back z-score`, na.rm = TRUE),
                                                                    quad_z_score = mean(`Quad z-score`, na.rm = TRUE),
                                                                    total_muscle_soreness_z_score = mean(`Total Muscle Soreness z-score`, na.rm = TRUE))
