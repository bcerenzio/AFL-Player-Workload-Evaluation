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
library(stargazer)
set.seed(101)

MuscleSoreness <- read_excel("MuscleSoreness.xlsx")
#creating a new date variable to join on
MuscleSoreness <- MuscleSoreness %>% mutate(date = date(ymd_hms(Start)))
MuscleSoreness <- MuscleSoreness %>% distinct()
Wellness <- read_excel("Wellness.xlsx")
Wellness <- Wellness %>% mutate(date = date(ymd_hms(Start)))
Wellness <- Wellness %>% distinct()
Catapult <- Catapult %>% mutate(date = date(ymd_hms(Start)))
Catapult <- Catapult %>% distinct()

#joining data
data <- Wellness %>% 
  left_join(y = Catapult, by = c("Entity Id", "date")) %>% 
  left_join(y = MuscleSoreness, by = c("Entity Id", "date"))
#filtering out data where session load is 0
data2 <- data %>% filter(`Session Player Load` > 0)
#creating z score calculation to filter out potential bad data collection
data2 <- data2 %>% mutate(z_score_session_load = scale(`Session Player Load`))
#using z-score to remove obscene outliers
data2 <- data2 %>% filter(abs(z_score_session_load) <= 7)
#rescaling to see if old outliers shielded other outliers from showing themselves. It didn't. 
data2 <- data2 %>% mutate(z_score_session_load = scale(`Session Player Load`))

#using this function so I can autofill columns when I create the model since column names are very messy
new_lm <- function(data, formula) {
  lm(formula, data)
}

#adding to phase column to fill in NA values. Got dates from the play-by-play data
data2 <-  data2 %>% mutate(Phase = ifelse(ymd(date) %within% interval(ymd('2017-03-23'), ymd('2017-09-30')) | 
                                            ymd(date) %within% interval(ymd('2018-03-22'), ymd('2018-09-29')), "In-Season",
                                          ifelse(ymd(date) %within% interval(ymd('2018-02-15'), ymd('2018-02-17')),"AFLX",
                                                 ifelse(ymd(date) %within% interval(ymd('2018-02-24'), ymd('2018-03-11')) | 
                                                          ymd(date) %within% interval(ymd('2017-02-16'), ymd('2018-03-12')),"JLT Series","Pre-Season"))))

#creating a new data frame with the mean player load for each session
sessions1 <- data2 %>% group_by(`Session ID`) %>% reframe(`Session Player Load` = mean(`Session Player Load`))

#create plot of number of clusters vs total within sum of squares to find ideal number of clusters
fviz_nbclust(as.data.frame(sessions1$`Session Player Load`), kmeans, method = "wss") #ideal 4
fviz_nbclust(as.data.frame(sessions1$`Session Player Load`), kmeans, method = "silhouette") #ideal is 2
fviz_nbclust(as.data.frame(sessions1$`Session Player Load`), kmeans, method = "gap_stat") # ideal is 4

sessions1$session_type <- kmeans(sessions1$`Session Player Load`, 4)$cluster
summary(sessions1)
sessions1 %>% group_by(session_type) %>% reframe(`Session Player Load` = mean(`Session Player Load`))
average_player_load_by_session_type <- sessions1 %>% group_by(session_type) %>% reframe(`Session Player Load` = mean(`Session Player Load`))
average_player_load_by_session_type$rank <- rank(average_player_load_by_session_type$`Session Player Load`)

average_player_load_by_session_type <- average_player_load_by_session_type %>% dplyr::select(-`Session Player Load`)
#joining average player load onto sessions1 so that each rank is paired with session id
sessions1 <- left_join(sessions1, average_player_load_by_session_type, by = "session_type")
#removes the orignal value given by the kmeans function and replaces it with the rank based on
#session intensity
sessions1 <- sessions1 %>% dplyr::select(rank, `Session ID`)
#renames rank column
sessions1 <- sessions1 %>% mutate(session_type = rank) %>% dplyr::select(-rank)
#adding session type column to dataset
data2 <- left_join(data2, sessions1, by = c("Session ID"))

#selecting necessary variables and dropping all rows with NA values
data3 <- data2 %>% dplyr::select(`Session Player Load`, `Total Muscle Soreness z-score`,
                                 `Leg Heaviness z-score`, session_type , 
                                 `Acute Player Load (Rolling Ave)` , `Session Duration` , `Distance (Last 7 Days)` , 
                                 Phase) %>% drop_na()
# first linear model
model <- data3 %>% new_lm(`Session Player Load` ~ `Total Muscle Soreness z-score` 
                          + `Leg Heaviness z-score` +  session_type +
                            `Acute Player Load (Rolling Ave)` + `Session Duration` + `Distance (Last 7 Days)` + Phase)


summary(model)
vif(model)
durbinWatsonTest(model, alternative = c("two.sided")) #autocorrelation present
#Newey West Method
robust_se <- coeftest(model, vcov = NeweyWest((model)))
summary(robust_se)
#Leg Heaviness and Total Muscle Soreness no longer significant
(robust_se <- as.array(robust_se))
#getting se for stargazer
nw_se <- sqrt(diag(NeweyWest(model)))
stargazer(model, type = "html", out = 'model.html',
          covariate.labels = c("Total Muscle Soreness z-score", 'Leg Heaviness z-score',
                               'Session Type', 'Acute Player Load', 'Session Duration', 'Distance (Last 7 Days)', 
                               'JLT Series', 'Pre-Season'),
          column.labels = c("Session Player Load"),
          se = list(nw_se),
          dep.var.labels.include = FALSE,
          title = c("Muscle Soreness and Leg Heaviness Effect on Player Session Load"),
          notes = "Robust Standard Errors Used to Account for Autocorrleation (Newey West Method)",
          notes.append = TRUE,
          notes.align = "c")

#filtering out for only the most intensive practice; session_types 3 & 4
data4 <- data3 %>% filter(session_type >= 3)
#revaluing session_type
data4 <- data4 %>% mutate(session_type = ifelse(session_type == 3, 1,2))
#new model
new_model <- data4 %>% new_lm(`Session Player Load` ~ `Total Muscle Soreness z-score` 
                              + `Leg Heaviness z-score` +  session_type +
                                `Acute Player Load (Rolling Ave)` + `Session Duration` + `Distance (Last 7 Days)` + 
                                Phase)
vif(new_model)
summary(new_model)
durbinWatsonTest(new_model, alernative = 'two.sided') #autocorrelation present
new_robust_se <- coeftest(new_model, vcov = NeweyWest((new_model)))
summary(new_robust_se)
(new_robust_se <- as.array(new_robust_se)) #no significance
nw_se1 <- sqrt(diag(NeweyWest(new_model)))
stargazer(new_model, type = "html", out = 'high_intensity_model.html',
          covariate.labels = c("Total Muscle Soreness z-score", 'Leg Heaviness z-score',
                               'Session Type', 'Acute Player Load', 'Session Duration', 'Distance (Last 7 Days)', 
                               'JLT Series', 'Pre-Season'),
          column.labels = c("Session Player Load"),
          se = list(nw_se1),
          dep.var.labels.include = FALSE,
          title = c("Muscle Soreness and Leg Heaviness Effect on Player Session Load (High Intensity Sessions)"),
          notes = "Robust Standard Errors Used to Account for Autocorrleation (Newey West Method)",
          notes.append = TRUE,
          notes.align = "c")


#filtering out for only the least intensive practice; session_types 1 & 2
data4 <- data3 %>% filter(session_type < 3)

#new model
new_model <- data4 %>% new_lm(`Session Player Load` ~ `Total Muscle Soreness z-score` 
                              + `Leg Heaviness z-score` +  session_type +
                                `Acute Player Load (Rolling Ave)` + `Session Duration` + `Distance (Last 7 Days)` + 
                                Phase)
vif(new_model)
summary(new_model)
durbinWatsonTest(new_model, alernative = 'two.sided') #autocorrelation present
new_robust_se <- coeftest(new_model, vcov = NeweyWest((new_model)))
summary(new_robust_se)
(new_robust_se <- as.array(new_robust_se))
nw_se2 <- sqrt(diag(NeweyWest(new_model)))
#Leg Heaviness and Total Muscle Soreness are significant in lower intensity practices
stargazer(new_model, type = "html", out = 'low_intensity_model.html',
          covariate.labels = c("Total Muscle Soreness z-score", 'Leg Heaviness z-score',
                               'Session Type', 'Acute Player Load', 'Session Duration', 'Distance (Last 7 Days)', 
                               'JLT Series', 'Pre-Season'),
          column.labels = c("Session Player Load"),
          se = list(nw_se2),
          dep.var.labels.include = FALSE,
          title = c("Muscle Soreness and Leg Heaviness Effect on Player Session Load (Low Intensity Sessions)"),
          notes = "Robust Standard Errors Used to Account for Autocorrleation (Newey West Method)",
          notes.append = TRUE,
          notes.align = "c")



