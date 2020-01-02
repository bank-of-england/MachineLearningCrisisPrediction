# SUPPLEMENTARY CODE FOR BOE SWP 848: 
# Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

# This script collects the results of the forecasting experiments and produces figures.

source("utils_analysis.R")
library(grDevices)
library(dplyr)

folder_data <- "../results/forecasting_up/"
folder_figures <- "../figures/"

all_files = list.files(folder_data) # list of all files in the folder

# load data
dataset <- read.csv(paste0(folder_data,grep("data_", all_files, value = T, fixed = T)[1]),
                    sep = "\t", stringsAsFactors = F)[,-1]
n_objects <- nrow(dataset)
all_years <- unique(dataset$year)
countries <- unique(dataset$iso)
n_countries <- length(countries)
true_class <- dataset$crisis

years_last_train <- as.numeric(sapply(unlist(all_files), function(x) substr(sapply(strsplit(x, "year"), "[", 2),1,4)))
years_last_train <- sort(unique(years_last_train))

prediction_files <- sapply(years_last_train, function(y) grep_multi(c("pred_", paste0("year",y)), all_files))
names(prediction_files) <- years_last_train

pred_example <- read.csv(paste0(folder_data, prediction_files[1]), sep = "\t")[,-1]
algos <- setdiff(colnames(pred_example), c("year", "iso", "iter", "crisis", "index", "fold"))


# Create a single data set that includes the recurisve forecasting predictions
# We need to have a two year gap between the last training observations and the first test observation.
# The reason is that, for instance in training in 1990 we already know that 1992 is a crisis, due to the coding of the outcome.
# Therefore, we earliest date we can predict with training data from 1990 is 1992
predictions_mean <- NULL
for (i in length(years_last_train):1){
  cat("|")
  year <- years_last_train[i]
  dat <- read.csv(paste0(folder_data, prediction_files[as.character(year)]), sep = "\t")[,-1]
  dat$index <- rep(1:n_objects, max(dat$iter) + 1)
  dat <- dat[dat$year > year + 1,]  # skip the first two years in the test set
  dat <- dat[!dat$year %in% unique(predictions_mean$year),]# only predict the next observations that are not covered by a more recent training set
  
  
  dat_out <- dat[dat$iter == 0,]
  # average predictions
  dat_mean <- dat %>%
    group_by(index) %>% 
    select(c(index, !!algos)) %>% 
    summarise_if(is.numeric, mean) %>%
    ungroup() %>% select(-index) %>% data.frame()  
  
  dat_out[,colnames(dat_mean)] <- dat_mean
  
  
  predictions_mean <- rbind(predictions_mean, dat_out)
  
}

predictions_mean <- predictions_mean[order(predictions_mean$iso, predictions_mean$year),]
predictions_mean <- predictions_mean[predictions_mean$year > 1945,]# only consider observations after WW2
all_years <- min(predictions_mean$year):max(predictions_mean$year) 

#### AUC performance ####


ix_year <- predictions_mean$year > 1000 # performance on all years after 2000
sapply(predictions_mean[, algos], function(x) measurePerf(predictions_mean$crisis[ix_year],x[ix_year],threshold = .5))



#### Plot prediction matrix ####

minimum_hitrate <- .8
model <- "logreg"
all_thresholds <- sort(unique(predictions_mean[,model]))
performance_thresholds <- sapply(all_thresholds, function(x) measurePerf(
  predictions_mean$crisis, 
  predictions_mean[,model],
  threshold = x))
threshold_ix <- which.min(abs(performance_thresholds["tp.rate",] - minimum_hitrate))
threshold <- all_thresholds[threshold_ix]

col_hit <- "forestgreen"
col_tn <- "darkseagreen1"
col_fa <- "gray75"
col_miss <- "red"

cairo_pdf(paste0(folder_figures, "forecast_predmat_",model, ".pdf"), height = 6, width = 10)
cx <- 0.7
par(mar = c(3,7,3,1))
plot.new()
plot.window(xlim = c(min(all_years), max(all_years)+4), ylim = c(1,n_countries+ .5))
axis(1, at = min(all_years):max(all_years), labels = NA, las = 2, cex.axis = 1, tck = -0.01)
axis(1, at = (min(all_years):max(all_years))[seq(1,length((min(all_years):max(all_years))),5)], las = 2, cex.axis = 1)
axis(2, at = 1:n_countries, labels = country_names_print[countries], las = 2)
par(xpd = T)
legend(x = 2008,y = 20.7, legend = c("Correct crises","Correct non-crises", "Missed crises", "False alarms"),
       pt.bg = c(col_hit, col_tn, col_miss, col_fa), col = c(col_hit, col_tn, col_miss, col_fa),
       pch = c(16,22, 24,25), bty = "n", y.intersp = 0.8)
par(xpd = F)

counter <- 0
for(country in countries){
  counter <- counter + 1
  abline(h = counter, lty = 3, col = "gray50", lwd = 0.6)
  ix = predictions_mean$iso == country
  pred <- predictions_mean[ix, model]
  crit <- predictions_mean[ix,"crisis"]
  years <- predictions_mean[ix,"year"]
  yearsadd <- setdiff(all_years,predictions_mean[ix,"year"])
  
  years <- c(years, yearsadd)
  oo <- order(years)
  years <- years[oo]
  crit <- c(crit,rep(NA, length(yearsadd)))[oo]
  pred <- c(pred,rep(NA, length(yearsadd)))[oo]
  
  ix_hit = crit == 1 & pred >= threshold; ix_hit[is.na(ix_hit)] <- F
  ix_miss = crit == 1 & pred < threshold; ix_miss[is.na(ix_miss)] <- F
  ix_fa = crit == 0 & pred >= threshold; ix_fa[is.na(ix_fa)] <- F
  ix_tn = crit == 0 & pred < threshold; ix_tn[is.na(ix_tn)] <- F
  n_all <- sum(ix_hit) + sum(ix_miss) + sum(ix_fa) + sum(ix_tn)
  lines(years,pred + counter, type = "o", pch = 20, cex =.3) 
  points(years[ix_hit], rep(counter + 0.5, sum(ix_hit)), col = col_hit, pch = 16, cex = cx + 0.2)
  points(years[ix_miss], rep(counter + 0.5, sum(ix_miss)), col = col_miss, bg = col_miss, pch = 24, cex = cx)
  points(years[ix_fa], rep(counter + 0.5, sum(ix_fa)), col = col_fa, bg = col_fa, pch = 25, cex = cx - .05)
  cols <-  c(col_hit,col_tn, col_fa, col_miss)
  xses <- c(sum(ix_hit), sum(ix_tn), sum(ix_fa), sum(ix_miss))
  cols <- cols[xses!=0]
  xses <- xses[xses!=0]
  plotrix::floating.pie(xpos = max(years) + 4, ypos = counter + .5, x = xses, col = cols, radius = 1.2, edges = 1000)
}
dev.off()








