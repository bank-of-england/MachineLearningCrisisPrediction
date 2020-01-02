# SUPPLEMENTARY CODE FOR BOE SWP 848: 
# Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

# This script collects the results of the cross-validation experiments and produces figures.

source("utils_analysis.R")
library(grDevices)
library(dplyr)

folder_data <- "../results/baseline/" # where the results are found
folder_figures <- "../figures/" # where the figures are placed

all_files = list.files(folder_data) # list of all files in the folder

# load dataset
dataset <- read.csv(paste0(folder_data,grep("data_", all_files, value = T, fixed = T)[1]), sep = "\t", stringsAsFactors = F)[,-1]
n_objects <- nrow(dataset)
all_years <- min(dataset$year): max(dataset$year)
countries <- unique(dataset$iso)
n_countries <- length(countries)
true_class <- dataset$crisis

# load predictions
predictions <- read.csv(paste0(folder_data,grep("all_pred", all_files, value = T, fixed = T)[1]), sep = "\t", stringsAsFactors = F)[,-1]
algos <- setdiff(colnames(predictions), c("year", "iso", "iter", "crisis", "index", "fold"))
n_replications <- nrow(predictions)/n_objects
n_folds <- max(predictions$fold)
predictions$index <- rep(1:n_objects, n_replications)


# load performance results
auc_mean <- read.csv(paste0(folder_data,grep_multi("mean_fold",all_files)),
                     sep = "\t")[,"auc"]; names(auc_mean) <- algos
auc_se <- read.csv(paste0(folder_data,grep_multi("se_fold",all_files)),
                   sep = "\t")[,"auc"]; names(auc_se) <- algos
auc_iter <- read.csv(paste0(folder_data,grep_multi("mean_fold",all_files)),
                     sep = "\t")[,"iter"]; names(auc_iter) <- algos
auc_summary <- cbind(auc_mean, auc_se, auc_iter)




# compute mean prediction across all iterations
predictions_mean <- predictions[predictions$iter ==0,]
predictions_mean[, algos] <- predictions %>%
  group_by(index) %>%
  select(c(index, !! algos)) %>%
  summarise_all(mean) %>%
  ungroup() %>% select(-index) %>% data.frame()


algos_shapley <- setdiff(algos, "r_c50") # we cannot produce Shapley values for the model trained in R (r_c50)
shapleys <- lapply(algos_shapley, function(algo)
  read.csv(paste0(folder_data,grep(paste0("shapley_append_", algo), all_files, value = T, fixed = T)[1]),
           sep = "\t", stringsAsFactors = F)[,-1])
features <- setdiff(colnames(shapleys[[1]]), c("year", "iso", "crisis", "pred"))
shapleys_mean <- lapply(algos_shapley, function(algo)
  read.csv(paste0(folder_data,grep(paste0("shapley_mean_", algo), all_files, value = T, fixed = T)[1]),
           sep = "\t", stringsAsFactors = F)[,-1])

names(shapleys) <- names(shapleys_mean) <-  algos_shapley



algos_show <- algos # the user may want to subset the models she wants to show
algos_show_shapley <- algos_shapley
features_show <- c("drate", "tloan_gdp_rdiff2", "global_drate", "global_loan2")# the user may want to subset the features she wants to show
minimum_hitrate <- 0.8 # main model


#### Plot prediction matrix ####
model <- "extree"
# try all threshold on the predicted probability and pcik the one that is closest to our minimum hit rate.
all_thresholds <- sort(unique(predictions_mean[,model]))
performance_thresholds <- sapply(all_thresholds, function(x) measurePerf(predictions_mean$crisis,
                                                                         predictions_mean[,model], threshold = x))
threshold_ix <- which.min(abs(performance_thresholds["tp.rate",] - minimum_hitrate))
threshold <- all_thresholds[threshold_ix]

col_hit <- "forestgreen"
col_tn <- "darkseagreen1"
col_fa <- "gray75"
col_miss <- "red"

cairo_pdf(paste0(folder_figures, "prediction_matrix_",model, ".pdf"), height = 6, width = 10)
cx <- 0.7
par(mar = c(3,7,3,1))
plot.new()
plot.window(xlim = c(min(all_years), max(all_years)+4), ylim = c(1,n_countries+ .5))
axis(1, at = min(all_years):max(all_years), labels = NA, las = 2, cex.axis = 1, tck = -0.01)
axis(1, at = (min(all_years):max(all_years))[seq(1,length((min(all_years):max(all_years))),5)], las = 2, cex.axis = 1)
axis(2, at = 1:n_countries, labels = country_names_print[countries], las = 2)
par(xpd = T)
legend(x = 1990,y = 20.5,
       legend = c("Correct crises","Correct non-crises", "Missed crises", "False alarms"),
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
  lines(years,pred + counter, type = "o", pch = 20, cex =.3) # predicted prob
  points(years[ix_hit], rep(counter + 0.5, sum(ix_hit)), col = col_hit, pch = 16, cex = cx + 0.2)
  points(years[ix_miss], rep(counter + 0.5, sum(ix_miss)), col = col_miss, bg = col_miss, pch = 24, cex = cx)
  points(years[ix_fa], rep(counter + 0.5, sum(ix_fa)), col = col_fa, bg = col_fa, pch = 25, cex = cx - .05)
  cols <-  c(col_hit,col_tn, col_fa, col_miss)
  xses <- c(sum(ix_hit), sum(ix_tn), sum(ix_fa), sum(ix_miss))
  cols <- cols[xses!=0]
  xses <- xses[xses!=0]
  plotrix::floating.pie(xpos = max(years) + 4, ypos = counter + .5, x = xses, col = cols, radius = 2.3, edges = 1000)
}
dev.off()



#### ROC curves ####

# compute all the points in the ROC space
folds <- predictions$fold
roc_points <- list()
tp_rates = seq(0.0, 1, by = 0.05)
areas_all <- array(NA, dim = c(2500, length(algos_show),2))
dimnames(areas_all) <- list(NULL, algos_show, c("auroc", "aupr"))

for(m in algos_show){
  pred_single <- predictions[[m]]
  true_cat <- rep(true_class, n_replications)
  out_single_mod <- array(NA, dim = c(n_replications, n_folds, length(tp_rates) + 2,3))
  counter <- 0
  for(i in 1:n_replications){
    ix_rep <- (1:n_objects)+(i-1)*n_objects
    for(f in 1:n_folds){ # compute for each folds
      counter <- counter + 1
      ix <- ix_rep[folds[ix_rep]== f]
      areas_all[counter,m, ] <- measurePerf(true_cat[ix], pred_single[ix])[c("auc", "aupr")]

      unique_thresholds <- sort(unique(pred_single[ix]))
      unique_thresholds <- unique(c(0, unique_thresholds, 1))
      pp <- sapply(unique_thresholds, function(t) rocPoint(true_cat[ix], pred_single[ix], t))
      pp <- pp[,order(pp[2,])]

      ix_p <- sapply(tp_rates,function(x)which.min(abs(pp[1,] - x))[1])
      out_single_mod[i,f,,] <- t(cbind(c(0,0,NA),pp[,ix_p], c(1,1, NA)))

    }
  }
  out_single_mod <- apply(out_single_mod ,c(1,3,4), mean, na.rm = T)
  roc_points[[m]] <- apply(out_single_mod,2:3, mean, na.rm = T)
}

# plot ROC cruve
cairo_pdf(paste0(folder_figures, "roc_curve.pdf"), height = 5, width = 5)
par(mar = c(3,3,0.5,0.5))
plot.new()
plot.window(xlim = c(0,1), ylim = c(0,1))
axis(1); axis(2)
abline(0,1, lty = 2, col = "gray50")
title(xlab = "False positive rate", ylab = "True positive rate", line = 2)
for(m in algos_show){
  x <- roc_points[[m]][,2]
  y <- roc_points[[m]][,1]
  ix <- !is.na(x)
  lines(x[ix],y[ix], col = algos_col[m], pch = algos_pch[m], type = "o", lwd = 1.5)
}
ordr <- order(auc_summary[algos_show, "auc_mean"], decreasing = T)
legend("bottomright",
       legend = paste0(algos_names_print[algos_show[ordr]]),
       col = algos_col[algos_show[ordr]], pch = algos_pch[algos_show[ordr]], bty = "n", y.intersp = 0.9, lty = 1, lwd = 2)
dev.off()



# plot Precision recall curve
cairo_pdf(paste0(folder_figures, "precision_recall.pdf"), height = 5, width = 5)
par(mar = c(3,3,0.5,0.5))
plot.new()
plot.window(xlim = c(0,1), ylim = c(0,1))
axis(1); axis(2)
#abline(0,1, lty = 2, col = "gray50")
title(xlab = "Recall", ylab = "Precision", line = 2)
for(m in algos_show){
  x <- roc_points[[m]][,1]
  y <- roc_points[[m]][,3]
  ix <- !is.na(x)
  lines(x[ix],y[ix], col = algos_col[m], pch = algos_pch[m], type = "o", lwd = 1.5)
  abline(h = mean(predictions_mean$crisis), lty = 2, col = "gray50")
}

legend("topright", legend = algos_names_print[algos_show[ordr]],
       col = algos_col[algos_show[ordr]],
       pch = algos_pch[algos_show[ordr]], bty = "n", y.intersp = 0.8, lty = 1, lwd = 2)
dev.off()



#### Plot mean absolute Shapley values ####


mean_shap_abs <- sapply(shapleys, function(x) colMeans(abs(x[,features]))) # mean absolute values
mean_shap_abs <- apply(mean_shap_abs, 2, function(x) x/sum(x)) # normalise such that mean Shapley vlaues sum to 1 for each model.

cairo_pdf(paste0(folder_figures, "mean_absolute_shapley.pdf"), width = 7, height = 5)
oo <- order(mean_shap_abs[,"extree"], decreasing = T)
par(mar = c(8,4,2,1))
plot.new()
plot.window(xlim = c(1, length(features)), ylim = c(0,.25)) #ylim = c(0, max(pp_allModels)))
axis(2)
#axis(1, at = 1:length(feature_names), labels = features_names_print[oo] , las = 2)
text(1:length(features), -.03, srt = 60, adj= 1, xpd = TRUE, labels = features_names_print[features[oo]], cex=1)
axis(1, at = 1:length(features), labels = NA)
title(ylab = "Mean absolute Shapley values \n(normalized)", line = 2)
legend("topright", legend = algos_names_print[algos_show_shapley],
       col = algos_col[algos_show_shapley], pch = algos_pch[algos_show_shapley],
       lty = algos_lty[algos_show_shapley], bty = "n", y.intersp = 0.8, cex = 1.2)
for(m in algos_show_shapley)
  lines(mean_shap_abs[oo,m], col = makeTransparent(algos_col[m], alpha = .75),
        pch = algos_pch[m], type = ifelse(is.na(algos_lty[m] == 1), "p", "o"), cex = 1.5, lwd = 1.6)
dev.off()


#### Plot Shapley scatter plots ####
for (f in features) {
  cairo_pdf(paste0(folder_figures, "scatter_", f, "_", model, ".pdf"), width = 4, height = 4)
  # png(paste0(plot_folder,"correlation/corr_feature_shap_",m,"_", f, ".png"), width = 400, height = 400)
  par(mar = c(3,3,2,0.5))
  feature_values = dataset[,f]
  shap_values <- shapleys_mean[[model]][,f]

  plot.new()
  plot.window(xlim = minmax(feature_values), ylim = minmax(shap_values))
  axis(1); axis(2)
  ix <- true_class == 1
  points(feature_values[!ix], shap_values[!ix], pch = 20, cex = .75,
         col = makeTransparent("gray50", alpha = 0.5))# first non-crises
  points(feature_values[ix], shap_values[ix], pch = 20, cex = .75,
         col = makeTransparent("red",alpha = 0.5))# then crises
  model_degree1 <- lm(shap_values ~ feature_values)
  ol = order(feature_values, decreasing = T)
  lines(feature_values[ol], fitted(model_degree1)[ol], col = "black")

  model_degree3 <- lm(shap_values ~ poly(feature_values, 3))
  lines(feature_values[ol], fitted(model_degree3)[ol], col = "blue")

  abline(h = 0, lwd = 0.5, col = "gray50")
  title(xlab = "Predictor values", ylab = "Shapley values", line = 1.8)
  title(main = features_names_print[f], line = 1)

  legpos <- ifelse(f %in% c("drate", "global_drate", paste0("cpi_pdiff2")), "topright", "topleft")

  legend(legpos, bty = "n", text.col = c("black","blue", "black", "black"), pch = c(NA, NA, 20, 20), legend =
           c(as.expression(bquote(R['degree = 1']^2 == .(round(cor(shap_values, feature_values)^2,2)))),
             as.expression(bquote(R['degree = 3']^2  == .(round(cor(shap_values, model_degree3$fitted.values)^2,2)))),
             "Crisis", "Non-crises"), col = c(NA, NA, "red", "gray50")
  )
  dev.off()
}

#### Plot Shapley values for a specific country ####

alpha = 0.75
intercept <- mean(shapleys_mean[[model]][,"pred"])  - mean(rowSums(shapleys_mean[[model]][, features]))

for(country in countries){
  df <- shapleys_mean[[model]]
  ix = df$iso == country
  years_span <- min(df$year):max(df$year)
  lw = 3.7
  ylim <- c(-0.1,.6)
  cairo_pdf(paste0(folder_figures, "country_", country, "_" , model, "_shap.pdf"), height = 4, width = 8)
  par(mar = c(4,3,1,0))
  plot.new()
  plot.window(xlim = c(minmax(years_span)), ylim = ylim)
  title(main = country_names_print[country])
  title(ylab = "Predicted probability of crisis")

  years_x <- sort(years_span[years_span%%5==0])
  axis(1, las = 2, at = years_x, labels = years_x); axis(2)

  if(sum(df$crisis[ix])> 0)
    segments(x0 = df$year[ix][df$crisis[ix]== 1],y0 = ylim[1], y1 = ylim[2],
             col = makeTransparent("red",alpha = 0.2), lwd = 2, lend = 1) # crisis
  abline(h = intercept, lty = 2, lwd = 0.5)

  minpos = rep(intercept, sum(ix))
  maxpos = rep(intercept, sum(ix))
  for(f in features_show){
    ycurrent = ifelse(df[ix,f]>0, maxpos,minpos)
    maxpos <- ifelse(df[ix,f]>0, maxpos +df[ix,f], maxpos)
    minpos <- ifelse(df[ix,f]<0, minpos +df[ix,f], minpos)
    segments(x0= df$year[ix], y0 = ycurrent, y1 = ycurrent + df[ix,f],
             col = makeTransparent(features_col[f], alpha=alpha), lwd = lw, lend = 1)
  }

  rest_shapley <- rowSums(df[ix, setdiff(features, features_show), drop = F])
  ycurrent = ifelse(rest_shapley>0, maxpos,minpos)
  maxpos <- ifelse(rest_shapley>0, maxpos + rest_shapley, maxpos)
  minpos <- ifelse(rest_shapley<0, minpos + rest_shapley, minpos)
  segments(x0= df$year[ix], y0 = ycurrent, y1 = ycurrent + rest_shapley,
           col = makeTransparent("gray50",alpha=alpha), lwd = lw, lend = 1)

  pp <- cbind(df$year[ix],df$pred[ix])
  add <- setdiff(min(years_span):max(years_span), pp[,1])
  pp <- rbind(pp, cbind(add, rep(NA,length(add))))
  pp <- pp[order(pp[,1]),]
  lines(pp[,1], pp[,2], type = "o", col = makeTransparent("black",alpha = 1),
        lwd = 0.9, pch = 20, cex = 1) # pred values
  legend("topleft", legend = c(features_names_print[features_show], "Predicted value"),
         col = c(features_col[features_show], "black"), pch = c(rep(15, length(features_show)), 16), bty = "n", y.intersp = 0.8)
  dev.off()
}



##### Regression analyses ####


# Logistic regression fitted to the whole sample #

dataset_regression <- dataset
dataset_regression[,features] <- scale(dataset_regression[,features])

model_logit <- glm(crisis~., dataset_regression[,c("crisis", features)],
                   family = binomial(link = logit))
summary(model_logit)
log_coef <- model_logit$coefficients[-1]* apply(dataset_regression[,features], 2, sd)


# Shapley regression #
data_shap_regression <- shapleys[[model]]

model_shapley <- glm(data.frame(crisis = rep(true_class, n_replications),
                                data_shap_regression[,features]),
                     family = binomial(link = logit))

model_shapley <- clrobustse(model_shapley, rep(1:n_objects, n_replications)) # robust standard errors
model_shapley[,"Estimate"][-1] * apply(data_shap_regression[,features], 2, sd) # standardise weights




