# Statistical Helper Functions for R
# Author: [Your Name]
# Date: 2024
#
# Reusable functions for common statistical operations in research.

#' Calculate Robust Standard Errors
#'
#' Computes heteroskedasticity-robust (HC) standard errors for regression models
#'
#' @param model A fitted lm or glm model object
#' @param type Type of robust standard errors (HC0, HC1, HC2, HC3)
#' @return A matrix with coefficients and robust standard errors
#' @export
robust_se <- function(model, type = "HC3") {
  # Check if required packages are available
  if (!requireNamespace("sandwich", quietly = TRUE)) {
    stop("Package 'sandwich' is required but not installed.")
  }
  if (!requireNamespace("lmtest", quietly = TRUE)) {
    stop("Package 'lmtest' is required but not installed.")
  }
  
  # Compute robust standard errors
  robust_cov <- sandwich::vcovHC(model, type = type)
  robust_se <- sqrt(diag(robust_cov))
  
  # Create summary table
  coef_table <- data.frame(
    coefficient = names(coef(model)),
    estimate = coef(model),
    robust_se = robust_se,
    t_stat = coef(model) / robust_se,
    p_value = 2 * pt(abs(coef(model) / robust_se), 
                    df = model$df.residual, 
                    lower.tail = FALSE)
  )
  
  rownames(coef_table) <- NULL
  
  return(coef_table)
}


#' Calculate Clustered Standard Errors
#'
#' Computes cluster-robust standard errors for regression models
#'
#' @param model A fitted lm model object
#' @param cluster Vector indicating cluster membership
#' @return A matrix with coefficients and clustered standard errors
#' @export
clustered_se <- function(model, cluster) {
  if (!requireNamespace("sandwich", quietly = TRUE)) {
    stop("Package 'sandwich' is required but not installed.")
  }
  
  # Compute clustered standard errors
  cluster_cov <- sandwich::vcovCL(model, cluster = cluster)
  cluster_se <- sqrt(diag(cluster_cov))
  
  # Create summary table
  coef_table <- data.frame(
    coefficient = names(coef(model)),
    estimate = coef(model),
    clustered_se = cluster_se,
    t_stat = coef(model) / cluster_se,
    p_value = 2 * pt(abs(coef(model) / cluster_se),
                    df = length(unique(cluster)) - 1,
                    lower.tail = FALSE)
  )
  
  rownames(coef_table) <- NULL
  
  return(coef_table)
}


#' Calculate Summary Statistics by Group
#'
#' Computes comprehensive summary statistics for a numeric variable by groups
#'
#' @param data A data frame
#' @param var Name of the numeric variable
#' @param group Name of the grouping variable
#' @return A data frame with summary statistics
#' @export
summary_by_group <- function(data, var, group) {
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Package 'dplyr' is required but not installed.")
  }
  
  library(dplyr)
  
  summary_stats <- data %>%
    group_by(!!sym(group)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(var), na.rm = TRUE),
      median = median(!!sym(var), na.rm = TRUE),
      sd = sd(!!sym(var), na.rm = TRUE),
      min = min(!!sym(var), na.rm = TRUE),
      max = max(!!sym(var), na.rm = TRUE),
      q25 = quantile(!!sym(var), 0.25, na.rm = TRUE),
      q75 = quantile(!!sym(var), 0.75, na.rm = TRUE),
      .groups = 'drop'
    )
  
  return(summary_stats)
}


#' Test for Heteroskedasticity (Breusch-Pagan Test)
#'
#' Performs Breusch-Pagan test for heteroskedasticity
#'
#' @param model A fitted lm model object
#' @return Test results as a list
#' @export
test_heteroskedasticity <- function(model) {
  if (!requireNamespace("lmtest", quietly = TRUE)) {
    stop("Package 'lmtest' is required but not installed.")
  }
  
  bp_test <- lmtest::bptest(model)
  
  result <- list(
    test = "Breusch-Pagan Test for Heteroskedasticity",
    statistic = bp_test$statistic,
    p_value = bp_test$p.value,
    df = bp_test$parameter,
    interpretation = ifelse(
      bp_test$p.value < 0.05,
      "Evidence of heteroskedasticity (p < 0.05). Consider using robust standard errors.",
      "No strong evidence of heteroskedasticity (p >= 0.05)."
    )
  )
  
  return(result)
}


#' Test for Autocorrelation (Durbin-Watson Test)
#'
#' Performs Durbin-Watson test for autocorrelation in residuals
#'
#' @param model A fitted lm model object
#' @return Test results as a list
#' @export
test_autocorrelation <- function(model) {
  if (!requireNamespace("lmtest", quietly = TRUE)) {
    stop("Package 'lmtest' is required but not installed.")
  }
  
  dw_test <- lmtest::dwtest(model)
  
  result <- list(
    test = "Durbin-Watson Test for Autocorrelation",
    statistic = dw_test$statistic,
    p_value = dw_test$p.value,
    interpretation = ifelse(
      dw_test$p.value < 0.05,
      "Evidence of autocorrelation (p < 0.05). Consider using time series models.",
      "No strong evidence of autocorrelation (p >= 0.05)."
    )
  )
  
  return(result)
}


#' Calculate Cohen's d Effect Size
#'
#' Computes Cohen's d effect size for two groups
#'
#' @param group1 Numeric vector for group 1
#' @param group2 Numeric vector for group 2
#' @param pooled Logical, whether to use pooled standard deviation
#' @return Cohen's d value
#' @export
cohens_d <- function(group1, group2, pooled = TRUE) {
  mean_diff <- mean(group1, na.rm = TRUE) - mean(group2, na.rm = TRUE)
  
  if (pooled) {
    n1 <- sum(!is.na(group1))
    n2 <- sum(!is.na(group2))
    sd1 <- sd(group1, na.rm = TRUE)
    sd2 <- sd(group2, na.rm = TRUE)
    
    pooled_sd <- sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))
    d <- mean_diff / pooled_sd
  } else {
    sd1 <- sd(group1, na.rm = TRUE)
    d <- mean_diff / sd1
  }
  
  # Interpretation
  interpretation <- case_when(
    abs(d) < 0.2 ~ "negligible",
    abs(d) < 0.5 ~ "small",
    abs(d) < 0.8 ~ "medium",
    TRUE ~ "large"
  )
  
  result <- list(
    cohens_d = d,
    interpretation = interpretation
  )
  
  return(result)
}


#' Winsorize Data
#'
#' Caps extreme values at specified percentiles
#'
#' @param x Numeric vector
#' @param lower Lower percentile (0-1)
#' @param upper Upper percentile (0-1)
#' @return Winsorized vector
#' @export
winsorize <- function(x, lower = 0.01, upper = 0.99) {
  lower_bound <- quantile(x, lower, na.rm = TRUE)
  upper_bound <- quantile(x, upper, na.rm = TRUE)
  
  x_wins <- ifelse(x < lower_bound, lower_bound,
                   ifelse(x > upper_bound, upper_bound, x))
  
  return(x_wins)
}


#' Create Lag Variables
#'
#' Creates lagged versions of variables for time series or panel data
#'
#' @param data A data frame
#' @param vars Character vector of variable names to lag
#' @param lags Integer vector of lag periods
#' @param id Optional grouping variable for panel data
#' @return Data frame with lagged variables
#' @export
create_lags <- function(data, vars, lags, id = NULL) {
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Package 'dplyr' is required but not installed.")
  }
  
  library(dplyr)
  
  result <- data
  
  for (var in vars) {
    for (lag in lags) {
      lag_name <- paste0(var, "_lag", lag)
      
      if (!is.null(id)) {
        result <- result %>%
          group_by(!!sym(id)) %>%
          mutate(!!lag_name := dplyr::lag(!!sym(var), n = lag)) %>%
          ungroup()
      } else {
        result <- result %>%
          mutate(!!lag_name := dplyr::lag(!!sym(var), n = lag))
      }
    }
  }
  
  return(result)
}


#' Bootstrap Confidence Intervals
#'
#' Computes bootstrap confidence intervals for a statistic
#'
#' @param data Numeric vector or data frame
#' @param stat_fun Function to compute statistic
#' @param R Number of bootstrap replications
#' @param conf_level Confidence level
#' @return List with confidence interval
#' @export
bootstrap_ci <- function(data, stat_fun = mean, R = 1000, conf_level = 0.95) {
  n <- length(data)
  boot_stats <- numeric(R)
  
  for (i in 1:R) {
    boot_sample <- sample(data, size = n, replace = TRUE)
    boot_stats[i] <- stat_fun(boot_sample)
  }
  
  alpha <- 1 - conf_level
  ci <- quantile(boot_stats, c(alpha/2, 1 - alpha/2))
  
  result <- list(
    estimate = stat_fun(data),
    conf_int = ci,
    conf_level = conf_level,
    R = R
  )
  
  return(result)
}


#' Format Regression Table
#'
#' Creates a formatted regression output table
#'
#' @param models List of fitted model objects
#' @param model_names Character vector of model names
#' @return Formatted table
#' @export
format_reg_table <- function(models, model_names = NULL) {
  if (!requireNamespace("broom", quietly = TRUE)) {
    stop("Package 'broom' is required but not installed.")
  }
  
  library(broom)
  library(dplyr)
  
  if (is.null(model_names)) {
    model_names <- paste0("Model ", seq_along(models))
  }
  
  # Extract coefficients from each model
  coef_list <- lapply(seq_along(models), function(i) {
    tidy(models[[i]]) %>%
      mutate(model = model_names[i]) %>%
      select(model, term, estimate, std.error, p.value)
  })
  
  # Combine all coefficients
  all_coefs <- bind_rows(coef_list)
  
  return(all_coefs)
}


# Example usage
if (interactive()) {
  cat("Statistical Helper Functions - Example Usage\n")
  cat("================================================\n\n")
  
  # Generate sample data
  set.seed(123)
  data <- data.frame(
    y = rnorm(100, mean = 10, sd = 2),
    x1 = rnorm(100),
    x2 = rnorm(100),
    group = rep(c("A", "B"), each = 50)
  )
  
  # Fit a model
  model <- lm(y ~ x1 + x2, data = data)
  
  # Get robust standard errors
  cat("Robust Standard Errors:\n")
  print(robust_se(model))
  cat("\n")
  
  # Test for heteroskedasticity
  cat("Heteroskedasticity Test:\n")
  het_test <- test_heteroskedasticity(model)
  print(het_test$interpretation)
  cat("\n")
  
  # Calculate Cohen's d
  group_a <- data$y[data$group == "A"]
  group_b <- data$y[data$group == "B"]
  cat("Cohen's d Effect Size:\n")
  d_result <- cohens_d(group_a, group_b)
  cat(sprintf("d = %.3f (%s effect)\n", d_result$cohens_d, d_result$interpretation))
}
