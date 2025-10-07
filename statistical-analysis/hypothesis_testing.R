# Hypothesis Testing Examples in R
# Author: [Your Name]
# Date: 2024
#
# This script demonstrates various hypothesis testing procedures
# commonly used in statistical analysis and research.

# Load required libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(broom)
})

#' Generate Sample Data for Hypothesis Testing
#'
#' @param n1 Sample size for group 1
#' @param n2 Sample size for group 2
#' @return A data frame with sample data
#' @export
generate_test_data <- function(n1 = 100, n2 = 100) {
  set.seed(456)
  
  # Generate two groups with different means
  group1 <- data.frame(
    group = "Treatment",
    value = rnorm(n1, mean = 25, sd = 5),
    outcome = rnorm(n1, mean = 100, sd = 15)
  )
  
  group2 <- data.frame(
    group = "Control",
    value = rnorm(n2, mean = 22, sd = 5),
    outcome = rnorm(n2, mean = 95, sd = 15)
  )
  
  data <- rbind(group1, group2)
  data$group <- factor(data$group, levels = c("Control", "Treatment"))
  
  return(data)
}


#' Perform One-Sample T-Test
#'
#' Tests whether the mean of a sample differs from a hypothesized value
#'
#' @param data Vector of numeric values
#' @param mu Hypothesized population mean
#' @return List with test results
#' @export
perform_one_sample_ttest <- function(data, mu = 0) {
  test_result <- t.test(data, mu = mu)
  
  result_summary <- list(
    test = "One-Sample t-test",
    statistic = test_result$statistic,
    p_value = test_result$p.value,
    confidence_interval = test_result$conf.int,
    mean_estimate = test_result$estimate,
    null_hypothesis = sprintf("True mean = %.2f", mu),
    decision = ifelse(test_result$p.value < 0.05, 
                     "Reject H0 (p < 0.05)", 
                     "Fail to reject H0 (p >= 0.05)")
  )
  
  return(result_summary)
}


#' Perform Two-Sample T-Test
#'
#' Tests whether means of two independent groups differ
#'
#' @param data Data frame with group and value columns
#' @param equal_var Whether to assume equal variances
#' @return List with test results
#' @export
perform_two_sample_ttest <- function(data, equal_var = FALSE) {
  test_result <- t.test(value ~ group, data = data, var.equal = equal_var)
  
  # Calculate effect size (Cohen's d)
  group_stats <- data %>%
    group_by(group) %>%
    summarise(
      mean = mean(value),
      sd = sd(value),
      n = n(),
      .groups = 'drop'
    )
  
  pooled_sd <- sqrt(sum((group_stats$n - 1) * group_stats$sd^2) / 
                   (sum(group_stats$n) - 2))
  cohens_d <- diff(group_stats$mean) / pooled_sd
  
  result_summary <- list(
    test = ifelse(equal_var, "Two-Sample t-test (equal variance)", 
                 "Welch's t-test (unequal variance)"),
    statistic = test_result$statistic,
    p_value = test_result$p.value,
    confidence_interval = test_result$conf.int,
    mean_diff = diff(test_result$estimate),
    cohens_d = cohens_d,
    decision = ifelse(test_result$p.value < 0.05, 
                     "Reject H0 (p < 0.05)", 
                     "Fail to reject H0 (p >= 0.05)")
  )
  
  return(result_summary)
}


#' Perform Paired T-Test
#'
#' Tests whether means of paired observations differ
#'
#' @param before Vector of measurements before treatment
#' @param after Vector of measurements after treatment
#' @return List with test results
#' @export
perform_paired_ttest <- function(before, after) {
  test_result <- t.test(after, before, paired = TRUE)
  
  result_summary <- list(
    test = "Paired t-test",
    statistic = test_result$statistic,
    p_value = test_result$p.value,
    confidence_interval = test_result$conf.int,
    mean_diff = test_result$estimate,
    decision = ifelse(test_result$p.value < 0.05, 
                     "Reject H0 (p < 0.05)", 
                     "Fail to reject H0 (p >= 0.05)")
  )
  
  return(result_summary)
}


#' Perform One-Way ANOVA
#'
#' Tests whether means of multiple groups differ
#'
#' @param data Data frame with group and value columns
#' @return List with ANOVA results
#' @export
perform_anova <- function(data) {
  # Run ANOVA
  anova_model <- aov(outcome ~ group, data = data)
  anova_summary <- summary(anova_model)
  
  # Calculate eta-squared (effect size)
  ss_total <- sum((data$outcome - mean(data$outcome))^2)
  ss_group <- anova_summary[[1]]["group", "Sum Sq"]
  eta_squared <- ss_group / ss_total
  
  result_summary <- list(
    test = "One-Way ANOVA",
    f_statistic = anova_summary[[1]]["group", "F value"],
    p_value = anova_summary[[1]]["group", "Pr(>F)"],
    df_between = anova_summary[[1]]["group", "Df"],
    df_within = anova_summary[[1]]["Residuals", "Df"],
    eta_squared = eta_squared,
    decision = ifelse(anova_summary[[1]]["group", "Pr(>F)"] < 0.05,
                     "Reject H0 (p < 0.05)", 
                     "Fail to reject H0 (p >= 0.05)")
  )
  
  return(result_summary)
}


#' Perform Chi-Square Test of Independence
#'
#' Tests whether two categorical variables are independent
#'
#' @param data Data frame
#' @param var1 Name of first categorical variable
#' @param var2 Name of second categorical variable
#' @return List with test results
#' @export
perform_chisquare_test <- function(data, var1, var2) {
  # Create contingency table
  cont_table <- table(data[[var1]], data[[var2]])
  
  # Perform chi-square test
  test_result <- chisq.test(cont_table)
  
  # Calculate Cramer's V (effect size)
  n <- sum(cont_table)
  min_dim <- min(dim(cont_table)) - 1
  cramers_v <- sqrt(test_result$statistic / (n * min_dim))
  
  result_summary <- list(
    test = "Chi-Square Test of Independence",
    statistic = test_result$statistic,
    p_value = test_result$p.value,
    df = test_result$parameter,
    cramers_v = cramers_v,
    contingency_table = cont_table,
    decision = ifelse(test_result$p.value < 0.05,
                     "Reject H0 - Variables are associated (p < 0.05)",
                     "Fail to reject H0 - No evidence of association (p >= 0.05)")
  )
  
  return(result_summary)
}


#' Perform Wilcoxon Rank-Sum Test (Mann-Whitney U)
#'
#' Non-parametric test for comparing two independent groups
#'
#' @param data Data frame with group and value columns
#' @return List with test results
#' @export
perform_wilcoxon_test <- function(data) {
  test_result <- wilcox.test(value ~ group, data = data)
  
  result_summary <- list(
    test = "Wilcoxon Rank-Sum Test (Mann-Whitney U)",
    statistic = test_result$statistic,
    p_value = test_result$p.value,
    decision = ifelse(test_result$p.value < 0.05,
                     "Reject H0 (p < 0.05)",
                     "Fail to reject H0 (p >= 0.05)")
  )
  
  return(result_summary)
}


#' Create Visualization for Group Comparisons
#'
#' @param data Data frame with group and value columns
#' @return ggplot object
#' @export
plot_group_comparison <- function(data) {
  p <- ggplot(data, aes(x = group, y = value, fill = group)) +
    geom_violin(alpha = 0.6, trim = FALSE) +
    geom_boxplot(width = 0.2, alpha = 0.8, outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.3, size = 1) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 4,
                fill = "red", color = "darkred") +
    labs(
      title = "Comparison of Groups",
      subtitle = "Red diamonds indicate mean values",
      x = "Group",
      y = "Value"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      legend.position = "none"
    )
  
  return(p)
}


#' Main Analysis Function
#'
#' Demonstrates various hypothesis testing procedures
#' @export
main <- function() {
  cat("======================================================\n")
  cat("Hypothesis Testing Examples\n")
  cat("======================================================\n\n")
  
  # Generate sample data
  cat("Generating sample data...\n")
  data <- generate_test_data(n1 = 100, n2 = 100)
  cat(sprintf("Data generated: %d observations\n\n", nrow(data)))
  
  # One-sample t-test
  cat("1. One-Sample t-test\n")
  cat("------------------------------------------------------\n")
  cat("Testing if mean differs from 23\n")
  one_sample_result <- perform_one_sample_ttest(data$value, mu = 23)
  cat(sprintf("Test statistic: t = %.3f\n", one_sample_result$statistic))
  cat(sprintf("P-value: %.4f\n", one_sample_result$p_value))
  cat(sprintf("Decision: %s\n\n", one_sample_result$decision))
  
  # Two-sample t-test
  cat("2. Two-Sample t-test (Welch's)\n")
  cat("------------------------------------------------------\n")
  two_sample_result <- perform_two_sample_ttest(data, equal_var = FALSE)
  cat(sprintf("Test statistic: t = %.3f\n", two_sample_result$statistic))
  cat(sprintf("P-value: %.4f\n", two_sample_result$p_value))
  cat(sprintf("Mean difference: %.3f\n", two_sample_result$mean_diff))
  cat(sprintf("Cohen's d: %.3f\n", two_sample_result$cohens_d))
  cat(sprintf("Decision: %s\n\n", two_sample_result$decision))
  
  # ANOVA
  cat("3. One-Way ANOVA\n")
  cat("------------------------------------------------------\n")
  anova_result <- perform_anova(data)
  cat(sprintf("F-statistic: F(%d, %d) = %.3f\n", 
             anova_result$df_between, 
             anova_result$df_within,
             anova_result$f_statistic))
  cat(sprintf("P-value: %.4f\n", anova_result$p_value))
  cat(sprintf("Eta-squared: %.3f\n", anova_result$eta_squared))
  cat(sprintf("Decision: %s\n\n", anova_result$decision))
  
  # Non-parametric test
  cat("4. Wilcoxon Rank-Sum Test (Non-parametric)\n")
  cat("------------------------------------------------------\n")
  wilcox_result <- perform_wilcoxon_test(data)
  cat(sprintf("Test statistic: W = %.0f\n", wilcox_result$statistic))
  cat(sprintf("P-value: %.4f\n", wilcox_result$p_value))
  cat(sprintf("Decision: %s\n\n", wilcox_result$decision))
  
  # Create visualization
  cat("Creating visualization...\n")
  p <- plot_group_comparison(data)
  # ggsave("group_comparison.png", p, width = 8, height = 6)
  
  cat("\nAnalysis complete!\n")
  cat("======================================================\n")
  
  return(list(
    data = data,
    tests = list(
      one_sample = one_sample_result,
      two_sample = two_sample_result,
      anova = anova_result,
      wilcoxon = wilcox_result
    ),
    plot = p
  ))
}


# Run analysis if script is executed directly
if (sys.nframe() == 0) {
  results <- main()
}
