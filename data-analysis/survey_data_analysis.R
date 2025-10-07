# Survey Data Analysis in R
# Author: [Your Name]
# Date: 2024
#
# This script demonstrates data analysis skills using R tidyverse.
# It analyzes survey data with focus on data wrangling and visualization.

# Load required libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
})

#' Load and Clean Survey Data
#'
#' @param filepath Character string with path to survey data file
#' @return A cleaned tibble with survey responses
#' @export
load_survey_data <- function(filepath = NULL) {
  # For demonstration, create sample survey data
  set.seed(123)
  n <- 500
  
  data <- tibble(
    respondent_id = 1:n,
    age = sample(18:75, n, replace = TRUE),
    gender = sample(c("Male", "Female", "Other", NA), n, replace = TRUE, 
                   prob = c(0.48, 0.48, 0.03, 0.01)),
    education = sample(c("High School", "Bachelor", "Master", "PhD", NA), 
                      n, replace = TRUE, 
                      prob = c(0.30, 0.40, 0.20, 0.08, 0.02)),
    income_category = sample(c("Low", "Medium", "High", NA), 
                            n, replace = TRUE,
                            prob = c(0.25, 0.50, 0.23, 0.02)),
    satisfaction = sample(1:10, n, replace = TRUE),
    hours_worked = round(rnorm(n, mean = 40, sd = 10), 1)
  )
  
  # Clip hours worked to realistic range
  data$hours_worked <- pmax(0, pmin(80, data$hours_worked))
  
  return(data)
}


#' Clean Survey Data
#'
#' @param data Raw survey data tibble
#' @return Cleaned survey data
#' @export
clean_survey_data <- function(data) {
  data_clean <- data %>%
    # Remove rows with too many missing values
    filter(rowSums(is.na(.)) <= 2) %>%
    # Convert categorical variables to factors with appropriate levels
    mutate(
      gender = factor(gender, levels = c("Male", "Female", "Other")),
      education = factor(education, 
                        levels = c("High School", "Bachelor", "Master", "PhD"),
                        ordered = TRUE),
      income_category = factor(income_category, 
                              levels = c("Low", "Medium", "High"),
                              ordered = TRUE)
    ) %>%
    # Create age groups
    mutate(
      age_group = cut(age, 
                     breaks = c(0, 25, 35, 50, 65, Inf),
                     labels = c("18-25", "26-35", "36-50", "51-65", "65+"),
                     right = FALSE)
    )
  
  return(data_clean)
}


#' Calculate Summary Statistics by Group
#'
#' @param data Cleaned survey data
#' @return Summary statistics tibble
#' @export
calculate_group_statistics <- function(data) {
  summary_stats <- data %>%
    group_by(education, income_category) %>%
    summarise(
      n = n(),
      mean_satisfaction = mean(satisfaction, na.rm = TRUE),
      sd_satisfaction = sd(satisfaction, na.rm = TRUE),
      mean_hours = mean(hours_worked, na.rm = TRUE),
      median_age = median(age, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    arrange(education, income_category)
  
  return(summary_stats)
}


#' Create Satisfaction Distribution Plot
#'
#' @param data Survey data
#' @return ggplot object
#' @export
plot_satisfaction_distribution <- function(data) {
  p <- ggplot(data, aes(x = satisfaction)) +
    geom_histogram(binwidth = 1, fill = "steelblue", color = "white", alpha = 0.8) +
    geom_vline(aes(xintercept = mean(satisfaction, na.rm = TRUE)),
               color = "red", linetype = "dashed", size = 1) +
    labs(
      title = "Distribution of Satisfaction Scores",
      subtitle = "Red dashed line indicates mean satisfaction",
      x = "Satisfaction Score (1-10)",
      y = "Count"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10)
    )
  
  return(p)
}


#' Create Satisfaction by Education Plot
#'
#' @param data Survey data
#' @return ggplot object
#' @export
plot_satisfaction_by_education <- function(data) {
  # Remove NA values for cleaner visualization
  data_clean <- data %>% filter(!is.na(education))
  
  p <- ggplot(data_clean, aes(x = education, y = satisfaction, fill = education)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.2, size = 1) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 3, 
                fill = "red", color = "darkred") +
    labs(
      title = "Satisfaction Scores by Education Level",
      subtitle = "Red diamonds indicate mean values",
      x = "Education Level",
      y = "Satisfaction Score"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      legend.position = "none",
      axis.text.x = element_text(angle = 15, hjust = 1)
    )
  
  return(p)
}


#' Create Hours Worked vs Satisfaction Scatter Plot
#'
#' @param data Survey data
#' @return ggplot object
#' @export
plot_hours_vs_satisfaction <- function(data) {
  p <- ggplot(data, aes(x = hours_worked, y = satisfaction, color = income_category)) +
    geom_point(alpha = 0.6, size = 2) +
    geom_smooth(method = "lm", se = TRUE, alpha = 0.2) +
    facet_wrap(~ income_category) +
    labs(
      title = "Hours Worked vs Satisfaction by Income Category",
      x = "Hours Worked per Week",
      y = "Satisfaction Score",
      color = "Income Category"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom"
    )
  
  return(p)
}


#' Perform Statistical Tests
#'
#' @param data Survey data
#' @return List with test results
#' @export
perform_statistical_tests <- function(data) {
  # ANOVA: Satisfaction by education level
  anova_result <- aov(satisfaction ~ education, data = data)
  
  # Correlation: Hours worked vs satisfaction
  cor_test <- cor.test(data$hours_worked, data$satisfaction, method = "pearson")
  
  # T-test: Gender differences in satisfaction
  data_gender <- data %>% filter(gender %in% c("Male", "Female"))
  ttest_result <- t.test(satisfaction ~ gender, data = data_gender)
  
  results <- list(
    anova = summary(anova_result),
    correlation = cor_test,
    t_test = ttest_result
  )
  
  return(results)
}


#' Main Analysis Function
#'
#' Runs the complete survey data analysis pipeline
#' @export
main <- function() {
  cat("======================================================\n")
  cat("Survey Data Analysis\n")
  cat("======================================================\n\n")
  
  # Load data
  cat("Loading survey data...\n")
  raw_data <- load_survey_data()
  cat(sprintf("Loaded %d survey responses\n\n", nrow(raw_data)))
  
  # Clean data
  cat("Cleaning data...\n")
  clean_data <- clean_survey_data(raw_data)
  cat(sprintf("After cleaning: %d observations\n\n", nrow(clean_data)))
  
  # Summary statistics
  cat("Calculating summary statistics...\n")
  summary_stats <- calculate_group_statistics(clean_data)
  print(summary_stats)
  cat("\n")
  
  # Statistical tests
  cat("Performing statistical tests...\n")
  test_results <- perform_statistical_tests(clean_data)
  
  cat("\nANOVA: Satisfaction by Education Level\n")
  cat("------------------------------------------------------\n")
  print(test_results$anova)
  
  cat("\nCorrelation: Hours Worked vs Satisfaction\n")
  cat("------------------------------------------------------\n")
  cat(sprintf("Pearson's r = %.3f, p-value = %.4f\n", 
             test_results$correlation$estimate,
             test_results$correlation$p.value))
  
  # Create visualizations
  cat("\nCreating visualizations...\n")
  p1 <- plot_satisfaction_distribution(clean_data)
  p2 <- plot_satisfaction_by_education(clean_data)
  p3 <- plot_hours_vs_satisfaction(clean_data)
  
  # In a real analysis, would save plots:
  # ggsave("satisfaction_distribution.png", p1, width = 10, height = 6)
  # ggsave("satisfaction_by_education.png", p2, width = 10, height = 6)
  # ggsave("hours_vs_satisfaction.png", p3, width = 12, height = 6)
  
  cat("\nAnalysis complete!\n")
  cat("======================================================\n")
  
  return(list(
    data = clean_data,
    summary = summary_stats,
    tests = test_results,
    plots = list(p1 = p1, p2 = p2, p3 = p3)
  ))
}


# Run analysis if script is executed directly
if (sys.nframe() == 0) {
  results <- main()
}
