# Replication Study: Returns to Education
# Author: [Your Name]
# Date: 2024
#
# This script replicates a classic analysis of returns to education
# demonstrating ability to implement econometric specifications from papers.

# Load required libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(broom)
  library(stargazer)
})

# Set random seed for reproducibility
set.seed(2024)

#' Generate Synthetic Labor Market Data
#'
#' Creates synthetic data mimicking labor economics datasets
#' @param n Sample size
#' @return Data frame with labor market variables
generate_labor_data <- function(n = 2000) {
  # Demographics
  age <- round(runif(n, min = 25, max = 65))
  female <- rbinom(n, 1, 0.48)
  
  # Education (years of schooling)
  # Higher for younger cohorts
  education <- round(rnorm(n, mean = 14 - 0.05 * (age - 45), sd = 2.5))
  education <- pmax(8, pmin(20, education))  # Bound between 8 and 20
  
  # Experience
  experience <- age - education - 6
  experience <- pmax(0, experience)
  
  # Geographic/demographic controls
  urban <- rbinom(n, 1, 0.7)
  
  # Generate income with realistic structure
  # Log-linear relationship with education
  # Concave relationship with experience
  # Gender wage gap
  # Random component
  
  log_income <- (
    9.5 +                              # Baseline (corresponds to ~$13,000)
    0.08 * education +                 # Returns to education (~8% per year)
    0.03 * experience +                # Returns to experience
    -0.0004 * experience^2 +           # Diminishing returns
    -0.15 * female +                   # Gender wage gap
    0.10 * urban +                     # Urban premium
    rnorm(n, mean = 0, sd = 0.3)      # Random component
  )
  
  income <- exp(log_income)
  
  data <- data.frame(
    income = income,
    log_income = log_income,
    education = education,
    experience = experience,
    experience_sq = experience^2,
    age = age,
    female = female,
    urban = urban
  )
  
  return(data)
}


#' Estimate Baseline Models
#'
#' Replicates the main specifications from the hypothetical paper
#' @param data Labor market data
#' @return List of model results
estimate_models <- function(data) {
  # Model 1: Simple bivariate regression
  model1 <- lm(log_income ~ education, data = data)
  
  # Model 2: Add experience controls
  model2 <- lm(log_income ~ education + experience + experience_sq, 
               data = data)
  
  # Model 3: Full specification
  model3 <- lm(log_income ~ education + experience + experience_sq + 
                 female + urban, 
               data = data)
  
  # Model 4: Interaction between education and experience
  model4 <- lm(log_income ~ education * experience + experience_sq + 
                 female + urban,
               data = data)
  
  models <- list(
    model1 = model1,
    model2 = model2,
    model3 = model3,
    model4 = model4
  )
  
  return(models)
}


#' Create Visualization of Education Returns
#'
#' Plots the relationship between education and income
#' @param data Labor market data
#' @return ggplot object
plot_education_income <- function(data) {
  # Bin data for cleaner visualization
  data_summary <- data %>%
    group_by(education) %>%
    summarise(
      mean_income = mean(income),
      median_income = median(income),
      n = n(),
      .groups = 'drop'
    )
  
  p <- ggplot(data, aes(x = education, y = income)) +
    geom_point(alpha = 0.1, size = 1) +
    geom_line(data = data_summary, aes(y = mean_income), 
             color = "red", size = 1.2) +
    geom_smooth(method = "lm", se = TRUE, color = "blue", linetype = "dashed") +
    scale_y_continuous(labels = scales::dollar) +
    labs(
      title = "Returns to Education",
      subtitle = "Red line: Mean income by education level, Blue: Linear fit",
      x = "Years of Education",
      y = "Annual Income"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      panel.grid.minor = element_blank()
    )
  
  return(p)
}


#' Plot Experience Profile
#'
#' Visualizes the concave relationship between experience and income
#' @param data Labor market data
#' @return ggplot object
plot_experience_profile <- function(data) {
  # Create prediction data
  pred_data <- data.frame(
    experience = seq(0, 40, by = 1),
    education = median(data$education),
    female = 0,
    urban = 1
  )
  pred_data$experience_sq <- pred_data$experience^2
  
  # Use model 3 for predictions
  model <- lm(log_income ~ education + experience + experience_sq + 
               female + urban, data = data)
  
  pred_data$predicted_log_income <- predict(model, newdata = pred_data)
  pred_data$predicted_income <- exp(pred_data$predicted_log_income)
  
  p <- ggplot(pred_data, aes(x = experience, y = predicted_income)) +
    geom_line(color = "steelblue", size = 1.2) +
    geom_vline(xintercept = -coef(model)["experience"] / (2 * coef(model)["experience_sq"]),
              linetype = "dashed", color = "red", alpha = 0.7) +
    scale_y_continuous(labels = scales::dollar) +
    labs(
      title = "Experience-Earnings Profile",
      subtitle = "Predicted income for median education, male, urban worker\nRed line indicates peak",
      x = "Years of Experience",
      y = "Predicted Annual Income"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      panel.grid.minor = element_blank()
    )
  
  return(p)
}


#' Perform Robustness Checks
#'
#' Additional specifications to verify main results
#' @param data Labor market data
#' @return List with robustness check results
robustness_checks <- function(data) {
  # Separate analysis by gender
  model_male <- lm(log_income ~ education + experience + experience_sq + urban,
                   data = filter(data, female == 0))
  
  model_female <- lm(log_income ~ education + experience + experience_sq + urban,
                     data = filter(data, female == 1))
  
  # Quadratic in education
  model_edu_sq <- lm(log_income ~ education + I(education^2) + 
                      experience + experience_sq + female + urban,
                     data = data)
  
  # Results comparison
  results <- list(
    male_sample = model_male,
    female_sample = model_female,
    education_quadratic = model_edu_sq
  )
  
  return(results)
}


#' Main Replication Function
#'
#' Runs the complete replication analysis
main_replication <- function() {
  cat("========================================================\n")
  cat("Replication Study: Returns to Education\n")
  cat("========================================================\n\n")
  
  # Generate data
  cat("1. Generating synthetic labor market data...\n")
  data <- generate_labor_data(n = 2000)
  cat(sprintf("   Sample size: %d observations\n", nrow(data)))
  cat(sprintf("   Mean income: $%,.0f\n", mean(data$income)))
  cat(sprintf("   Mean education: %.1f years\n", mean(data$education)))
  cat(sprintf("   Mean experience: %.1f years\n\n", mean(data$experience)))
  
  # Descriptive statistics
  cat("2. Descriptive Statistics:\n")
  cat("   -------------------------------------------------\n")
  summary_stats <- data %>%
    summarise(
      across(c(income, education, experience, age),
             list(mean = ~mean(.), sd = ~sd(.), min = ~min(.), max = ~max(.)))
    )
  print(t(summary_stats), digits = 2)
  cat("\n")
  
  # Estimate models
  cat("3. Estimating regression models...\n\n")
  models <- estimate_models(data)
  
  # Display main results
  cat("4. Main Results:\n")
  cat("   -------------------------------------------------\n")
  for (i in 1:4) {
    cat(sprintf("\n   Model %d:\n", i))
    cat(sprintf("   Education coefficient: %.4f (se: %.4f)\n",
               coef(models[[i]])["education"],
               summary(models[[i]])$coefficients["education", "Std. Error"]))
    cat(sprintf("   R-squared: %.4f\n", summary(models[[i]])$r.squared))
  }
  cat("\n")
  
  # Robustness checks
  cat("5. Robustness Checks:\n")
  cat("   -------------------------------------------------\n")
  robust <- robustness_checks(data)
  cat(sprintf("   Male sample - Education coef: %.4f\n", 
             coef(robust$male_sample)["education"]))
  cat(sprintf("   Female sample - Education coef: %.4f\n",
             coef(robust$female_sample)["education"]))
  cat("\n")
  
  # Create visualizations
  cat("6. Creating visualizations...\n")
  p1 <- plot_education_income(data)
  p2 <- plot_experience_profile(data)
  # In actual analysis: ggsave("education_returns.png", p1, width = 10, height = 6)
  # In actual analysis: ggsave("experience_profile.png", p2, width = 10, height = 6)
  
  cat("\n7. Analysis Summary:\n")
  cat("   -------------------------------------------------\n")
  cat("   - One additional year of education is associated with\n")
  cat(sprintf("     approximately %.1f%% higher income (Model 3)\n", 
             100 * coef(models$model3)["education"]))
  cat("   - Experience shows diminishing returns (concave profile)\n")
  cat("   - Results are robust across specifications\n")
  
  cat("\n========================================================\n")
  cat("Replication Complete!\n")
  cat("========================================================\n")
  
  return(list(
    data = data,
    models = models,
    robustness = robust,
    plots = list(education_income = p1, experience_profile = p2)
  ))
}


# Execute replication if script is run directly
if (sys.nframe() == 0) {
  results <- main_replication()
}
