# Crazy Distribution Simulation and Central Limit Theorem Demonstration
# Author: Analysis based on NSDUH dataset structure
# Purpose: Demonstrate CLT with a highly non-normal, irregular distribution

# Load required libraries
library(ggplot2)
library(dplyr)
library(gridExtra)

# Set working directory
#setwd("~/Desktop/Replace with your own path")

# Set seed for reproducibility
set.seed(42)

# ============================================================================
# CREATE CRAZY POPULATION DISTRIBUTION (N = 10,000)
# ============================================================================

# Create a "totally crazy" distribution with multiple components:
# Mixture of different distributions
# Gaps and discontinuities
# Extreme asymmetry
# Multiple modes

create_crazy_distribution <- function(n = 10000) {
  # Generate more than needed to account for filtering
  n_generate <- round(n * 1.5)  # Generate 50% more

  # Component 1: Exponential cluster (30% of data)
  comp1 <- rexp(round(n_generate * 0.3), rate = 2) + 1

  # Component 2: Uniform in middle range with gaps (20% of data)
  comp2 <- runif(round(n_generate * 0.2), min = 8, max = 12)

  # Component 3: Another exponential cluster, flipped (25% of data)
  comp3 <- 25 - rexp(round(n_generate * 0.25), rate = 1.5)

  # Component 4: Extreme outliers (5% of data)
  comp4 <- runif(round(n_generate * 0.05), min = 50, max = 100)

  # Component 5: Normal-ish cluster (20% of data)
  comp5 <- rnorm(round(n_generate * 0.2), mean = 15, sd = 2)

  # Combine all components
  population <- c(comp1, comp2, comp3, comp4, comp5)

  # Remove some values to create gaps (remove values between 5-7 and 30-45)
  population <- population[!(population > 5 & population < 7)]
  population <- population[!(population > 30 & population < 45)]

  # If we still don't have enough, sample with replacement from what we have
  if(length(population) < n) {
    additional_needed <- n - length(population)
    population <- c(population, sample(population, additional_needed, replace = TRUE))
  }

  # Return exactly n observations
  return(population[1:n])
}

# Generate the crazy population
population <- create_crazy_distribution(10000)

# Calculate population parameters
pop_mean <- mean(population)
pop_sd <- sd(population)
pop_var <- var(population)

cat("Population Statistics:\n")
cat(sprintf("Mean: %.3f\n", pop_mean))
cat(sprintf("Standard Deviation: %.3f\n", pop_sd))
cat(sprintf("Variance: %.3f\n", pop_var))
cat(sprintf("Min: %.3f, Max: %.3f\n", min(population), max(population)))

# Create density plot of the population
p_population <- ggplot(data.frame(x = population), aes(x = x)) +
  geom_density(fill = "lightblue", alpha = 0.7, color = "darkblue") +
  labs(title = "Population Distribution (N = 10,000)",
       subtitle = "Crazy Distribution: Multi-modal, Asymmetric, with Gaps",
       x = "Value", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Save population distribution plot
pdf("population_distribution.pdf", width = 10, height = 7.5)
print(p_population)
dev.off()

# ============================================================================
# PART (a): Sample 200 observations
# ============================================================================

sample_200 <- sample(population, size = 200, replace = FALSE)
cat(sprintf("\nPart (a) - Sample of 200 observations:\n"))
cat(sprintf("Sample mean: %.3f\n", mean(sample_200)))
cat(sprintf("Sample SD: %.3f\n", sd(sample_200)))

# ============================================================================
# PART (b): 500 different samples of size 200, calculate sample means
# ============================================================================

n_samples <- 500
sample_size_large <- 200
sample_means_large <- numeric(n_samples)

set.seed(42)  # Reset seed for reproducibility
for(i in 1:n_samples) {
  sample_means_large[i] <- mean(sample(population, size = sample_size_large, replace = FALSE))
}

# Create density plot for sample means (n=200)
p_means_200 <- ggplot(data.frame(x = sample_means_large), aes(x = x)) +
  geom_density(fill = "lightgreen", alpha = 0.7, color = "darkgreen") +
  labs(title = "Distribution of Sample Means (n = 200, 500 samples)",
       x = "Sample Mean", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# ============================================================================
# PART (c): Theoretical sampling distribution
# ============================================================================

# Theoretical sampling distribution parameters
theoretical_mean <- pop_mean
theoretical_se <- pop_sd / sqrt(sample_size_large)

cat(sprintf("\nTheoretical Sampling Distribution (n=200):\n"))
cat(sprintf("Mean: %.3f\n", theoretical_mean))
cat(sprintf("Standard Error: %.3f\n", theoretical_se))

# Add theoretical distribution to the plot
x_range <- seq(min(sample_means_large) - 1, max(sample_means_large) + 1, length.out = 1000)
theoretical_density <- dnorm(x_range, mean = theoretical_mean, sd = theoretical_se)

p_means_200_with_theory <- p_means_200 +
  geom_line(data = data.frame(x = x_range, y = theoretical_density),
            aes(x = x, y = y), color = "red", linewidth = 1.2, linetype = "dashed") +
  labs(title = "Sample Means vs Theoretical Distribution (n = 200)",
       subtitle = "Green = Empirical, Red Dashed = Theoretical Normal") +
  theme(plot.subtitle = element_text(hjust = 0.5))

# Save sample means distribution plot (n=200)
pdf("sample_means_n200_with_theory.pdf", width = 10, height = 7.5)
print(p_means_200_with_theory)
dev.off()

# ============================================================================
# PART (d): Repeat with sample size = 20
# ============================================================================

sample_size_small <- 20
sample_means_small <- numeric(n_samples)

set.seed(42)  # Reset seed for reproducibility
for(i in 1:n_samples) {
  sample_means_small[i] <- mean(sample(population, size = sample_size_small, replace = FALSE))
}

# Theoretical parameters for n=20
theoretical_se_small <- pop_sd / sqrt(sample_size_small)

cat(sprintf("\nTheoretical Sampling Distribution (n=20):\n"))
cat(sprintf("Mean: %.3f\n", theoretical_mean))
cat(sprintf("Standard Error: %.3f\n", theoretical_se_small))

# Create comparison plot
x_range_small <- seq(min(sample_means_small) - 2, max(sample_means_small) + 2, length.out = 1000)
theoretical_density_small <- dnorm(x_range_small, mean = theoretical_mean, sd = theoretical_se_small)

p_means_20 <- ggplot(data.frame(x = sample_means_small), aes(x = x)) +
  geom_density(fill = "lightcoral", alpha = 0.7, color = "darkred") +
  geom_line(data = data.frame(x = x_range_small, y = theoretical_density_small),
            aes(x = x, y = y), color = "blue", linewidth = 1.2, linetype = "dashed") +
  labs(title = "Sample Means vs Theoretical Distribution (n = 20)",
       subtitle = "Coral = Empirical, Blue Dashed = Theoretical Normal",
       x = "Sample Mean", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Save sample means distribution plot (n=20)
pdf("sample_means_n20_with_theory.pdf", width = 10, height = 7.5)
print(p_means_20)
dev.off()

# ============================================================================
# COMPARISON PLOTS
# ============================================================================

# Side-by-side comparison
comparison_data <- data.frame(
  value = c(sample_means_large, sample_means_small),
  sample_size = rep(c("n = 200", "n = 20"), each = n_samples)
)

p_comparison <- ggplot(comparison_data, aes(x = value, fill = sample_size)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~sample_size, scales = "free") +
  labs(title = "Comparison: Distribution of Sample Means",
       x = "Sample Mean", y = "Density",
       fill = "Sample Size") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Save comparison plot
pdf("sample_means_comparison.pdf", width = 10, height = 7.5)
print(p_comparison)
dev.off()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("SUMMARY RESULTS\n")
cat(paste(rep("=", 60), collapse=""), "\n")

cat(sprintf("Population mean: %.3f\n", pop_mean))
cat(sprintf("Population SD: %.3f\n\n", pop_sd))

cat("Sample Means Distribution (n=200):\n")
cat(sprintf("  Empirical mean: %.3f\n", mean(sample_means_large)))
cat(sprintf("  Empirical SD: %.3f\n", sd(sample_means_large)))
cat(sprintf("  Theoretical SE: %.3f\n\n", theoretical_se))

cat("Sample Means Distribution (n=20):\n")
cat(sprintf("  Empirical mean: %.3f\n", mean(sample_means_small)))
cat(sprintf("  Empirical SD: %.3f\n", sd(sample_means_small)))
cat(sprintf("  Theoretical SE: %.3f\n\n", theoretical_se_small))

cat("Key Observations:\n")
cat(sprintf("1. Variance reduction with larger n: %.3f vs %.3f\n",
            var(sample_means_small), var(sample_means_large)))
cat(sprintf("2. SE ratio (theory): %.3f\n", theoretical_se_small / theoretical_se))
cat(sprintf("3. SE ratio (empirical): %.3f\n", sd(sample_means_small) / sd(sample_means_large)))

# ============================================================================
# PART (e): Discussion about normal population
# ============================================================================

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("PART (E): WHAT IF POPULATION WAS NORMAL?\n")
cat(paste(rep("=", 60), collapse=""), "\n")

# Simulate normal population with same mean and SD
normal_population <- rnorm(10000, mean = pop_mean, sd = pop_sd)

# Sample from normal population
set.seed(42)
normal_sample_means_200 <- numeric(n_samples)
normal_sample_means_20 <- numeric(n_samples)

for(i in 1:n_samples) {
  normal_sample_means_200[i] <- mean(sample(normal_population, size = 200))
  normal_sample_means_20[i] <- mean(sample(normal_population, size = 20))
}

# Create comparison plot
normal_comparison <- data.frame(
  value = c(sample_means_large, normal_sample_means_200,
            sample_means_small, normal_sample_means_20),
  distribution = rep(c("Crazy Pop", "Normal Pop"), times = 2, each = n_samples),
  sample_size = rep(c("n = 200", "n = 20"), each = n_samples * 2)
)

p_normal_comparison <- ggplot(normal_comparison, aes(x = value, fill = distribution)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~sample_size) +
  labs(title = "Crazy vs Normal Population: Sample Means Distribution",
       x = "Sample Mean", y = "Density",
       fill = "Population Type") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Save normal vs crazy population comparison plot
pdf("normal_vs_crazy_comparison.pdf", width = 10, height = 7.5)
print(p_normal_comparison)
dev.off()
