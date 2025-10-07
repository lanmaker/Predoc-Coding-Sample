# R Package Dependencies for Predoc Coding Sample
# Install with: Rscript install_packages.R

# List of required packages
required_packages <- c(
  # Data manipulation
  "dplyr",
  "tidyr",
  "readr",
  "data.table",
  
  # Visualization
  "ggplot2",
  
  # Statistical analysis
  "broom",
  "lmtest",
  "sandwich",
  "plm",
  
  # Additional utilities
  "scales",
  "stargazer"
)

# Function to install missing packages
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  
  if (length(new_packages) > 0) {
    cat("Installing missing packages:", paste(new_packages, collapse = ", "), "\n")
    install.packages(new_packages, repos = "https://cloud.r-project.org/")
  } else {
    cat("All required packages are already installed.\n")
  }
}

# Install packages
install_if_missing(required_packages)

cat("\nPackage installation complete!\n")
cat("You can now run the R scripts in this repository.\n")
