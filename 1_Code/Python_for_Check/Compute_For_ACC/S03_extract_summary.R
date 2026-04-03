# ==============================================================================
# S3_extract_summary.R
# ==============================================================================
# Purpose: Extract Cohen's d summary from empirical data (Study 3)
# Input: Study3/data/emp_data/, Study3/data/fast_dm/
# Output: summary_emp_S3_cohens_d.csv
# ==============================================================================

# ==============================================================================
# 1. SETUP
# ==============================================================================

# Load packages
if (!requireNamespace('pacman', quietly = TRUE)) {
    install.packages('pacman')
}

pacman::p_load(
  here, tidyverse, purrr, bruceR, boot
)

# Disable scientific notation
options(scipen = 999)

# Source utility functions
source(here::here("analysis", "utils.R"))

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================

# Load empirical data
cat("Loading empirical data...\n")
df_emp <- load_emp_data(here::here("Study3", "data", "emp_data"))
cat("Loaded:", nrow(df_emp), "rows\n")

# ==============================================================================
# 3. DATA PREPROCESSING
# ==============================================================================

# Add MatchKey and Matchness variables
df_emp <- df_emp %>%
  dplyr::filter(stage == "formal") %>%
  dplyr::mutate(
    MatchKey = dplyr::case_when(
      subjectID %% 4 == 1 ~ "f",
      subjectID %% 4 == 2 ~ "j",
      subjectID %% 4 == 3 ~ "j",
      subjectID %% 4 == 0 ~ "f",
      TRUE ~ NA_character_
    ),
    Matchness = ifelse(CorrectKey == MatchKey, "match", "mismatch"),
    T_val = T_val * 1000,  # Convert to milliseconds
    W = W * 1000,          # Convert to milliseconds
    RT_ms = RT * 1000      # Convert to milliseconds
  )

# Generate Press variable
df_emp$Press <- ifelse(!is.nan(df_emp$RT), 1, 0)

# Add block variable
df_emp <- df_emp[order(df_emp$groupID, df_emp$subjectID, df_emp$trialID), ]
df_emp <- df_emp %>%
  dplyr::group_by(groupID, subjectID) %>%
  dplyr::mutate(block = rep(1:ceiling(n()/52), each = 52, length.out = n())) %>%
  dplyr::ungroup()

# Filter for match trials only
df_emp_match <- df_emp %>% 
  dplyr::filter(Matchness == "match") %>%
  dplyr::mutate(Type = "empirical") %>%
  dplyr::select(groupID, subjectID, Type, block, trialID, P, T_val, W, Label, Press, Correct, RT, RT_ms)

cat("Match trials:", nrow(df_emp_match), "\n")

# ==============================================================================
# 4. DEFINE DESIGNS (Study 3: D1-D6)
# ==============================================================================

# Map groupID to designs for Study 3
# Based on the experimental setup:
# New design naming: D3->D3a, D4->D4a
# groupID 1 = D1 (P=0, T=30, W=300)
# groupID 2 = D2 (P=0, T=30, W=600)
# groupID 3 = D3a (P=120, T=30, W=600)
# groupID 4 = D4a (P=120, T=80, W=600)
# groupID 5 = D5 (P=8, T=100, W=1100)
# groupID 6 = D6 (P=120, T=500, W=1500)

designs_S3 <- data.frame(
  groupID = 1:6,
  Design = c("D1", "D2", "D3a", "D4a", "D5", "D6"),
  P = c(0, 0, 120, 120, 8, 120),
  T = c(30, 30, 30, 80, 100, 500),
  W = c(300, 600, 600, 600, 1100, 1500)
)

cat("\nDesigns:\n")
print(designs_S3)

# ==============================================================================
# 5. CALCULATE COHEN'S D BY DESIGN
# ==============================================================================

# Bootstrap function for Cohen's d (RT)
bootstrap_cohens_d_rt <- function(data, indices) {
  d <- data[indices, ]
  # Separate self and stranger
  self_data <- d$RT_ms[d$Label == "self"]
  stranger_data <- d$RT_ms[d$Label == "stranger"]
  
  if (length(self_data) < 2 | length(stranger_data) < 2) {
    return(NA)
  }
  
  mean_self <- mean(self_data, na.rm = TRUE)
  mean_stranger <- mean(stranger_data, na.rm = TRUE)
  sd_self <- sd(self_data, na.rm = TRUE)
  sd_stranger <- sd(stranger_data, na.rm = TRUE)
  n_self <- length(self_data)
  n_stranger <- length(stranger_data)
  
  # Cohen's d (stranger - self, so positive = self faster)
  pooled_sd <- sqrt(((n_self - 1) * sd_self^2 + (n_stranger - 1) * sd_stranger^2) / (n_self + n_stranger - 2))
  d <- (mean_stranger - mean_self) / pooled_sd
  
  return(d)
}

# Bootstrap function for Cohen's d (ACC)
bootstrap_cohens_d_acc <- function(data, indices) {
  d <- data[indices, ]
  # Only include trials with valid RT (exclude timeout trials)
  d <- d[!is.na(d$RT_ms), ]
  # Separate self and stranger
  self_data <- d$Correct[d$Label == "self"]
  stranger_data <- d$Correct[d$Label == "stranger"]
  
  if (length(self_data) < 2 | length(stranger_data) < 2) {
    return(NA)
  }
  
  mean_self <- mean(self_data, na.rm = TRUE)
  mean_stranger <- mean(stranger_data, na.rm = TRUE)
  sd_self <- sd(self_data, na.rm = TRUE)
  sd_stranger <- sd(stranger_data, na.rm = TRUE)
  n_self <- length(self_data)
  n_stranger <- length(stranger_data)
  
  # Cohen's d (self - stranger, so positive = self more accurate)
  pooled_sd <- sqrt(((n_self - 1) * sd_self^2 + (n_stranger - 1) * sd_stranger^2) / (n_self + n_stranger - 2))
  d <- (mean_self - mean_stranger) / pooled_sd
  
  return(d)
}

# Run bootstrap for each design
cat("\nRunning bootstrap analysis for each design...\n")

results <- list()

for (i in 1:nrow(designs_S3)) {
  gid <- designs_S3$groupID[i]
  des <- designs_S3$Design[i]
  
  # Get data for this group
  group_data <- df_emp_match %>% dplyr::filter(groupID == gid)
  
  if (nrow(group_data) > 0) {
    # Run bootstrap for RT
    set.seed(42)
    boot_rt <- boot(group_data, bootstrap_cohens_d_rt, R = 1000)
    ci_rt <- boot.ci(boot_rt, type = "perc")
    
    # Run bootstrap for ACC
    set.seed(42)
    boot_acc <- boot(group_data, bootstrap_cohens_d_acc, R = 1000)
    ci_acc <- boot.ci(boot_acc, type = "perc")
    
    results[[i]] <- data.frame(
      Design = des,
      groupID = gid,
      P = designs_S3$P[i],
      T = designs_S3$T[i],
      W = designs_S3$W[i],
      n_subjects = length(unique(group_data$subjectID)),
      # RT
      d_RT = boot_rt$t0,
      d_RT_CI_lower = ci_rt$perc[4],
      d_RT_CI_upper = ci_rt$perc[5],
      # ACC
      d_ACC = boot_acc$t0,
      d_ACC_CI_lower = ci_acc$perc[4],
      d_ACC_CI_upper = ci_acc$perc[5]
    )
    
    cat(sprintf("Design %s: d_RT = %.3f [%.3f, %.3f], d_ACC = %.3f [%.3f, %.3f]\n", 
                des, results[[i]]$d_RT, results[[i]]$d_RT_CI_lower, results[[i]]$d_RT_CI_upper,
                results[[i]]$d_ACC, results[[i]]$d_ACC_CI_lower, results[[i]]$d_ACC_CI_upper))
  } else {
    results[[i]] <- data.frame(
      Design = des,
      groupID = gid,
      P = designs_S3$P[i],
      T = designs_S3$T[i],
      W = designs_S3$W[i],
      n_subjects = 0,
      d_RT = NA, d_RT_CI_lower = NA, d_RT_CI_upper = NA,
      d_ACC = NA, d_ACC_CI_lower = NA, d_ACC_CI_upper = NA
    )
    cat(sprintf("Design %s: NO DATA\n", des))
  }
}

# Combine results
summary_cohens_d <- dplyr::bind_rows(results)

# ==============================================================================
# 6. SAVE OUTPUT
# ==============================================================================

output_path <- here::here("Study3", "summary_emp_S3_cohens_d.csv")
write.csv(summary_cohens_d, output_path, row.names = FALSE)

cat("\n=== SUMMARY ===\n")
print(summary_cohens_d)
cat("\nSaved to:", output_path, "\n")
