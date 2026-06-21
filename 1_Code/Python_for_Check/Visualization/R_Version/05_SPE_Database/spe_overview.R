###############################################################################
# 05_SPE_Database/spe_overview.R
# 对标: load_spe_overview() + renderSPEOverview()
# 生成: SPE RT 直方图、SPE ACC 直方图、实验列表统计
###############################################################################

source(file.path("shared", "utils.R"), chdir = TRUE)
OUT_DIR <- file.path(R_VERSION_DIR, "05_SPE_Database", "outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== 05_SPE_Database — Group Overview ===\n")

# ===========================================================================
# 1. 加载处理日志并批量计算 SPE
# ===========================================================================
log <- load_spe_log()
cat("Processing log entries:", nrow(log), "\n")

results <- list()
for (i in seq_len(nrow(log))) {
  entry <- log[i, ]
  pk <- entry$Pair_Key
  sp_file <- file.path(SPE_DIR, basename(entry$Output_File))
  if (!file.exists(sp_file)) {
    cat("  SKIP (file missing):", pk, "\n")
    next
  }
  df <- tryCatch(read.csv(sp_file, stringsAsFactors = FALSE, fileEncoding = "UTF-8"),
                 error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) next

  # Select identity column (default: Label_Standardized_Identity)
  id_col <- select_identity_column(df, "label")
  if (is.null(id_col)) {
    cat("  SKIP (no identity col):", pk, "\n")
    next
  }

  # SPE for all, matching, nonmatching
  result_all <- compute_subject_spe(df, id_col, condition_filter = "all")
  result_m   <- tryCatch(compute_subject_spe(df, id_col, condition_filter = "Matching"), error = function(e) NULL)
  result_nm  <- tryCatch(compute_subject_spe(df, id_col, condition_filter = "NonMatching"), error = function(e) NULL)

  leg <- result_all$legacy
  leg_m <- if (!is.null(result_m)) result_m$legacy else NULL
  leg_nm <- if (!is.null(result_nm)) result_nm$legacy else NULL

  P_val <- suppressWarnings(as.numeric(entry$P_Parsed_ms))
  T_val <- suppressWarnings(as.numeric(entry$T_Parsed_ms))
  W_val <- suppressWarnings(as.numeric(entry$W_Parsed_ms))

  results[[pk]] <- data.frame(
    pairKey          = pk,
    n_subjects       = result_all$n_subjects,
    n_valid          = leg$n_subjects_valid,
    P_ms             = if (!is.na(P_val)) P_val else NA_real_,
    T_ms             = if (!is.na(T_val)) T_val else NA_real_,
    W_ms             = if (!is.na(W_val)) W_val else NA_real_,
    spe_rt_d         = if (length(leg$d_vals) > 0) round(mean(leg$d_vals), 4) else NA_real_,
    spe_acc_d        = if (length(leg$acc_d_vals) > 0) round(mean(leg$acc_d_vals), 4) else NA_real_,
    spe_rt_d_m       = if (!is.null(leg_m) && length(leg_m$d_vals) > 0) round(mean(leg_m$d_vals), 4) else NA_real_,
    spe_acc_d_m      = if (!is.null(leg_m) && length(leg_m$acc_d_vals) > 0) round(mean(leg_m$acc_d_vals), 4) else NA_real_,
    spe_rt_d_nm      = if (!is.null(leg_nm) && length(leg_nm$d_vals) > 0) round(mean(leg_nm$d_vals), 4) else NA_real_,
    spe_acc_d_nm     = if (!is.null(leg_nm) && length(leg_nm$acc_d_vals) > 0) round(mean(leg_nm$acc_d_vals), 4) else NA_real_,
    self_mean_rt     = if (length(leg$self_rts_all)) round(mean(leg$self_rts_all), 1) else NA_real_,
    stranger_mean_rt = if (length(leg$stranger_rts_all)) round(mean(leg$stranger_rts_all), 1) else NA_real_,
    identity_types   = paste(result_all$identity_types, collapse = ","),
    primary_comp     = if (is.null(result_all$primary_comparison)) "None" else result_all$primary_comparison,
    stringsAsFactors = FALSE
  )
  cat(sprintf("  [%3d/%3d] %s d=%.3f n=%d\n", i, nrow(log), pk,
              results[[pk]]$spe_rt_d, results[[pk]]$n_valid))
}

spe_df <- bind_rows(results)
cat("\nTotal valid experiments:", nrow(spe_df), "\n")

# 保存数据表
write.csv(spe_df, file.path(OUT_DIR, "spe_overview_data.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# ===========================================================================
# 2. SPE(RT) 直方图
# ===========================================================================
cat("\n--- SPE RT Histogram ---\n")
valid_rt <- spe_df[!is.na(spe_df$spe_rt_d), ]
cat("Valid RT SPE entries:", nrow(valid_rt), "\n")

# 统计量
mean_rt <- mean(valid_rt$spe_rt_d)
sd_rt   <- sd(valid_rt$spe_rt_d)
min_rt  <- min(valid_rt$spe_rt_d)
max_rt  <- max(valid_rt$spe_rt_d)
cat(sprintf("SPE(RT) mean=%.3f, sd=%.3f, range=[%.3f, %.3f]\n",
            mean_rt, sd_rt, min_rt, max_rt))

p_rt_hist <- ggplot(valid_rt, aes(x = spe_rt_d)) +
  geom_histogram(bins = 20, fill = "#ff9800", alpha = 0.7, color = "white", linewidth = 0.3) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  geom_vline(xintercept = mean_rt, color = "#e65100", linewidth = 1, linetype = "dotted") +
  labs(
    title = "SPE (RT) — Cohen's d Distribution",
    subtitle = sprintf("Mean = %.3f, SD = %.3f, Range [%.3f, %.3f] | N = %d experiments",
                       mean_rt, sd_rt, min_rt, max_rt, nrow(valid_rt)),
    x = expression("SPE (RT Cohen's d) (Other − Self)"), y = "Count"
  ) +
  theme_spe()
save_plot_png(p_rt_hist, file.path(OUT_DIR, "SPE_RT_Histogram.png"))

# ===========================================================================
# 3. SPE(ACC) 直方图
# ===========================================================================
cat("\n--- SPE ACC Histogram ---\n")
valid_acc <- spe_df[!is.na(spe_df$spe_acc_d), ]
cat("Valid ACC SPE entries:", nrow(valid_acc), "\n")

mean_acc <- mean(valid_acc$spe_acc_d)
sd_acc   <- sd(valid_acc$spe_acc_d)

p_acc_hist <- ggplot(valid_acc, aes(x = spe_acc_d)) +
  geom_histogram(bins = 20, fill = "#4caf50", alpha = 0.7, color = "white", linewidth = 0.3) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  geom_vline(xintercept = mean_acc, color = "#2e7d32", linewidth = 1, linetype = "dotted") +
  labs(
    title = "SPE (ACC) — Cohen's d Distribution",
    subtitle = sprintf("Mean = %.3f, SD = %.3f | N = %d experiments", mean_acc, sd_acc, nrow(valid_acc)),
    x = expression("SPE (ACC Cohen's d) (Self − Other)"), y = "Count"
  ) +
  theme_spe()
save_plot_png(p_acc_hist, file.path(OUT_DIR, "SPE_ACC_Histogram.png"))

# ===========================================================================
# 4. SPE(RT) vs SPE(ACC) 散点图
# ===========================================================================
cat("\n--- SPE RT vs ACC Scatter ---\n")
both_valid <- spe_df[!is.na(spe_df$spe_rt_d) & !is.na(spe_df$spe_acc_d), ]
cor_rt_acc <- cor(both_valid$spe_rt_d, both_valid$spe_acc_d)
cat(sprintf("RT-ACC correlation: r = %.3f\n", cor_rt_acc))

p_scatter_sp <- ggplot(both_valid, aes(x = spe_rt_d, y = spe_acc_d)) +
  geom_point(color = "#9c27b0", alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "#e91e63", linewidth = 0.8, linetype = "dashed") +
  geom_hline(yintercept = 0, color = "grey70", linewidth = 0.3) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.3) +
  labs(
    title = "SPE(RT) vs SPE(ACC)",
    subtitle = sprintf("r = %.3f | N = %d", cor_rt_acc, nrow(both_valid)),
    x = "SPE RT (Cohen's d)", y = "SPE ACC (Cohen's d)"
  ) +
  theme_spe()
save_plot_png(p_scatter_sp, file.path(OUT_DIR, "SPE_RT_vs_ACC.png"))

# ===========================================================================
# 5. 双直方图对比
# ===========================================================================
p_dual <- cowplot::plot_grid(p_rt_hist, p_acc_hist, ncol = 1,
                              rel_heights = c(1, 1),
                              labels = c("A", "B"), label_size = 12)
save_plot_png(p_dual, file.path(OUT_DIR, "SPE_Overview_Dual_Histograms.png"), height = 9)

cat("\n=== SPE Overview DONE ===\n")

