###############################################################################
# 05_SPE_Database/spe_individual_detail.R
# 对标: load_spe_experiment_detail() + renderSPEIndividual()
# 生成: 个体 SPE 条形图、RT 分布对比直方图、SPE 瀑布图（累积效应量）
###############################################################################

source(file.path("..", "shared", "utils.R"), chdir = TRUE)
OUT_DIR <- file.path(R_VERSION_DIR, "05_SPE_Database", "outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== 05_SPE_Database — Individual Detail ===\n")

# ===========================================================================
# 1. 加载 SPE 实验列表
# ===========================================================================
spe_csv <- file.path(OUT_DIR, "spe_overview_data.csv")
if (!file.exists(spe_csv)) stop("Run spe_overview.R first")
spe_df <- read.csv(spe_csv, stringsAsFactors = FALSE)

# 选取前 8 个有效实验生成详细图
valid_exps <- spe_df[!is.na(spe_df$spe_rt_d), ]
example_eks <- head(valid_exps$pairKey, 8)
cat("Example experiments:", paste(example_eks, collapse = ", "), "\n")

# ===========================================================================
# 2. 为每个实验生成个体详情图
# ===========================================================================
plots_indiv <- list()

for (pk in example_eks) {
  cat("  Processing:", pk, "\n")
  df <- tryCatch(load_spe_file(pk), error = function(e) NULL)
  if (is.null(df)) next

  id_col <- select_identity_column(df, "label")
  if (is.null(id_col)) next

  result <- compute_subject_spe(df, id_col)
  primary_other <- result$primary_comparison
  self_key <- "Self"

  # 每位被试的 SPE
  subj_spe <- data.frame(
    Subject = character(0),
    spe_rt_d = numeric(0),
    self_rt  = numeric(0),
    other_rt = numeric(0),
    stringsAsFactors = FALSE
  )
  for (sid in sort(unique(df$Subject), method = "radix")) {
    s_rt <- df$RT_ms[df$Subject == sid & df[[id_col]] == self_key]
    o_rt <- df$RT_ms[df$Subject == sid & df[[id_col]] == primary_other]
    s_rt <- as.numeric(s_rt); o_rt <- as.numeric(o_rt)
    s_rt <- s_rt[!is.na(s_rt) & s_rt > 0]
    o_rt <- o_rt[!is.na(o_rt) & o_rt > 0]
    if (length(s_rt) >= 3 && length(o_rt) >= 3) {
      d_val <- cohens_d(o_rt, s_rt)
      subj_spe <- rbind(subj_spe, data.frame(
        Subject = as.character(sid),
        spe_rt_d = d_val,
        self_rt  = mean(s_rt),
        other_rt = mean(o_rt),
        stringsAsFactors = FALSE
      ))
    }
  }

  if (nrow(subj_spe) < 2) next

  # --- (a) SPE 条形图 ---
  subj_spe <- subj_spe[order(subj_spe$spe_rt_d, decreasing = TRUE), ]
  subj_spe$Subject <- factor(subj_spe$Subject, levels = subj_spe$Subject)
  subj_spe$color_grp <- ifelse(subj_spe$spe_rt_d >= 0, "#ff9800", "#2196f3")

  p_bar <- ggplot(subj_spe, aes(x = Subject, y = spe_rt_d, fill = color_grp)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = 0, color = "grey40", linewidth = 0.4) +
    scale_fill_identity() +
    labs(title = paste0("Per-Subject SPE RT — ", pk),
         subtitle = paste0("Self vs ", primary_other, " | N = ", nrow(subj_spe)),
         y = "SPE RT (Cohen's d)") +
    theme_spe(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))

  # --- (b) RT 分布对比 ---
  rt_self_long   <- data.frame(RT = result$legacy$self_rts_all,    Identity = self_key, stringsAsFactors = FALSE)
  rt_other_long  <- data.frame(RT = result$legacy$stranger_rts_all, Identity = primary_other, stringsAsFactors = FALSE)
  rt_long <- rbind(rt_self_long, rt_other_long)

  p_rt <- ggplot(rt_long, aes(x = RT, fill = Identity)) +
    geom_histogram(bins = 30, alpha = 0.5, position = "identity", color = "white", linewidth = 0.2) +
    scale_fill_manual(values = setNames(c("#ff9800", "#2196f3"), c(self_key, primary_other))) +
    labs(title = "RT Distribution", x = "RT (ms)", y = "Count") +
    theme_spe(base_size = 9)

  # --- (c) SPE 瀑布图 (累积) ---
  subj_spe_sorted <- subj_spe[order(subj_spe$spe_rt_d, decreasing = TRUE), ]
  subj_spe_sorted$cumsum <- cumsum(subj_spe_sorted$spe_rt_d)
  subj_spe_sorted$idx <- seq_len(nrow(subj_spe_sorted))

  p_water <- ggplot(subj_spe_sorted, aes(x = idx, y = cumsum)) +
    geom_area(fill = "#e91e63", alpha = 0.2) +
    geom_line(color = "#e91e63", linewidth = 1) +
    geom_point(color = "#e91e63", size = 1.5) +
    geom_hline(yintercept = 0, color = "grey50", linewidth = 0.3) +
    labs(title = "Cumulative SPE (Waterfall)",
         x = "Subjects (sorted by SPE)", y = "Cumulative SPE (Cohen's d)",
         subtitle = paste0("Final cumulative: ", round(tail(subj_spe_sorted$cumsum, 1), 3))) +
    theme_spe(base_size = 9)

  # 合并三图
  p_triple <- cowplot::plot_grid(p_bar, p_rt, p_water, ncol = 1,
                                   rel_heights = c(1, 0.8, 0.8))
  plots_indiv[[pk]] <- p_triple
}

# 保存为 PDF
if (length(plots_indiv) > 0) {
  pdf(file.path(OUT_DIR, "SPE_Individual_Details.pdf"), width = 10, height = 12)
  for (pk in names(plots_indiv)) {
    print(plots_indiv[[pk]])
  }
  dev.off()
  cat("PDF saved:", file.path(OUT_DIR, "SPE_Individual_Details.pdf"), "\n")
}

# 首个实验的 PNG 预览
if (length(plots_indiv) > 0) {
  save_plot_png(plots_indiv[[1]],
                file.path(OUT_DIR, "SPE_Individual_Example.png"),
                width = 10, height = 12, dpi = 150)
}

cat("\n=== SPE Individual Detail DONE ===\n")
