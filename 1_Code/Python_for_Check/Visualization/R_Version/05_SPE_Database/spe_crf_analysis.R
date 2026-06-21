###############################################################################
# 05_SPE_Database/spe_crf_analysis.R
# 对标: load_spe_trials() + renderSPECRF()
# 生成: 
#   - 分组模式: 4条件 CRF (Self/Stranger × Matching/NonMatching)
#   - 整体模式: 2线 CRF (Self Overall vs Stranger Overall)
#   - 支持 P(Matching) 和 ACC 两种 Y 轴模式
###############################################################################

source(file.path("..", "shared", "utils.R"), chdir = TRUE)
OUT_DIR <- file.path(R_VERSION_DIR, "05_SPE_Database", "outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== 05_SPE_Database — CRF Analysis ===\n")

# ===========================================================================
# 1. 加载实验
# ===========================================================================
spe_csv <- file.path(OUT_DIR, "spe_overview_data.csv")
if (!file.exists(spe_csv)) stop("Run spe_overview.R first")
spe_df <- read.csv(spe_csv, stringsAsFactors = FALSE)
valid_exps <- spe_df[!is.na(spe_df$spe_rt_d), ]

# 选取前 4 个有 Matching 列的有效实验
plot_count <- 0
for (pk in valid_exps$pairKey) {
  cat("  Loading trials for:", pk, "\n")
  df <- tryCatch(load_spe_file(pk), error = function(e) NULL)
  if (is.null(df) || nrow(df) < 20) next
  if (!"Matching" %in% names(df)) next  # need Matching column

  id_col <- select_identity_column(df, "label")
  if (is.null(id_col)) next

  # 构建 lightweight trial 格式
  df$RT_ms <- as.numeric(df$RT_ms)
  df$ACC   <- as.numeric(df$ACC)
  df <- df[!is.na(df$RT_ms) & df$RT_ms > 0, ]

  trials <- df[, c("Subject", id_col, "RT_ms", "Matching", "ACC")]
  names(trials)[2] <- "Identity"
  trials <- trials[trials$Identity %in% c("Self", "Stranger"), ]

  if (nrow(trials) < 30) next

  # =========================================================================
  # 2. 分组模式 — 4条件 CRF
  # =========================================================================
  n_q <- 5  # 分位数

  bins_4cond <- bind_rows(
    data.frame(compute_crf_bins(trials, identity_sel = "Self",    matching_sel = "Matching",    n_quantiles = n_q, y_mode = "pMatch"),
               Condition = "Self_Matching",    stringsAsFactors = FALSE),
    data.frame(compute_crf_bins(trials, identity_sel = "Self",    matching_sel = "NonMatching",   n_quantiles = n_q, y_mode = "pMatch"),
               Condition = "Self_NonMatching",   stringsAsFactors = FALSE),
    data.frame(compute_crf_bins(trials, identity_sel = "Stranger", matching_sel = "Matching",    n_quantiles = n_q, y_mode = "pMatch"),
               Condition = "Stranger_Matching",  stringsAsFactors = FALSE),
    data.frame(compute_crf_bins(trials, identity_sel = "Stranger", matching_sel = "NonMatching",   n_quantiles = n_q, y_mode = "pMatch"),
               Condition = "Stranger_NonMatching", stringsAsFactors = FALSE)
  )

  if (nrow(bins_4cond) < 8) next

  crf_colors <- c(
    "Self_Matching"       = "#ff9800",
    "Self_NonMatching"    = "#ffcc80",
    "Stranger_Matching"   = "#2196f3",
    "Stranger_NonMatching" = "#90caf9"
  )
  crf_ltypes <- c(
    "Self_Matching"       = "solid",
    "Self_NonMatching"    = "dashed",
    "Stranger_Matching"   = "solid",
    "Stranger_NonMatching" = "dashed"
  )

  # =========================================================================
  # 2a. P(Matching) 模式
  # =========================================================================
  p_4cond <- ggplot(bins_4cond, aes(x = x, y = y, color = Condition, linetype = Condition)) +
    geom_ribbon(aes(ymin = pmax(0, y - ySEM), ymax = pmin(1, y + ySEM), fill = Condition),
                alpha = 0.08, color = NA) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2) +
    scale_color_manual(values = crf_colors) +
    scale_fill_manual(values = crf_colors) +
    scale_linetype_manual(values = crf_ltypes) +
    scale_x_continuous(name = "RT (ms)") +
    scale_y_continuous(name = "P(Matching)", limits = c(0, 1),
                       labels = percent_format(accuracy = 1)) +
    labs(title = paste0("CRF — ", pk, " (P(Matching) Mode)"),
         subtitle = paste0("4 Conditions | Quantiles = ", n_q, " | n = ", nrow(trials), " trials")) +
    theme_spe()

  # =========================================================================
  # 2b. ACC 模式
  # =========================================================================
  bins_4cond_acc <- bind_rows(
    data.frame(compute_crf_bins(trials, identity_sel = "Self",    matching_sel = "Matching",    n_quantiles = n_q, y_mode = "acc"),
               Condition = "Self_Matching",    stringsAsFactors = FALSE),
    data.frame(compute_crf_bins(trials, identity_sel = "Self",    matching_sel = "NonMatching",   n_quantiles = n_q, y_mode = "acc"),
               Condition = "Self_NonMatching",   stringsAsFactors = FALSE),
    data.frame(compute_crf_bins(trials, identity_sel = "Stranger", matching_sel = "Matching",    n_quantiles = n_q, y_mode = "acc"),
               Condition = "Stranger_Matching",  stringsAsFactors = FALSE),
    data.frame(compute_crf_bins(trials, identity_sel = "Stranger", matching_sel = "NonMatching",   n_quantiles = n_q, y_mode = "acc"),
               Condition = "Stranger_NonMatching", stringsAsFactors = FALSE)
  )

  p_4cond_acc <- ggplot(bins_4cond_acc, aes(x = x, y = y, color = Condition, linetype = Condition)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2) +
    scale_color_manual(values = crf_colors) +
    scale_linetype_manual(values = crf_ltypes) +
    scale_x_continuous(name = "RT (ms)") +
    scale_y_continuous(name = "ACC", limits = c(0, 1), labels = percent_format(accuracy = 1)) +
    labs(title = paste0("CRF — ", pk, " (ACC Mode)")) +
    theme_spe()

  # 合并两种模式
  p_grouped <- cowplot::plot_grid(p_4cond, p_4cond_acc, ncol = 1,
                                   rel_heights = c(1, 1),
                                   labels = c("A", "B"), label_size = 11)
  save_plot_png(p_grouped,
                file.path(OUT_DIR, paste0("CRF_", gsub("[^A-Za-z0-9_]", "_", pk), "_Grouped.png")),
                width = 10, height = 9)

  # =========================================================================
  # 3. 整体模式 — Self Overall vs Stranger Overall (合并 Matching/NonMatching)
  # =========================================================================
  bins_self_overall   <- compute_crf_bins(trials, identity_sel = "Self",    n_quantiles = n_q, y_mode = "pMatch")
  bins_strang_overall <- compute_crf_bins(trials, identity_sel = "Stranger", n_quantiles = n_q, y_mode = "pMatch")

  if (nrow(bins_self_overall) >= 3 && nrow(bins_strang_overall) >= 3) {
    crf_overall <- bind_rows(
      data.frame(bins_self_overall,   Identity = "Self (Overall)",    stringsAsFactors = FALSE),
      data.frame(bins_strang_overall, Identity = "Stranger (Overall)", stringsAsFactors = FALSE)
    )

    p_overall <- ggplot(crf_overall, aes(x = x, y = y, color = Identity, fill = Identity)) +
      geom_ribbon(aes(ymin = pmax(0, y - ySEM), ymax = pmin(1, y + ySEM)),
                  alpha = 0.12, color = NA) +
      geom_line(linewidth = 1.3) +
      geom_point(size = 3) +
      geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50", linewidth = 0.5) +
      scale_color_manual(values = c("Self (Overall)" = "#ff9800", "Stranger (Overall)" = "#2196f3")) +
      scale_fill_manual(values = c("Self (Overall)" = "#ff9800", "Stranger (Overall)" = "#2196f3")) +
      scale_x_continuous(name = "RT (ms)") +
      scale_y_continuous(name = "P(Matching)", limits = c(0, 1),
                         labels = percent_format(accuracy = 1)) +
      labs(title = paste0("CRF — ", pk, " (Overall Mode)"),
           subtitle = "All Matching/NonMatching trials combined | Self vs Stranger") +
      theme_spe()
    save_plot_png(p_overall,
                  file.path(OUT_DIR, paste0("CRF_", gsub("[^A-Za-z0-9_]", "_", pk), "_Overall.png")))
  }

  plot_count <- plot_count + 1
  if (plot_count >= 4) break
}

# ===========================================================================
# 4. 所有实验的 CRF 面板图（分组模式）
# ===========================================================================
cat("\n--- Generating Multi-Experiment CRF Panel ---\n")
panel_plots <- list()
plot_count <- 0

for (pk in valid_exps$pairKey) {
  df <- tryCatch(load_spe_file(pk), error = function(e) NULL)
  if (is.null(df) || nrow(df) < 30 || !"Matching" %in% names(df)) next

  id_col <- select_identity_column(df, "label")
  if (is.null(id_col)) next

  df$RT_ms <- as.numeric(df$RT_ms)
  df <- df[!is.na(df$RT_ms) & df$RT_ms > 0, ]
  trials <- df[, c("Subject", id_col, "RT_ms", "Matching", "ACC")]
  names(trials)[2] <- "Identity"
  trials <- trials[trials$Identity %in% c("Self", "Stranger"), ]
  if (nrow(trials) < 30) next

  self_bins   <- compute_crf_bins(trials, identity_sel = "Self",    n_quantiles = 5, y_mode = "pMatch")
  strang_bins <- compute_crf_bins(trials, identity_sel = "Stranger", n_quantiles = 5, y_mode = "pMatch")
  if (nrow(self_bins) < 3 || nrow(strang_bins) < 3) next

  bins <- bind_rows(
    data.frame(self_bins, Identity = "Self", stringsAsFactors = FALSE),
    data.frame(strang_bins, Identity = "Stranger", stringsAsFactors = FALSE)
  )

  p <- ggplot(bins, aes(x = x, y = y, color = Identity)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.5) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey70", linewidth = 0.3) +
    scale_color_manual(values = c("Self" = "#ff9800", "Stranger" = "#2196f3")) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(title = pk, x = NULL, y = NULL) +
    theme_spe(base_size = 7) +
    theme(legend.position = "none", plot.title = element_text(size = 7, face = "plain"))

  panel_plots[[pk]] <- p
  plot_count <- plot_count + 1
  if (plot_count >= 16) break
}

if (length(panel_plots) > 0) {
  p_panel <- cowplot::plot_grid(plotlist = panel_plots, ncol = 4)
  save_plot_png(p_panel, file.path(OUT_DIR, "CRF_Multi_Experiment_Panel.png"),
                width = 16, height = 12, dpi = 150)
}

cat("\n=== SPE CRF Analysis DONE ===\n")
