###############################################################################
# 03_Interactive_CRF/interactive_crf.R
# 对标前端 renderCRFOverview(), renderCRFFileList() 等
# 对标 Python load_file_data(), load_all_data()
#
# 生成:
#   1. 单被试 CRF 曲线 (Self vs Stranger, Matching vs NonMatching)
#   2. 聚合组 CRF 曲线 + SPE 差异图
#   3. 所有被试 CRF 汇总图
#   4. 四条件 CRF 曲线 (Self/Stranger × Matching/NonMatching)
###############################################################################

# ===========================================================================
# 0. 初始化
# ===========================================================================
source(file.path( "shared", "utils.R"), chdir = TRUE)
OUT_DIR <- file.path(R_VERSION_DIR, "03_Interactive_CRF", "outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== 03_Interactive_CRF ===\n")

# ===========================================================================
# 1. 加载所有正式试次数据
# ===========================================================================
cat("Loading all raw data...\n")
all_data <- load_all_data()

# 过滤正式试次
if ("stage" %in% names(all_data)) {
  formal <- all_data[grepl("test", all_data$stage, ignore.case = TRUE), ]
} else {
  formal <- all_data
}

# 判定 Matching/NonMatching
formal$Matching <- ifelse(
  formal$CorrectKey == formal$Response,
  "Matching", "NonMatching"
)

# 判定 Identity (基于 Shape-Label 配对: Self 配对 vs Stranger 配对)
# 根据 CorrectKey 逻辑: 正确按键 f 或 j 对应 Self/Stranger
# 简化: 按 Label 区分 —— "self" label → Self, "stranger" label → Stranger
# (这里近似于前端逻辑)
formal$Identity <- ifelse(
  grepl("self", tolower(formal$Label)), "Self",
  ifelse(grepl("stranger", tolower(formal$Label)), "Stranger", "Unknown")
)

# 统一 RT 列到毫秒
formal$RT_ms <- as.numeric(formal$RT)
formal <- formal[!is.na(formal$RT_ms) & formal$RT_ms > 0 & formal$RT_ms < 5000, ]

cat("Formal trials:", nrow(formal), "\n")
cat("Unique subjects:", length(unique(formal$subjectID)), "\n")

# ===========================================================================
# 2. 单被试 CRF 曲线（所有被试同时生成）
# ===========================================================================
cat("\n--- Generating Single-Subject CRF Curves ---\n")

generate_single_crf <- function(subj_data, n_q = 5) {
  bins_self <- compute_crf_bins(
    subj_data, identity_sel = "Self", n_quantiles = n_q, y_mode = "pMatch"
  )
  bins_stranger <- compute_crf_bins(
    subj_data, identity_sel = "Stranger", n_quantiles = n_q, y_mode = "pMatch"
  )
  if (nrow(bins_self) == 0 && nrow(bins_stranger) == 0) return(NULL)
  bind_rows(
    data.frame(bins_self, Identity = "Self", stringsAsFactors = FALSE),
    data.frame(bins_stranger, Identity = "Stranger", stringsAsFactors = FALSE)
  )
}

# 为所有被试生成 CRF，用于聚合分析
subjects <- unique(formal$subjectID)
crf_all <- lapply(subjects, function(sid) {
  sdata <- formal[formal$subjectID == sid, ]
  if (nrow(sdata) < 20) return(NULL)
  bins <- generate_single_crf(sdata, n_q = 5)
  if (is.null(bins)) return(NULL)
  bins$Subject <- sid
  bins$GroupID <- unique(sdata$groupID)[1]
  bins
})
crf_all <- bind_rows(crf_all[!sapply(crf_all, is.null)])
cat("CRF data rows:", nrow(crf_all), "\n")

# 每个被试的 CRF 图
plot_single_crf <- function(bins, title_str) {
  ggplot(bins, aes(x = x, y = y, color = Identity, fill = Identity)) +
    geom_ribbon(aes(ymin = pmax(0, y - ySEM), ymax = pmin(1, y + ySEM)),
                alpha = 0.15, color = NA) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey60", linewidth = 0.5) +
    scale_color_manual(values = c("Self" = "#ff9800", "Stranger" = "#2196f3")) +
    scale_fill_manual(values = c("Self" = "#ff9800", "Stranger" = "#2196f3")) +
    scale_x_continuous(name = "RT (ms)") +
    scale_y_continuous(name = "P(Matching)", limits = c(0, 1),
                       labels = scales::percent_format(accuracy = 1)) +
    labs(title = title_str, subtitle = paste0("Quantile bins: ", 5)) +
    theme_spe()
}

# 前 16 个被试的 CRF 面板
example_sids <- unique(crf_all$Subject)[1:min(16, length(unique(crf_all$Subject)))]
crf_plots <- lapply(example_sids, function(sid) {
  plot_single_crf(crf_all[crf_all$Subject == sid, ], paste0("Subject ", sid))
})
p_grid <- cowplot::plot_grid(plotlist = crf_plots, ncol = 4)
save_plot_png(p_grid, file.path(OUT_DIR, "CRF_Single_Subject_Grid.png"),
              width = 16, height = 12, dpi = 150)

# ===========================================================================
# 3. 聚合组 CRF + SPE 差异图
# ===========================================================================
cat("\n--- Generating Group-Aggregated CRF + SPE Difference ---\n")

# 全部数据聚合 CRF
crf_self_all   <- compute_crf_bins(formal, identity_sel = "Self",    n_quantiles = 5, y_mode = "pMatch")
crf_strang_all <- compute_crf_bins(formal, identity_sel = "Stranger", n_quantiles = 5, y_mode = "pMatch")
crf_agg <- bind_rows(
  data.frame(crf_self_all,   Identity = "Self",    stringsAsFactors = FALSE),
  data.frame(crf_strang_all, Identity = "Stranger", stringsAsFactors = FALSE)
)

p_agg_crf <- plot_single_crf(crf_agg, "Aggregated CRF — All Groups (N = all subjects)")
save_plot_png(p_agg_crf, file.path(OUT_DIR, "CRF_Aggregated_All_Groups.png"))

# SPE 差异图 (Self - Stranger)
if (nrow(crf_self_all) >= 2 && nrow(crf_strang_all) >= 2) {
  spe_diff <- data.frame(
    x       = crf_self_all$x,
    y       = crf_self_all$y - crf_strang_all$y,
    ySEM    = sqrt(crf_self_all$ySEM^2 + crf_strang_all$ySEM^2),
    stringsAsFactors = FALSE
  )
  p_spe <- ggplot(spe_diff, aes(x = x, y = y)) +
    geom_ribbon(aes(ymin = y - ySEM, ymax = y + ySEM),
                fill = "#e91e63", alpha = 0.15) +
    geom_line(color = "#e91e63", linewidth = 1.5) +
    geom_point(color = "#e91e63", size = 3) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    labs(title = "SPE = Self − Stranger  (Aggregated)",
         x = "RT (ms)", y = expression(Delta * " P(Matching)")) +
    theme_spe()
  save_plot_png(p_spe, file.path(OUT_DIR, "CRF_SPE_Difference_All_Groups.png"))

  # 合并图
  p_combined <- cowplot::plot_grid(p_agg_crf, p_spe, ncol = 1, rel_heights = c(1, 0.8))
  save_plot_png(p_combined, file.path(OUT_DIR, "CRF_Combined_CRF_SPE.png"), height = 8)
}

# 按组聚合 CRF
cat("\n--- Generating Group-Specific Aggregated CRF ---\n")
group_crf_list <- list()
for (g in sort(unique(formal$groupID))) {
  gdata <- formal[formal$groupID == g, ]
  if (nrow(gdata) < 30) next
  self_bins   <- compute_crf_bins(gdata, identity_sel = "Self",    n_quantiles = 5, y_mode = "pMatch")
  strang_bins <- compute_crf_bins(gdata, identity_sel = "Stranger", n_quantiles = 5, y_mode = "pMatch")
  if (nrow(self_bins) < 2 || nrow(strang_bins) < 2) next
  g_bins <- bind_rows(
    data.frame(self_bins,   Identity = "Self", stringsAsFactors = FALSE),
    data.frame(strang_bins, Identity = "Stranger", stringsAsFactors = FALSE)
  )
  g_bins$Group <- g
  group_crf_list[[as.character(g)]] <- g_bins
}
crf_by_group <- bind_rows(group_crf_list)

p_group_crf <- ggplot(crf_by_group, aes(x = x, y = y, color = Identity)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  facet_wrap(~ Group, ncol = 4) +
  scale_color_manual(values = c("Self" = "#ff9800", "Stranger" = "#2196f3")) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey60", linewidth = 0.3) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "CRF by Group", x = "RT (ms)", y = "P(Matching)") +
  theme_spe(base_size = 9)
save_plot_png(p_group_crf, file.path(OUT_DIR, "CRF_By_Group.png"),
              width = 14, height = 10)

# ===========================================================================
# 4. 四条件 CRF (Self/Stranger × Matching/NonMatching)
# ===========================================================================
cat("\n--- Generating 4-Condition CRF ---\n")

crf_4cond <- bind_rows(
  data.frame(compute_crf_bins(formal, identity_sel = "Self",    matching_sel = "Matching",    n_quantiles = 5), Condition = "Self_Matching",    stringsAsFactors = FALSE),
  data.frame(compute_crf_bins(formal, identity_sel = "Self",    matching_sel = "NonMatching",   n_quantiles = 5), Condition = "Self_NonMatching",   stringsAsFactors = FALSE),
  data.frame(compute_crf_bins(formal, identity_sel = "Stranger", matching_sel = "Matching",    n_quantiles = 5), Condition = "Stranger_Matching",  stringsAsFactors = FALSE),
  data.frame(compute_crf_bins(formal, identity_sel = "Stranger", matching_sel = "NonMatching",   n_quantiles = 5), Condition = "Stranger_NonMatching", stringsAsFactors = FALSE)
)

crf_colors <- c(
  "Self_Matching"      = "#ff9800",
  "Self_NonMatching"   = "#ffcc80",
  "Stranger_Matching"  = "#2196f3",
  "Stranger_NonMatching" = "#90caf9"
)
crf_linetypes <- c(
  "Self_Matching"      = "solid",
  "Self_NonMatching"   = "dashed",
  "Stranger_Matching"  = "solid",
  "Stranger_NonMatching" = "dashed"
)

p_4cond <- ggplot(crf_4cond, aes(x = x, y = y, color = Condition, linetype = Condition)) +
  geom_ribbon(aes(ymin = pmax(0, y - ySEM), ymax = pmin(1, y + ySEM), fill = Condition),
              alpha = 0.08, color = NA) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  scale_color_manual(values = crf_colors) +
  scale_fill_manual(values = crf_colors) +
  scale_linetype_manual(values = crf_linetypes) +
  scale_x_continuous(name = "RT (ms)") +
  scale_y_continuous(name = "P(Matching)", limits = c(0, 1),
                     labels = percent_format(accuracy = 1)) +
  labs(title = "4-Condition CRF (Self/Stranger × Matching/NonMatching)",
       subtitle = paste0("All groups aggregated | n = ", nrow(formal), " trials")) +
  theme_spe()
save_plot_png(p_4cond, file.path(OUT_DIR, "CRF_Four_Conditions.png"))

# ===========================================================================
# 5. ACC (正确率) 模式的 CRF
# ===========================================================================
cat("\n--- Generating ACC-mode CRF ---\n")

crf_self_acc   <- compute_crf_bins(formal, identity_sel = "Self",    n_quantiles = 5, y_mode = "acc")
crf_strang_acc <- compute_crf_bins(formal, identity_sel = "Stranger", n_quantiles = 5, y_mode = "acc")
crf_acc <- bind_rows(
  data.frame(crf_self_acc,   Identity = "Self",    stringsAsFactors = FALSE),
  data.frame(crf_strang_acc, Identity = "Stranger", stringsAsFactors = FALSE)
)

p_acc <- ggplot(crf_acc, aes(x = x, y = y, color = Identity, fill = Identity)) +
  geom_ribbon(aes(ymin = pmax(0, y - ySEM), ymax = pmin(1, y + ySEM)),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  scale_color_manual(values = c("Self" = "#ff9800", "Stranger" = "#2196f3")) +
  scale_fill_manual(values = c("Self" = "#ff9800", "Stranger" = "#2196f3")) +
  scale_x_continuous(name = "RT (ms)") +
  scale_y_continuous(name = "ACC", limits = c(0, 1),
                     labels = percent_format(accuracy = 1)) +
  labs(title = "CRF (ACC mode) — Self vs Stranger Accuracy by RT Quantile",
       subtitle = "All groups aggregated") +
  theme_spe()
save_plot_png(p_acc, file.path(OUT_DIR, "CRF_ACC_Mode.png"))

# 合并 P(Matching) + ACC 模式
p_both <- cowplot::plot_grid(
  p_agg_crf + ggtitle("P(Matching) Mode"),
  p_acc + ggtitle("ACC Mode"),
  ncol = 1, rel_heights = c(1, 1)
)
save_plot_png(p_both, file.path(OUT_DIR, "CRF_Both_Modes.png"), height = 9)

cat("\n=== 03_Interactive_CRF DONE ===\n")
cat("Outputs saved to:", OUT_DIR, "\n")
