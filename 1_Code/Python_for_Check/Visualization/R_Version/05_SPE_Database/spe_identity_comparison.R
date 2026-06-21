###############################################################################
# 05_SPE_Database/spe_identity_comparison.R
# 对标: load_identity_summary() 和前端 Identity 筛选维度
# 生成: 
#   - 所有 Identity 类型在多少实验中出现
#   - Self vs 每种其他 Identity 的 SPE 比较（森林图）
#   - 多 Identity SPE 对比箱线图
###############################################################################

source(file.path("..", "shared", "utils.R"), chdir = TRUE)
OUT_DIR <- file.path(R_VERSION_DIR, "05_SPE_Database", "outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== 05_SPE_Database — Multi-Identity Comparison ===\n")

# ===========================================================================
# 1. 收集所有实验的 Identity 分布
# ===========================================================================
log <- load_spe_log()
cat("Scanning", nrow(log), "experiments for identity types...\n")

identity_summary <- list()      # 跨实验统计
all_comparisons <- list()       # 所有实验的 Self vs Other 比较

for (i in seq_len(nrow(log))) {
  entry <- log[i, ]
  pk <- entry$Pair_Key
  sp_file <- file.path(SPE_DIR, basename(entry$Output_File))
  if (!file.exists(sp_file)) next

  df <- tryCatch(read.csv(sp_file, stringsAsFactors = FALSE, fileEncoding = "UTF-8"),
                 error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) next

  id_col <- select_identity_column(df, "label")
  if (is.null(id_col)) next

  result <- compute_subject_spe(df, id_col)
  identity_summary[[pk]] <- data.frame(
    pairKey = pk,
    identity_types = paste(result$identity_types, collapse = ", "),
    n_identity_types = length(result$identity_types),
    stringsAsFactors = FALSE
  )

  # 收集所有比较对
  for (other_ident in names(result$comparisons)) {
    comp <- result$comparisons[[other_ident]]
    if (!is.null(comp$spe_rt_d)) {
      all_comparisons[[length(all_comparisons) + 1]] <- data.frame(
        pairKey = pk,
        comparison = paste0("Self vs ", other_ident),
        other_identity = other_ident,
        spe_rt_d = comp$spe_rt_d,
        spe_acc_d = comp$spe_acc_d,
        n_valid = comp$n_valid,
        stringsAsFactors = FALSE
      )
    }
  }
}

identity_df <- bind_rows(identity_summary)
comp_df     <- bind_rows(all_comparisons)

cat("Identity summary:", nrow(identity_df), "experiments\n")
cat("Comparisons:", nrow(comp_df), "pairs\n")

# ===========================================================================
# 2. Identity 类型频率条形图
# ===========================================================================
cat("\n--- Identity Type Frequency ---\n")

# 展开所有 identity_types
id_counts <- table(unlist(strsplit(identity_df$identity_types, ",\\s*")))
id_freq <- data.frame(
  Identity = names(id_counts),
  Count    = as.integer(id_counts),
  stringsAsFactors = FALSE
)
id_freq <- id_freq[order(id_freq$Count, decreasing = TRUE), ]
cat("Identity frequencies:\n"); print(id_freq)

colors_ident <- c(
  "Self"       = "#ff9800",
  "Stranger"   = "#2196f3",
  "Close"      = "#4caf50",
  "Friend"     = "#81c784",
  "Other"      = "#9c27b0",
  "NonPerson"  = "#607d8b",
  "Celebrity"  = "#e91e63",
  "Acquaintance" = "#00bcd4",
  "You"        = "#ff5722",
  "NA"         = "#9e9e9e"
)

p_id_freq <- ggplot(id_freq, aes(x = reorder(Identity, Count), y = Count, fill = Identity)) +
  geom_col(width = 0.7, alpha = 0.85) +
  geom_text(aes(label = Count), hjust = -0.2, size = 3.5) +
  scale_fill_manual(values = colors_ident) +
  coord_flip() +
  labs(title = paste0("Identity Types Across ", nrow(identity_df), " SPE Experiments"),
       subtitle = "Label_Standardized_Identity frequency",
       x = NULL, y = "Number of Experiments") +
  theme_spe() +
  theme(legend.position = "none")
save_plot_png(p_id_freq, file.path(OUT_DIR, "SPE_Identity_Frequency.png"))

# ===========================================================================
# 3. 森林图：Self vs 每种 Identity 的 SPE (所有实验聚合)
# ===========================================================================
cat("\n--- Forest Plot: Self vs Other Identity SPE ---\n")

forest_data <- comp_df %>%
  filter(!is.na(spe_rt_d)) %>%
  group_by(other_identity) %>%
  summarise(
    N_experiments = n(),
    mean_spe  = mean(spe_rt_d, na.rm = TRUE),
    sd_spe    = sd(spe_rt_d, na.rm = TRUE),
    se_spe    = sd_spe / sqrt(N_experiments),
    mean_acc  = mean(spe_acc_d, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_spe))

cat("Forest data:\n"); print(forest_data)

p_forest <- ggplot(forest_data, aes(x = mean_spe, y = reorder(other_identity, mean_spe), color = other_identity)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = mean_spe - se_spe, xmax = mean_spe + se_spe), height = 0.2, linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  scale_color_manual(values = colors_ident, guide = "none") +
  labs(
    title = "SPE (RT Cohen's d): Self vs Other Identity",
    subtitle = paste0("Aggregated across ", length(unique(comp_df$pairKey)), " experiments"),
    x = expression("SPE RT (Cohen's d) (Other − Self)"), y = NULL
  ) +
  theme_spe() +
  theme(legend.position = "none")
save_plot_png(p_forest, file.path(OUT_DIR, "SPE_Identity_Forest_Plot.png"))

# ===========================================================================
# 4. 箱线图：每种比较对的 SPE 分布
# ===========================================================================
cat("\n--- Boxplot: SPE by Identity Pair ---\n")
comp_df_valid <- comp_df[!is.na(comp_df$spe_rt_d), ]

p_box <- ggplot(comp_df_valid, aes(x = reorder(other_identity, spe_rt_d, median), y = spe_rt_d, fill = other_identity)) +
  geom_boxplot(outlier.size = 1, alpha = 0.7, width = 0.6) +
  geom_jitter(width = 0.15, alpha = 0.3, size = 1, color = "grey40") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  scale_fill_manual(values = colors_ident) +
  labs(
    title = "SPE Distribution by Identity Comparison Pair",
    subtitle = "Each point = one experiment | All conditions",
    x = NULL, y = "SPE RT (Cohen's d)"
  ) +
  theme_spe() +
  theme(legend.position = "none")
save_plot_png(p_box, file.path(OUT_DIR, "SPE_Identity_Boxplot.png"))

# ===========================================================================
# 5. 多 Identity 实验的详情 (选取前 3 个)
# ===========================================================================
cat("\n--- Multi-Identity Experiment Detail ---\n")
multi_id_exps <- identity_df[identity_df$n_identity_types >= 3, ]
cat("Experiments with 3+ identity types:", nrow(multi_id_exps), "\n")

for (pk in head(multi_id_exps$pairKey, 3)) {
  cat("  Processing:", pk, "\n")
  exp_comps <- comp_df[comp_df$pairKey == pk, ]
  if (nrow(exp_comps) < 2) next

  p_bars <- ggplot(exp_comps, aes(x = reorder(other_identity, spe_rt_d), y = spe_rt_d, fill = other_identity)) +
    geom_col(width = 0.6, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.3f", spe_rt_d)),
              vjust = ifelse(exp_comps$spe_rt_d >= 0, -0.3, 1.3), size = 3) +
    geom_hline(yintercept = 0, color = "grey40", linewidth = 0.4) +
    scale_fill_manual(values = colors_ident) +
    labs(title = paste0("Multi-Identity SPE — ", pk),
         subtitle = "SPE(RT) = Other − Self (Cohen's d)",
         x = NULL, y = "SPE RT (Cohen's d)") +
    theme_spe() +
    theme(legend.position = "none")

  save_plot_png(p_bars,
                file.path(OUT_DIR, paste0("SPE_Identity_", gsub("[^A-Za-z0-9_]", "_", pk), ".png")))
}

# ===========================================================================
# 6. 合并对比图
# ===========================================================================
p_combined <- cowplot::plot_grid(
  p_id_freq, p_forest,
  ncol = 2, labels = c("A", "B"), label_size = 11
)
save_plot_png(p_combined, file.path(OUT_DIR, "SPE_Identity_Combined.png"),
              width = 14, height = 6)

cat("\n=== SPE Identity Comparison DONE ===\n")
