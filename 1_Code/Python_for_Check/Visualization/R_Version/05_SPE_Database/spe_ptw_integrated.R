###############################################################################
# 05_SPE_Database/spe_ptw_integrated.R
# 对标: renderSPEPTWIntegrated()
# 生成: P|T|W × SPE 散点图 + 线性回归线 + 相关系数
#       三图联合视图、参数分组摘要
###############################################################################

source(file.path("..", "shared", "utils.R"), chdir = TRUE)
OUT_DIR <- file.path(R_VERSION_DIR, "05_SPE_Database", "outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== 05_SPE_Database — P|T|W Integrated ===\n")

# ===========================================================================
# 1. 读取 SPE overview 数据（从 spe_overview.R 生成或重新计算）
# ===========================================================================
spe_csv <- file.path(OUT_DIR, "spe_overview_data.csv")
if (file.exists(spe_csv)) {
  spe_df <- read.csv(spe_csv, stringsAsFactors = FALSE)
  cat("Loaded existing SPE data:", nrow(spe_df), "rows\n")
} else {
  cat("Please run spe_overview.R first\n")
  quit(save = "no")
}

# ===========================================================================
# 2. 散点图 + 回归线绘制函数
# ===========================================================================
plot_spe_scatter <- function(df, x_var, x_label, color_hex) {
  if (nrow(df) < 3) {
    return(ggplot() + annotate("text", x = 1, y = 1, label = "Insufficient data") + theme_void())
  }

  fit <- linear_fit(df[[x_var]], df$spe_rt_d)
  r_val <- pearson_r(df[[x_var]], df$spe_rt_d)

  # Regression equation annotation
  if (!is.null(fit)) {
    coefs <- coef(fit)
    eqn <- sprintf("y = %.4fx + %.3f\nr = %.3f, N = %d",
                   coefs[2], coefs[1], r_val, nrow(df))
  } else {
    eqn <- sprintf("r = %.3f, N = %d", r_val, nrow(df))
  }

  ggplot(df, aes_string(x = x_var, y = "spe_rt_d")) +
    geom_point(color = color_hex, alpha = 0.6, size = 2.5) +
    geom_smooth(method = "lm", se = TRUE, color = "#f44336",
                linewidth = 1, linetype = "dashed", alpha = 0.15) +
    geom_hline(yintercept = 0, color = "grey60", linewidth = 0.3) +
    annotate("text", x = min(df[[x_var]], na.rm = TRUE),
             y = max(df$spe_rt_d, na.rm = TRUE) * 0.95,
             label = eqn, hjust = 0, vjust = 1, size = 3, color = "grey30") +
    labs(x = x_label, y = "SPE RT (Cohen's d)") +
    theme_spe(base_size = 10)
}

# ===========================================================================
# 3. 三图并排
# ===========================================================================
validP <- spe_df[!is.na(spe_df$spe_rt_d) & !is.na(spe_df$P_ms), ]
validT <- spe_df[!is.na(spe_df$spe_rt_d) & !is.na(spe_df$T_ms), ]
validW <- spe_df[!is.na(spe_df$spe_rt_d) & !is.na(spe_df$W_ms), ]

cat("Valid P:", nrow(validP), " T:", nrow(validT), " W:", nrow(validW), "\n")

p_P <- plot_spe_scatter(validP, "P_ms", "P (Practice Trials)", "#e91e63")
p_T <- plot_spe_scatter(validT, "T_ms", "T (Stimulus ms)", "#9c27b0")
p_W <- plot_spe_scatter(validW, "W_ms", "W (Window ms)", "#2196f3")

p_triple <- cowplot::plot_grid(p_P, p_T, p_W, ncol = 3,
                                labels = c("P", "T", "W"), label_size = 12)
title_gg <- cowplot::ggdraw() +
  cowplot::draw_label("P|T|W Design Space × SPE (RT Cohen's d)", fontface = "bold", size = 13)
p_final <- cowplot::plot_grid(title_gg, p_triple, ncol = 1, rel_heights = c(0.08, 1))
save_plot_png(p_final, file.path(OUT_DIR, "SPE_PTW_Integrated.png"), width = 15, height = 5.5)

# ===========================================================================
# 4. 参数分组摘要（对标前端左侧面板）
# ===========================================================================
cat("\n--- Parameter Group Summaries ---\n")

summarize_group <- function(df, var_name) {
  df %>%
    group_by(!!sym(var_name)) %>%
    summarise(
      N = n(),
      mean_SPE = mean(spe_rt_d, na.rm = TRUE),
      sd_SPE   = sd(spe_rt_d, na.rm = TRUE),
      se_SPE   = sd_SPE / sqrt(N),
      .groups = "drop"
    ) %>%
    arrange(!!sym(var_name))
}

# P 分组
p_groups <- summarize_group(validP, "P_ms")
cat("P groups:\n"); print(p_groups)

# T 分组
t_groups <- summarize_group(validT, "T_ms")
cat("\nT groups:\n"); print(t_groups)

# W 分组
w_groups <- summarize_group(validW, "W_ms")
cat("\nW groups:\n"); print(w_groups)

# 保存分组表
write.csv(p_groups, file.path(OUT_DIR, "spe_P_groups.csv"), row.names = FALSE)
write.csv(t_groups, file.path(OUT_DIR, "spe_T_groups.csv"), row.names = FALSE)
write.csv(w_groups, file.path(OUT_DIR, "spe_W_groups.csv"), row.names = FALSE)

# ===========================================================================
# 5. 分组条形图
# ===========================================================================
plot_group_bars <- function(group_df, var_name, color_hex) {
  var_sym <- sym(var_name)
  ggplot(group_df, aes(x = factor(!!var_sym), y = mean_SPE)) +
    geom_col(fill = color_hex, alpha = 0.7, width = 0.7) +
    geom_errorbar(aes(ymin = mean_SPE - se_SPE, ymax = mean_SPE + se_SPE),
                  width = 0.2, linewidth = 0.6) +
    geom_text(aes(label = paste0("N=", N), y = mean_SPE + se_SPE + 0.02),
              size = 3, vjust = 0) +
    geom_hline(yintercept = 0, color = "grey50", linewidth = 0.3) +
    labs(x = var_name, y = "Mean SPE RT (Cohen's d)") +
    theme_spe(base_size = 10)
}

p_bars <- cowplot::plot_grid(
  plot_group_bars(p_groups, "P_ms", "#e91e63"),
  plot_group_bars(t_groups, "T_ms", "#9c27b0"),
  plot_group_bars(w_groups, "W_ms", "#2196f3"),
  ncol = 3, labels = c("P", "T", "W"), label_size = 12
)
save_plot_png(p_bars, file.path(OUT_DIR, "SPE_PTW_Group_Bars.png"), width = 15, height = 5)

cat("\n=== SPE P|T|W Integrated DONE ===\n")
