###############################################################################
# 02_Data_Browser/data_browser.R
# 对标 app_server.py: list_files(), load_file_data()
# 对标前端: renderDataPreview() — RT 分布直方图、数据预览表格
###############################################################################

# ===========================================================================
# 0. 初始化
# ===========================================================================
source(file.path("..", "shared", "utils.R"), chdir = TRUE)
OUT_DIR <- file.path(R_VERSION_DIR, "02_Data_Browser", "outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ===========================================================================
# 1. 文件列表
# ===========================================================================
cat("=== 02_Data_Browser ===\n")
files <- list_raw_files()
cat("Total files:", nrow(files), "\n")
cat("Files by group:\n")
print(table(files$group))

# ===========================================================================
# 2. 加载全部数据
# ===========================================================================
cat("\nLoading all raw data...\n")
all_data <- load_all_data()
cat("Total rows:", nrow(all_data), "\n")

# 过滤正式试次 (stage == "test" 或 stage != "practice")
if ("stage" %in% names(all_data)) {
  formal <- all_data[grepl("test", all_data$stage, ignore.case = TRUE), ]
} else {
  formal <- all_data  # 假设全部为正式试次
}
cat("Formal trials:", nrow(formal), "\n")

# ===========================================================================
# 3. 按组质量统计（对标前端数据预览表格）
# ===========================================================================
cat("\n--- Group-Level Statistics ---\n")
group_stats <- formal %>%
  mutate(group = as.character(groupID)) %>%
  group_by(group) %>%
  summarise(
    n_subjects = n_distinct(subjectID),
    n_trials   = n(),
    avg_rt     = mean(RT, na.rm = TRUE),
    sd_rt      = sd(RT, na.rm = TRUE),
    acc        = mean(Correct == 1, na.rm = TRUE),
    miss_rate  = mean(is.na(RT) | RT <= 0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    quality = QUALITY_MAP[group],
    label   = sapply(group, function(g) CONDITIONS[[as.character(g)]]$label)
  )
print(group_stats, n = 20)

# 保存统计表格
write.csv(group_stats, file.path(OUT_DIR, "group_statistics.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# ===========================================================================
# 4. RT 分布直方图（按组别）× Matching/NonMatching
# ===========================================================================
cat("\n--- Generating RT Distribution Histograms ---\n")

# 判定 Matching/NonMatching
formal$Matching <- ifelse(
  formal$CorrectKey == formal$Response,
  "Matching", "NonMatching"
)

plot_rt_hist <- function(data, title_str) {
  ggplot(data, aes(x = RT, fill = Matching)) +
    geom_histogram(bins = 30, alpha = 0.6, position = "identity", color = "white", linewidth = 0.2) +
    scale_fill_manual(values = c("Matching" = "#ff9800", "NonMatching" = "#2196f3")) +
    scale_x_continuous(limits = c(0, quantile(data$RT, 0.99, na.rm = TRUE))) +
    labs(title = title_str, x = "RT (ms)", y = "Count", fill = "") +
    theme_spe()
}

# 按组绘制
for (g in sort(unique(formal$groupID))) {
  gdata <- formal[formal$groupID == g, ]
  if (nrow(gdata) < 10) next
  p <- plot_rt_hist(gdata, CONDITIONS[[as.character(g)]]$label)
  save_plot_png(p, file.path(OUT_DIR, paste0("RT_hist_G", g, ".png")))
}

# 全局 RT 直方图
p_all <- plot_rt_hist(formal, "All Groups — RT Distribution (Matching vs NonMatching)")
save_plot_png(p_all, file.path(OUT_DIR, "RT_hist_All_Groups.png"))

# ===========================================================================
# 5. 每个被试个体 RT 分布示例
# ===========================================================================
cat("\n--- Generating Per-Subject RT Examples (first 12 subjects) ---\n")
example_subjects <- unique(formal$subjectID)[1:min(12, length(unique(formal$subjectID)))]

plot_list <- lapply(example_subjects, function(sid) {
  sdata <- formal[formal$subjectID == sid, ]
  if (nrow(sdata) < 10) return(NULL)
  gid <- unique(sdata$groupID)[1]
  ggplot(sdata, aes(x = RT, fill = Matching)) +
    geom_histogram(bins = 20, alpha = 0.6, position = "identity", color = "white", linewidth = 0.2) +
    scale_fill_manual(values = c("Matching" = "#ff9800", "NonMatching" = "#2196f3")) +
    labs(title = paste0("Subj ", sid, " (G", gid, ")"), x = "RT (ms)", y = "Count") +
    theme_spe(base_size = 9) + theme(legend.position = "none")
})

plot_list <- plot_list[!sapply(plot_list, is.null)]
p_subj <- cowplot::plot_grid(plotlist = plot_list, ncol = 3)
save_plot_png(p_subj, file.path(OUT_DIR, "RT_hist_Example_Subjects.png"),
              width = 14, height = 10)

# ===========================================================================
# 6. 正确率 & RT 散点图 × 组别
# ===========================================================================
cat("\n--- ACC vs RT by Group ---\n")
group_summary <- formal %>%
  group_by(groupID, subjectID) %>%
  summarise(
    mean_rt = mean(RT, na.rm = TRUE),
    acc     = mean(Correct == 1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(group = factor(groupID))

p_scatter <- ggplot(group_summary, aes(x = mean_rt, y = acc, color = group)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "Per-Subject Accuracy vs Mean RT (by Group)",
       x = "Mean RT (ms)", y = "Accuracy", color = "Group") +
  theme_spe()
save_plot_png(p_scatter, file.path(OUT_DIR, "ACC_vs_RT_by_Group.png"))

cat("\n=== 02_Data_Browser DONE ===\n")
cat("Outputs saved to:", OUT_DIR, "\n")
