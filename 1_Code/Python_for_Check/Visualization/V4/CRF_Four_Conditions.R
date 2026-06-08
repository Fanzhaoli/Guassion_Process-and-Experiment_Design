###############################################################################
# 四条件 CRF 线图 — Overall Average across All Subjects
# 四个条件: Matching×Self, Matching×Stranger, NonMatching×Self, NonMatching×Stranger
# 输出: PDF 图表文件
###############################################################################

# ===========================================================================
# 0. 环境
# ===========================================================================
required_packages <- c("ggplot2", "dplyr", "tidyr", "readr", "cowplot")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = "https://cran.r-project.org")
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}
if (.Platform$OS.type == "windows") tryCatch({ windowsFonts(yahei = windowsFont("Microsoft YaHei")) }, error = function(e) {})
pdf_family <- "sans"

# ===========================================================================
# 1. 路径
# ===========================================================================
RAW_DIR <- file.path("D:", "GitHub_programe", "GitHub", "Guassion-Process-Experiment-Design",
                      "2_Data", "Real_Data", "UnExtact", "raw")
OUT_DIR <- file.path("D:", "GitHub_programe", "GitHub", "Guassion-Process-Experiment-Design",
                      "1_Code", "Python_for_Check", "Visualization", "V4", "outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

PDF_PATH <- file.path(OUT_DIR, "CRF_Four_Conditions_Overall.pdf")

# ===========================================================================
# 2. 实验规则 (100% app_server.py)
# ===========================================================================
get_match_key    <- function(sid) c("f","j","j","f")[((sid-1)%%4)+1]
get_correct_order <- function(sid) if (sid%%2==0) list(square="self",circle="stranger") else list(square="stranger",circle="self")
compute_condition <- function(shape, label, sid) {
  ifelse(label == get_correct_order(sid)[[shape]], "Matching", "NonMatching")
}

# ===========================================================================
# 3. 数据加载
# ===========================================================================
parse_file_ids <- function(fname) {
  stem <- sub("\\.csv$","", fname)
  parts <- strsplit(sub("^EXP_data_group","", stem), "_")[[1]]
  list(groupID=as.integer(parts[1]), subjectID=as.integer(parts[2]))
}

load_one_file <- function(fp) {
  fname <- basename(fp)
  ids <- parse_file_ids(fname)
  sid_ref <- ids$subjectID
  df <- read.csv(fp, stringsAsFactors=FALSE, na.strings=c("NA","nan","NaN",""))
  df$groupID    <- as.integer(df$groupID)
  df$subjectID  <- as.integer(df$subjectID)
  df$trialID    <- as.integer(df$trialID)
  df$Shape      <- trimws(tolower(as.character(df$Shape)))
  df$Label      <- trimws(tolower(as.character(df$Label)))
  df$Response   <- trimws(tolower(as.character(df$Response)))
  df$CorrectKey <- trimws(tolower(as.character(df$CorrectKey)))
  df$stage      <- ifelse(is.na(df$stage)|df$stage=="", "formal", as.character(df$stage))
  df$RT         <- suppressWarnings(as.numeric(df$RT))
  df$Correct    <- suppressWarnings(as.numeric(df$Correct))
  df$responded  <- !is.na(df$RT) & !is.na(df$Response) & !(df$Response %in% c("na","nan",""))
  df$Condition  <- sapply(seq_len(nrow(df)), function(i) compute_condition(df$Shape[i], df$Label[i], sid_ref))
  df$Identity   <- ifelse(df$Label=="self", "Self", "Stranger")
  df$MatchKey   <- get_match_key(sid_ref)
  df$ResponseIsMatch <- ifelse(df$responded, as.integer(df$Response==df$MatchKey), NA_integer_)
  df$P   <- suppressWarnings(as.numeric(df$P))
  df$T   <- suppressWarnings(as.numeric(df$T))
  df$W   <- suppressWarnings(as.numeric(df$W))
  df$T_ms  <- df$T*1000; df$W_ms <- df$W*1000; df$M_ms <- df$T_ms+df$W_ms; df$RT_ms <- df$RT*1000
  df$source_file <- fname
  return(df)
}

load_all <- function() {
  files <- sort(list.files(RAW_DIR, pattern="EXP_data_group.*\\.csv$", full.names=TRUE))
  all_df <- do.call(rbind, lapply(files, load_one_file))
  rownames(all_df) <- NULL
  message("Loaded ", nrow(all_df), " trials from ", length(files), " files")
  all_df
}

# ===========================================================================
# 4. CRF 计算 — 按 Condition × Identity 四条件
# ===========================================================================

#' 对单一条件子集计算 CRF
compute_crf_subset <- function(trials, n_quantiles = 5) {
  d <- trials[trials$stage=="formal" & trials$responded & !is.na(trials$RT), ]
  if (nrow(d) < n_quantiles * 2) return(data.frame())
  d <- d[order(d$RT), ]
  n_total <- nrow(d)
  q_size  <- floor(n_total / n_quantiles)
  bins <- lapply(seq_len(n_quantiles), function(i) {
    start <- (i-1)*q_size + 1
    end   <- if(i==n_quantiles) n_total else start + q_size - 1
    b <- d[start:end, ]
    p <- mean(as.numeric(b$ResponseIsMatch), na.rm=TRUE)
    n_b <- nrow(b)
    sd_val <- if(n_b>1) sd(as.numeric(b$ResponseIsMatch), na.rm=TRUE) else 0
    data.frame(bin=i, n=n_b, rt_mean_ms=mean(b$RT_ms,na.rm=TRUE),
               upper_prop=p, sem=if(n_b>1) sd_val/sqrt(n_b) else 0,
               stringsAsFactors=FALSE)
  })
  do.call(rbind, bins)
}

#' 为每个被试计算四条件 CRF, 然后跨被试聚合
compute_four_condition_crf <- function(all_df, n_quantiles = 5) {
  subjects <- all_df %>%
    group_by(groupID, subjectID) %>%
    group_keys()

  conditions <- c("Matching", "NonMatching")
  identities <- c("Self", "Stranger")
  all_bins <- list()

  for (i in seq_len(nrow(subjects))) {
    gid <- subjects$groupID[i]
    sid <- subjects$subjectID[i]
    subj <- all_df[all_df$groupID==gid & all_df$subjectID==sid, ]
    for (cond in conditions) {
      for (ident in identities) {
        subset_trials <- subj[subj$Condition==cond & subj$Identity==ident, ]
        crf <- compute_crf_subset(subset_trials, n_quantiles)
        if (nrow(crf) > 0) {
          crf$Condition   <- cond
          crf$Identity    <- ident
          crf$groupID     <- gid
          crf$subjectID   <- sid
          all_bins[[length(all_bins)+1]] <- crf
        }
      }
    }
  }

  if (length(all_bins) == 0) return(data.frame())

  all_crf <- do.call(rbind, all_bins)

  # 跨被试聚合: 按 Condition × Identity × bin 平均
  overall_crf <- all_crf %>%
    group_by(Condition, Identity, bin) %>%
    summarise(
      rt_mean_ms  = mean(rt_mean_ms, na.rm=TRUE),
      upper_prop  = mean(upper_prop, na.rm=TRUE),
      sem         = sd(upper_prop, na.rm=TRUE) / sqrt(n()),
      n_subjects  = n(),
      .groups     = "drop"
    )
  overall_crf
}

# ===========================================================================
# 5. 绘图 — 四条件四条线
# ===========================================================================

plot_four_condition_crf <- function(overall_crf, n_quantiles = 5) {

  # 创建组合条件列用于颜色/线型区分
  overall_crf$ConditionIdentity <- paste(overall_crf$Condition, overall_crf$Identity, sep="\n")

  # 定义四条件的颜色和线型
  # Matching-Self:  橙色实线
  # Matching-Stranger: 橙色虚线
  # NonMatching-Self:  蓝色实线
  # NonMatching-Stranger: 蓝色虚线
  overall_crf$cond_ident <- factor(
    overall_crf$ConditionIdentity,
    levels = c("Matching\nSelf", "Matching\nStranger",
               "NonMatching\nSelf", "NonMatching\nStranger")
  )

  # ---- 诊断: 检查 sem 是否有异常值 ----
  cat("  CRF sem range by condition:\n")
  overall_crf %>%
    group_by(Condition, Identity) %>%
    summarise(
      sem_min = min(sem, na.rm = TRUE),
      sem_max = max(sem, na.rm = TRUE),
      sem_na  = sum(is.na(sem)),
      .groups = "drop"
    ) %>%
    as.data.frame() %>%
    print()

  # 过滤 NA sem 的行 (防止 ribbon 报错)
  crf_clean <- overall_crf[!is.na(overall_crf$sem), ]
  if (nrow(crf_clean) < nrow(overall_crf)) {
    cat(sprintf("  WARNING: %d rows with NA sem removed\n",
                nrow(overall_crf) - nrow(crf_clean)))
  }

  color_map <- c("Matching\nSelf"        = "#ff9800",
                 "Matching\nStranger"    = "#ff9800",
                 "NonMatching\nSelf"     = "#2196f3",
                 "NonMatching\nStranger" = "#2196f3")

  linetype_map <- c("Matching\nSelf"        = "solid",
                    "Matching\nStranger"    = "dashed",
                    "NonMatching\nSelf"     = "solid",
                    "NonMatching\nStranger" = "dashed")

  # 构造自定义图例: Condition 颜色 x Identity 线型
  legend_df <- data.frame(
    Condition = rep(c("Matching", "NonMatching"), each = 2),
    Identity  = rep(c("Self", "Stranger"), 2),
    x = 1, y = 1  # dummy position
  )

  # 注意: fill 不放入全局 aes, 用 geom_errorbar 替代 geom_ribbon (更鲁棒)
  p <- ggplot(crf_clean, aes(x = rt_mean_ms, y = upper_prop,
                              color = cond_ident,
                              linetype = cond_ident,
                              group = cond_ident)) +
    # 置信带: 用 errorbar + 半透明粗线模拟 ribbon 效果
    geom_errorbar(aes(ymin = upper_prop - sem, ymax = upper_prop + sem),
                  width = 0, linewidth = 6, alpha = 0.12, show.legend = FALSE) +
    # 折线 + 点
    geom_line(linewidth = 1.1) +
    geom_point(size = 2.8) +
    # 参考线 0.5
    geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray40", linewidth = 0.7) +
    # 颜色: Matching 橙色, NonMatching 蓝色
    scale_color_manual(
      values = color_map,
      labels = c("Matching\nSelf" = "Matching x Self",
                 "Matching\nStranger" = "Matching x Stranger",
                 "NonMatching\nSelf" = "NonMatching x Self",
                 "NonMatching\nStranger" = "NonMatching x Stranger")
    ) +
    # 线型: Self 实线, Stranger 虚线
    scale_linetype_manual(
      values = linetype_map,
      labels = c("Matching\nSelf" = "Matching x Self",
                 "Matching\nStranger" = "Matching x Stranger",
                 "NonMatching\nSelf" = "NonMatching x Self",
                 "NonMatching\nStranger" = "NonMatching x Stranger")
    ) +
    scale_y_continuous(limits = c(-0.05, 1.05), breaks = seq(0, 1, 0.25),
                       expand = c(0, 0)) +
    labs(
      title    = "Overall CRF - Four Conditions across All Subjects",
      subtitle = paste0("N = 88 subjects  |  Quantiles = ", n_quantiles,
                        "  |  Ribbon = +-1 SEM  |  Solid = Self, Dashed = Stranger"),
      x        = "RT Bin Mean (ms)",
      y        = "P(Match Key)",
      color    = "Condition x Identity",
      linetype = "Condition x Identity"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold", size = 16, color = "#1a237e"),
      plot.subtitle    = element_text(size = 10, color = "gray40"),
      axis.title       = element_text(size = 12, face = "bold"),
      axis.text        = element_text(size = 10),
      legend.position  = "bottom",
      legend.title     = element_text(face = "bold", size = 10),
      legend.text      = element_text(size = 9),
      legend.key.width = unit(2.5, "cm"),
      panel.grid.major = element_line(color = "gray88"),
      panel.grid.minor = element_blank(),
      plot.margin      = margin(15, 20, 10, 10)
    )

  # color + linetype 自动合并为单一图例, 只需一个 override.aes
  p <- p + guides(
    color = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(size = 3))
  )

  return(p)
}

# ===========================================================================
# 6. 附加图: CRF-SPE 分解 (Self-Stranger per Condition)
# ===========================================================================

plot_spe_by_condition <- function(overall_crf, n_quantiles = 5) {

  # 从四条件 CRF 计算 Matching 和 NonMatching 各自的 SPE
  calc_spe <- function(d, cond_val) {
    d_self     <- d[d$Condition==cond_val & d$Identity=="Self", ]
    d_stranger <- d[d$Condition==cond_val & d$Identity=="Stranger", ]
    # 需要按 bin 对齐
    m <- min(nrow(d_self), nrow(d_stranger))
    if (m == 0) return(data.frame())
    data.frame(
      bin            = d_self$bin[1:m],
      rt_mean_ms     = (d_self$rt_mean_ms[1:m] + d_stranger$rt_mean_ms[1:m]) / 2,
      spe_upper_prop = d_self$upper_prop[1:m] - d_stranger$upper_prop[1:m],
      spe_sem        = sqrt(d_self$sem[1:m]^2 + d_stranger$sem[1:m]^2),
      Condition      = cond_val,
      stringsAsFactors = FALSE
    )
  }

  spe_matching    <- calc_spe(overall_crf, "Matching")
  spe_nonmatching <- calc_spe(overall_crf, "NonMatching")

  spe_df <- rbind(spe_matching, spe_nonmatching)
  if (nrow(spe_df) == 0) {
    return(ggplot() + annotate("text", x=1, y=1, label="No SPE data") + theme_void())
  }

  spe_df$Condition <- factor(spe_df$Condition, levels = c("Matching", "NonMatching"))

  # 过滤 NA
  spe_clean <- spe_df[!is.na(spe_df$spe_sem) & !is.na(spe_df$spe_upper_prop), ]

  ggplot(spe_clean, aes(x = rt_mean_ms, y = spe_upper_prop,
                         color = Condition)) +
    geom_errorbar(aes(ymin = spe_upper_prop - 1.96 * spe_sem,
                       ymax = spe_upper_prop + 1.96 * spe_sem),
                  width = 0, linewidth = 6, alpha = 0.12, show.legend = FALSE) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.7) +
    scale_color_manual(values = c("Matching" = "#ff9800", "NonMatching" = "#2196f3")) +
    labs(
      title    = "CRF-SPE Decomposition: Self - Stranger by Condition",
      subtitle = "Positive = Self more likely to press Match key than Stranger",
      x        = "RT Bin Mean (ms)",
      y        = "Self - Stranger P(Match Key)"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold", size = 14, color = "#1a237e"),
      plot.subtitle    = element_text(size = 9, color = "gray40"),
      legend.position  = "top",
      panel.grid.major = element_line(color = "gray88"),
      panel.grid.minor = element_blank()
    )
}

# ===========================================================================
# 7. 主流程
# ===========================================================================
generate_four_condition_crf <- function(n_quantiles = 5) {
  cat("=== Four-Condition CRF Plot Generator ===\n\n")

  cat("[1/3] Loading data...\n")
  all_df <- load_all()

  cat("[2/3] Computing four-condition CRF...\n")
  overall_crf <- compute_four_condition_crf(all_df, n_quantiles)
  cat(sprintf("  Aggregated CRF: %d rows (Condition × Identity × Bin)\n", nrow(overall_crf)))

  # 打印数据摘要
  cat("\n  Per condition summary:\n")
  overall_crf %>%
    group_by(Condition, Identity) %>%
    summarise(
      n_bins   = n(),
      mean_p   = round(mean(upper_prop), 3),
      sd_p     = round(sd(upper_prop), 3),
      n_subs   = first(n_subjects),
      .groups  = "drop"
    ) %>%
    as.data.frame() %>%
    print()

  cat("\n[3/3] Generating PDF...\n")

  pdf(PDF_PATH, width = 11, height = 11, onefile = TRUE,
      title = "Four-Condition CRF", family = pdf_family)

  # 图1: 四条件 CRF 四条线
  p_main <- plot_four_condition_crf(overall_crf, n_quantiles)
  print(p_main)

  # 图2: SPE 分解 (Self-Stranger per Condition)
  p_spe <- plot_spe_by_condition(overall_crf, n_quantiles)
  print(p_spe)

  # 图3: 简化版面 — 大图 CRF + 底部数据表
  p_combined <- cowplot::plot_grid(
    p_main + theme(plot.subtitle = element_text(size = 8)),
    p_spe,
    ncol = 1, rel_heights = c(0.58, 0.42)
  )
  print(p_combined)

  # 图4: 只有大图 CRF, 适合单独展示
  print(p_main)

  dev.off()

  cat("\n  Output:", PDF_PATH, "\n")
  if (file.exists(PDF_PATH)) {
    cat(sprintf("  File size: %.1f MB\n", file.info(PDF_PATH)$size / (1024*1024)))
  }
  cat("  Done.\n")
}

# ===========================================================================
# 8. 入口
# ===========================================================================
if (sys.nframe() == 0) {
  generate_four_condition_crf(n_quantiles = 5)
} else {
  cat("CRF four-condition script loaded. Run generate_four_condition_crf()\n")
}
