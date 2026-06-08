###############################################################################
# Self-Matching Task 全被试可视化 V4 (R 语言版)
# 生成包含所有被试可视化结果的 PDF 文件
###############################################################################

# ===========================================================================
# 0. 环境配置与包加载
# ===========================================================================
required_packages <- c("ggplot2", "dplyr", "tidyr", "readr", "scales",
                        "cowplot", "RColorBrewer", "grid", "gridExtra")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.r-project.org")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

# Windows 中文字体支持 (不依赖 showtext)
if (.Platform$OS.type == "windows") {
  tryCatch({
    windowsFonts(yahei = windowsFont("Microsoft YaHei"))
  }, error = function(e) {
    message("Microsoft YaHei font not available, using default")
  })
  # 不依赖 showtext, 使用 base graphics 字体设置
  pdf_family <- "sans"
} else {
  pdf_family <- "sans"
}

# ===========================================================================
# 1. 路径配置
# ===========================================================================
PROJECT_ROOT <- file.path("D:", "GitHub_programe", "GitHub",
                           "Guassion-Process-Experiment-Design")
RAW_DIR <- file.path(PROJECT_ROOT, "2_Data", "Real_Data", "UnExtact", "raw")
V4_DIR  <- file.path(PROJECT_ROOT, "1_Code", "Python_for_Check",
                      "Visualization", "V4")
OUT_DIR <- file.path(V4_DIR, "outputs")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

PDF_PATH <- file.path(OUT_DIR, "all_subjects_visualization_V4.pdf")

# ===========================================================================
# 2. 实验参数常量 (来自 app_server.py CONDITIONS)
# ===========================================================================
CONDITIONS <- list(
  "1" = list(P = 0,   T = 0.03, W = 0.3, label = "G1 | P0_T30_W300"),
  "2" = list(P = 0,   T = 0.03, W = 0.6, label = "G2 | P0_T30_W600"),
  "3" = list(P = 120, T = 0.03, W = 0.6, label = "G3 | P120_T30_W600"),
  "4" = list(P = 120, T = 0.08, W = 0.6, label = "G4 | P120_T80_W600"),
  "5" = list(P = 8,   T = 0.10, W = 1.1, label = "G5 | P8_T100_W1100"),
  "6" = list(P = 120, T = 0.50, W = 1.5, label = "G6 | P120_T500_W1500"),
  "7" = list(P = 0,   T = 0.10, W = 1.1, label = "G7 | P0_T100_W1100"),
  "8" = list(P = 120, T = 0.03, W = 0.8, label = "G8 | P120_T30_W800"),
  "9" = list(P = 120, T = 0.08, W = 0.8, label = "G9 | P120_T80_W800")
)

QUALITY_MAP <- c(
  "1" = "exclude", "2" = "exclude", "3" = "caution",
  "4" = "good", "5" = "good", "6" = "good",
  "7" = "good", "8" = "good", "9" = "good"
)

QUALITY_COLORS <- c("good" = "#4caf50", "caution" = "#ff9800", "exclude" = "#f44336")
GROUP_PALETTE <- c("#e91e63", "#2196f3", "#4caf50", "#ff9800",
                    "#9c27b0", "#00bcd4", "#ffeb3b", "#795548", "#607d8b")

# ===========================================================================
# 3. 核心实验规则函数 (100% 复现 app_server.py 逻辑)
# 这些函数与 Python 版本逐行对应
# ===========================================================================

#' 获取被试的配对规则
#' @param subject_id 被试ID
#' @return 配对规则列表
get_pairing_rules <- function(subject_id) {
  mod_result <- subject_id %% 4
  rules <- list(
    "0" = list(square = list(self = "f", stranger = "j"),
                circle = list(self = "j", stranger = "f")),
    "1" = list(square = list(self = "j", stranger = "f"),
                circle = list(self = "f", stranger = "j")),
    "2" = list(square = list(self = "j", stranger = "f"),
                circle = list(self = "f", stranger = "j")),
    "3" = list(square = list(self = "f", stranger = "j"),
                circle = list(self = "j", stranger = "f"))
  )
  rules[[as.character(mod_result)]]
}

#' 获取被试的匹配键
#' @param subject_id 被试ID
#' @return 匹配键字符 ("f" 或 "j")
get_match_key <- function(subject_id) {
  match_keys <- c("f", "j", "j", "f")
  match_keys[((subject_id - 1) %% 4) + 1]
}

#' 获取正确顺序 (shape-label 正确匹配关系)
#' @param subject_id 被试ID
#' @return 正确顺序列表
get_correct_order <- function(subject_id) {
  if (subject_id %% 2 == 0) {
    return(list(square = "self", circle = "stranger"))
  } else {
    return(list(square = "stranger", circle = "self"))
  }
}

#' 计算条件 (Matching 或 NonMatching)
#' @param shape 形状 ("square" 或 "circle")
#' @param label 标签 ("self" 或 "stranger")
#' @param subject_id 被试ID
#' @return "Matching" 或 "NonMatching"
compute_condition <- function(shape, label, subject_id) {
  correct_order <- get_correct_order(subject_id)
  expected_label <- correct_order[[shape]]
  ifelse(label == expected_label, "Matching", "NonMatching")
}

# ===========================================================================
# 4. 数据加载函数
# ===========================================================================

#' 解析文件名中的 groupID 和 subjectID
#' @param filename 文件名
#' @return list(groupID, subjectID)
parse_file_ids <- function(filename) {
  stem <- sub("\\.csv$", "", filename)
  parts <- strsplit(sub("^EXP_data_group", "", stem), "_")[[1]]
  list(groupID = as.integer(parts[1]), subjectID = as.integer(parts[2]))
}

#' 加载单个被试数据文件并执行数据清理
#' 100% 复现 app_server.py load_file_data() 中的逻辑
#' @param filepath CSV文件完整路径
#' @return 清理后的 data.frame
load_one_file <- function(filepath) {
  fname <- basename(filepath)
  ids <- parse_file_ids(fname)
  gid_from_file <- ids$groupID
  sid_from_file <- ids$subjectID

  # 读取 CSV
  df <- read.csv(filepath, stringsAsFactors = FALSE,
                  na.strings = c("NA", "nan", "NaN", ""))

  # 类型转换并清洗 (与 Python 版本完全对应)
  df$groupID    <- as.integer(df$groupID)
  df$subjectID  <- as.integer(df$subjectID)
  df$trialID    <- as.integer(df$trialID)
  df$Shape      <- trimws(tolower(as.character(df$Shape)))
  df$Label      <- trimws(tolower(as.character(df$Label)))
  df$Response   <- trimws(tolower(as.character(df$Response)))
  df$CorrectKey <- trimws(tolower(as.character(df$CorrectKey)))

  # stage 列：缺失值填充为 "formal"
  df$stage <- ifelse(is.na(df$stage) | df$stage == "", "formal",
                      as.character(df$stage))

  # RT 和 Correct 转为数值
  df$RT      <- suppressWarnings(as.numeric(df$RT))
  df$Correct <- suppressWarnings(as.numeric(df$Correct))

  # 判断是否响应: RT 非 NA 且 Response 不是 NA/nan/空
  df$responded <- !is.na(df$RT) &
    !is.na(df$Response) &
    !(df$Response %in% c("na", "nan", ""))

  # ---- 以下逐行对应 app_server.py 第 126-163 行的逻辑 ----
  sid_values <- df$subjectID
  # 注意: 每个文件内部 subjectID 可能相同, 但某些文件可能包含不同的 subjectID
  # 逐行计算 Condition, Identity, MatchKey, ResponseIsMatch

  subject_id_ref <- sid_from_file  # 使用文件名中的 subjectID 作为计算依据

  df$Condition <- sapply(seq_len(nrow(df)), function(i) {
    compute_condition(df$Shape[i], df$Label[i], subject_id_ref)
  })

  df$Identity <- ifelse(df$Label == "self", "Self", "Stranger")

  df$MatchKey <- get_match_key(subject_id_ref)

  df$ResponseIsMatch <- ifelse(df$responded,
                                as.integer(df$Response == df$MatchKey),
                                NA_integer_)

  # P/T/W 数值化
  df$P <- suppressWarnings(as.numeric(df$P))
  df$T <- suppressWarnings(as.numeric(df$T))
  df$W <- suppressWarnings(as.numeric(df$W))

  # 派生毫秒单位列
  df$T_ms  <- df$T * 1000
  df$W_ms  <- df$W * 1000
  df$M_ms  <- df$T_ms + df$W_ms
  df$RT_ms <- df$RT * 1000

  # 质量标签 (标量赋值，避免向量化的 is.na 问题)
  qual_val <- QUALITY_MAP[as.character(gid_from_file)]
  if (is.na(qual_val)) qual_val <- "unknown"
  df$quality <- qual_val

  # 源文件信息
  df$source_file <- fname
  df$groupID_from_file    <- gid_from_file
  df$subjectID_from_file  <- sid_from_file

  return(df)
}

#' 加载所有被试数据
#' @param raw_dir 原始数据目录
#' @return 合并后的完整 data.frame
load_all_raw <- function(raw_dir = RAW_DIR) {
  files <- sort(list.files(raw_dir, pattern = "EXP_data_group.*\\.csv$",
                            full.names = TRUE))
  if (length(files) == 0) {
    stop("No EXP_data_group*.csv found in ", raw_dir)
  }

  message("Loading ", length(files), " files...")
  all_dfs <- lapply(files, load_one_file)
  all_df <- do.call(rbind, all_dfs)
  rownames(all_df) <- NULL
  message("Total rows loaded: ", nrow(all_df))
  return(all_df)
}

# ===========================================================================
# 5. 汇总统计函数
# ===========================================================================

#' 单个被试的条件汇总表
#' @param df 被试数据
#' @return 汇总 data.frame
subject_summary_table <- function(df) {
  f <- df[df$stage == "formal", ]
  if (nrow(f) == 0) return(data.frame())

  # 按 Identity × Condition 分组
  groupings <- expand.grid(
    Identity  = c("Self", "Stranger"),
    Condition = c("Matching", "NonMatching"),
    stringsAsFactors = FALSE
  )

  results <- lapply(seq_len(nrow(groupings)), function(i) {
    ident <- groupings$Identity[i]
    cond  <- groupings$Condition[i]
    g <- f[f$Identity == ident & f$Condition == cond, ]
    n_trials <- nrow(g)
    rr <- g[g$responded, ]
    n_resp <- nrow(rr)
    omission_rate <- if (n_trials > 0) 1 - n_resp / n_trials else NA
    cc <- rr[!is.na(rr$Correct) & rr$Correct == 1, ]
    data.frame(
      Identity       = ident,
      Condition      = cond,
      n_trials       = n_trials,
      n_resp         = n_resp,
      omission_rate  = omission_rate,
      accuracy       = if (n_resp > 0) mean(rr$Correct, na.rm = TRUE) else NA,
      rt_mean_ms     = if (n_resp > 0) mean(rr$RT_ms, na.rm = TRUE) else NA,
      rt_median_ms   = if (n_resp > 0) median(rr$RT_ms, na.rm = TRUE) else NA,
      correct_rt_mean_ms = if (nrow(cc) > 0) mean(cc$RT_ms, na.rm = TRUE) else NA,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, results)
}

#' 核密度估计的高斯曲线
#' @param x 数值向量
#' @param n 点数
#' @return data.frame(x, y)
gaussian_kde <- function(x, n = 100) {
  if (length(x) < 2) return(data.frame(x = numeric(0), y = numeric(0)))
  d <- density(x, n = n, na.rm = TRUE)
  data.frame(x = d$x, y = d$y)
}

#' 计算 CRF (Conditional Response Function)
#' 复现 Python compute_crf 逻辑
#' @param trials 被试 trial 数据
#' @param n_quantiles 分位数数量
#' @return CRF data.frame
compute_crf <- function(trials, n_quantiles = 5) {
  d <- trials[
    trials$stage == "formal" &
    trials$responded &
    !is.na(trials$RT),
  ]
  if (nrow(d) < n_quantiles * 2) return(data.frame())

  d <- d[order(d$RT), ]
  n_total <- nrow(d)
  q_size <- floor(n_total / n_quantiles)

  bins <- lapply(seq_len(n_quantiles), function(i) {
    start <- (i - 1) * q_size + 1
    end   <- if (i == n_quantiles) n_total else start + q_size - 1
    b <- d[start:end, ]
    p <- mean(as.numeric(b$ResponseIsMatch), na.rm = TRUE)
    n_b <- nrow(b)
    sd_val <- if (n_b > 1) sd(as.numeric(b$ResponseIsMatch), na.rm = TRUE) else 0
    data.frame(
      bin         = i,
      n           = n_b,
      rt_mean     = mean(b$RT, na.rm = TRUE),
      rt_mean_ms  = mean(b$RT_ms, na.rm = TRUE),
      upper_prop  = p,
      sem         = if (n_b > 1) sd_val / sqrt(n_b) else 0,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, bins)
}

#' 计算 CRF-SPE (Self-Priority Effect from CRF)
#' @param trials 被试所有 trial
#' @param n_quantiles 分位数数量
#' @return list(crf_self, crf_stranger, spe_curve)
compute_spe_crf <- function(trials, n_quantiles = 5) {
  crf_s  <- compute_crf(trials[trials$Identity == "Self", ], n_quantiles)
  crf_st <- compute_crf(trials[trials$Identity == "Stranger", ], n_quantiles)

  m <- min(nrow(crf_s), nrow(crf_st))
  if (m == 0) {
    return(list(crf_self = crf_s, crf_stranger = crf_st, spe_curve = data.frame()))
  }

  spe_curve <- data.frame(
    bin            = seq_len(m),
    rt_mean_ms     = (crf_s$rt_mean_ms[1:m] + crf_st$rt_mean_ms[1:m]) / 2,
    spe_upper_prop = crf_s$upper_prop[1:m] - crf_st$upper_prop[1:m],
    spe_sem        = sqrt(crf_s$sem[1:m]^2 + crf_st$sem[1:m]^2),
    stringsAsFactors = FALSE
  )
  list(crf_self = crf_s, crf_stranger = crf_st, spe_curve = spe_curve)
}

#' 计算单个被试的汇总指标
#' @param g 被试数据 (按 groupID + subjectID 分组后的子集)
#' @return 单行 data.frame
summarize_subject <- function(g) {
  formal <- g[g$stage == "formal", ]
  responded <- formal[formal$responded, ]
  correct <- responded[!is.na(responded$Correct) & responded$Correct == 1, ]

  n_formal <- nrow(formal)
  n_resp   <- nrow(responded)

  # Identity 分组统计
  acc_by_id <- if (n_resp > 0) {
    resp_by_id <- split(responded, responded$Identity)
    cor_by_id  <- split(correct, correct$Identity)
    sapply(c("Self", "Stranger"), function(id) {
      r <- resp_by_id[[id]]
      c <- cor_by_id[[id]]
      if (is.null(r) || nrow(r) == 0) return(NA_real_)
      if (is.null(c)) return(0)
      nrow(c) / nrow(r)
    })
  } else {
    c(Self = NA_real_, Stranger = NA_real_)
  }

  rt_by_id <- if (n_resp > 0) {
    sapply(c("Self", "Stranger"), function(id) {
      sub <- responded[responded$Identity == id, ]
      if (nrow(sub) == 0) return(NA_real_)
      mean(sub$RT_ms, na.rm = TRUE)
    })
  } else {
    c(Self = NA_real_, Stranger = NA_real_)
  }

  data.frame(
    groupID       = g$groupID[1],
    subjectID     = g$subjectID[1],
    quality       = g$quality[1],
    P             = g$P[1],
    T_ms          = g$T_ms[1],
    W_ms          = g$W_ms[1],
    M_ms          = g$M_ms[1],
    n_resp        = n_resp,
    n_formal      = n_formal,
    omission_rate = if (n_formal > 0) 1 - n_resp / n_formal else NA,
    accuracy      = if (n_resp > 0) mean(responded$Correct, na.rm = TRUE) else NA,
    acc_self      = unname(acc_by_id["Self"]),
    acc_stranger  = unname(acc_by_id["Stranger"]),
    SPE_ACC       = unname(acc_by_id["Self"] - acc_by_id["Stranger"]),
    rt_self       = unname(rt_by_id["Self"]),
    rt_stranger   = unname(rt_by_id["Stranger"]),
    SPE_RT_ms     = unname(rt_by_id["Stranger"] - rt_by_id["Self"]),
    stringsAsFactors = FALSE
  )
}

# ===========================================================================
# 6. 个体被试图表绘制函数
# ===========================================================================

# 颜色常量
COL_SELF     <- "#ff9800"
COL_STRANGER <- "#2196f3"
COL_SPE      <- "#9c27b0"
COL_ACC      <- "#4caf50"
COL_RT       <- "#64b5f6"
COL_OMISSION <- "#f44336"

#' Chart 1: RT 时序散点图
plot_rt_timeseries <- function(responded, title_prefix) {
  if (nrow(responded) == 0) {
    return(ggplot() + annotate("text", x = 1, y = 1, label = "No response data") +
             theme_void())
  }

  ggplot(responded, aes(x = trialID, y = RT_ms, color = Identity)) +
    geom_point(size = 2.5, alpha = 0.7) +
    scale_color_manual(values = c("Self" = COL_SELF, "Stranger" = COL_STRANGER)) +
    labs(title = paste0(title_prefix, ": RT Timeseries"),
         x = "Trial ID", y = "RT (ms)") +
    theme_minimal(base_size = 9) +
    theme(legend.position = "top",
          plot.title = element_text(face = "bold", size = 10),
          panel.grid.major = element_line(color = "gray85"),
          panel.grid.minor = element_blank())
}

#' Chart 2: RT 分布直方图 + 密度曲线
plot_rt_histogram <- function(responded, title_prefix) {
  if (nrow(responded) == 0) {
    return(ggplot() + annotate("text", x = 1, y = 1, label = "No data") +
             theme_void())
  }

  # 兼容 ggplot2 新旧版本: after_stat(density) vs ..density..
  p <- ggplot(responded, aes(x = RT_ms, fill = Identity, color = Identity))
  # 尝试新版语法, 失败则用旧版
  p <- tryCatch(
    p + geom_histogram(aes(y = after_stat(density)), bins = 28, alpha = 0.35,
                       position = "identity", boundary = 0),
    error = function(e) p + geom_histogram(aes(y = ..density..), bins = 28,
                                            alpha = 0.35, position = "identity")
  )
  p <- p + geom_density(alpha = 0.2, linewidth = 1) +
    facet_wrap(~ Condition, ncol = 2, scales = "free_y") +
    scale_fill_manual(values = c("Self" = COL_SELF, "Stranger" = COL_STRANGER)) +
    scale_color_manual(values = c("Self" = COL_SELF, "Stranger" = COL_STRANGER)) +
    labs(title = paste0(title_prefix, ": RT Distribution"),
         x = "RT (ms)", y = "Density") +
    theme_minimal(base_size = 9) +
    theme(legend.position = "top",
          plot.title = element_text(face = "bold", size = 10),
          strip.text = element_text(face = "bold", size = 9),
          panel.grid.major = element_line(color = "gray85"),
          panel.grid.minor = element_blank())

  return(p)
}

#' Chart 3: 条件分解柱状图
plot_condition_bars <- function(subj_summary, title_prefix) {
  if (nrow(subj_summary) == 0) {
    return(ggplot() + annotate("text", x = 1, y = 1, label = "No summary") +
             theme_void())
  }

  df <- subj_summary
  df$cell <- paste0(substr(df$Condition, 1, 4), "\n", substr(df$Identity, 1, 3))
  df$accuracy_pct    <- df$accuracy * 100
  df$omission_pct    <- df$omission_rate * 100

  # ACC 图
  p1 <- ggplot(df, aes(x = cell, y = accuracy_pct)) +
    geom_col(fill = COL_ACC, alpha = 0.85, color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.1f", accuracy_pct)),
              vjust = -0.5, size = 2.8, fontface = "bold") +
    labs(y = "Accuracy (%)") +
    expand_limits(y = max(df$accuracy_pct, na.rm = TRUE) * 1.2) +
    theme_minimal(base_size = 8) +
    theme(axis.title.x = element_blank(),
          panel.grid.major.x = element_blank())

  # RT 图
  p2 <- ggplot(df, aes(x = cell, y = rt_mean_ms)) +
    geom_col(fill = COL_RT, alpha = 0.85, color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.0f", rt_mean_ms)),
              vjust = -0.5, size = 2.8, fontface = "bold") +
    labs(y = "RT Mean (ms)") +
    expand_limits(y = max(df$rt_mean_ms, na.rm = TRUE) * 1.2) +
    theme_minimal(base_size = 8) +
    theme(axis.title.x = element_blank(),
          panel.grid.major.x = element_blank())

  # Omission 图
  p3 <- ggplot(df, aes(x = cell, y = omission_pct)) +
    geom_col(fill = COL_OMISSION, alpha = 0.85, color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.1f", omission_pct)),
              vjust = -0.5, size = 2.8, fontface = "bold") +
    labs(y = "Omission (%)") +
    expand_limits(y = max(df$omission_pct, na.rm = TRUE) * 1.2) +
    theme_minimal(base_size = 8) +
    theme(axis.title.x = element_blank(),
          panel.grid.major.x = element_blank())

  # 组合三个图，使用共享标题
  combined <- cowplot::plot_grid(p1, p2, p3, ncol = 3, align = "h", axis = "bt")
  title_gg <- cowplot::ggdraw() +
    cowplot::draw_label(paste0(title_prefix, ": Conditions"),
                         fontface = "bold", size = 10)
  cowplot::plot_grid(title_gg, combined, ncol = 1, rel_heights = c(0.12, 0.88))
}

#' Chart 4: CRF + SPE 曲线
plot_crf_spe <- function(crf_s, crf_st, spe_curve, title_prefix) {
  # CRF 面板
  p_crf <- ggplot() +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray50", linewidth = 0.6) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.5, ymax = 1,
             fill = COL_ACC, alpha = 0.05) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = 0.5,
             fill = COL_OMISSION, alpha = 0.05)

  if (nrow(crf_s) > 0) {
    p_crf <- p_crf +
      geom_errorbar(data = crf_s,
                     aes(x = rt_mean_ms, ymin = upper_prop - sem,
                         ymax = upper_prop + sem),
                     color = COL_SELF, width = 0, linewidth = 0.3) +
      geom_point(data = crf_s, aes(x = rt_mean_ms, y = upper_prop),
                 color = COL_SELF, size = 2.5) +
      geom_line(data = crf_s, aes(x = rt_mean_ms, y = upper_prop),
                color = COL_SELF, linewidth = 1)
  }
  if (nrow(crf_st) > 0) {
    p_crf <- p_crf +
      geom_errorbar(data = crf_st,
                     aes(x = rt_mean_ms, ymin = upper_prop - sem,
                         ymax = upper_prop + sem),
                     color = COL_STRANGER, width = 0, linewidth = 0.3) +
      geom_point(data = crf_st, aes(x = rt_mean_ms, y = upper_prop),
                 color = COL_STRANGER, size = 2.5) +
      geom_line(data = crf_st, aes(x = rt_mean_ms, y = upper_prop),
                color = COL_STRANGER, linewidth = 1)
  }

  p_crf <- p_crf +
    scale_y_continuous(limits = c(-0.05, 1.05)) +
    labs(x = "RT bin mean (ms)", y = "P(Match Key)", title = "CRF") +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold", size = 9),
          panel.grid.major = element_line(color = "gray85"),
          panel.grid.minor = element_blank())

  # 手动添加图例
  # 使用 annotate 或直接在数据中添加 group 列
  if (nrow(crf_s) > 0 && nrow(crf_st) > 0) {
    legend_data <- data.frame(
      x = c(Inf, Inf), y = c(Inf, Inf),
      Identity = c("Self", "Stranger")
    )
    p_crf <- p_crf +
      geom_point(data = data.frame(x = 0, y = 0), aes(x = x, y = y, color = "Self"),
                 alpha = 0) +
      geom_point(data = data.frame(x = 0, y = 0), aes(x = x, y = y, color = "Stranger"),
                 alpha = 0) +
      scale_color_manual(name = "", values = c("Self" = COL_SELF, "Stranger" = COL_STRANGER)) +
      theme(legend.position = "top")
  }

  # SPE 面板
  p_spe <- ggplot() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.6)

  if (nrow(spe_curve) > 0) {
    p_spe <- p_spe +
      geom_ribbon(data = spe_curve,
                   aes(x = rt_mean_ms,
                       ymin = spe_upper_prop - 1.96 * spe_sem,
                       ymax = spe_upper_prop + 1.96 * spe_sem),
                   fill = COL_SPE, alpha = 0.15) +
      geom_errorbar(data = spe_curve,
                     aes(x = rt_mean_ms,
                         ymin = spe_upper_prop - 1.96 * spe_sem,
                         ymax = spe_upper_prop + 1.96 * spe_sem),
                     color = COL_SPE, width = 0, linewidth = 0.3) +
      geom_point(data = spe_curve, aes(x = rt_mean_ms, y = spe_upper_prop),
                 color = COL_SPE, size = 2.5) +
      geom_line(data = spe_curve, aes(x = rt_mean_ms, y = spe_upper_prop),
                color = COL_SPE, linewidth = 1)
  }

  p_spe <- p_spe +
    labs(x = "RT bin mean (ms)", y = "Self - Stranger", title = "CRF-SPE") +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold", size = 9),
          panel.grid.major = element_line(color = "gray85"),
          panel.grid.minor = element_blank())

  # 组合
  combined <- cowplot::plot_grid(p_crf, p_spe, ncol = 2, align = "h")
  title_gg <- cowplot::ggdraw() +
    cowplot::draw_label(paste0(title_prefix, ": CRF & SPE"),
                         fontface = "bold", size = 10)
  cowplot::plot_grid(title_gg, combined, ncol = 1, rel_heights = c(0.1, 0.9))
}

#' Chart 5: Response 键偏好热力图
#' @param formal 所有正式试次 (含未响应), 用于正确计算 miss
#' @param match_key 被试的匹配键
#' @param title_prefix 标题前缀
plot_response_heatmap <- function(formal, match_key, title_prefix) {
  if (nrow(formal) == 0) {
    return(ggplot() + annotate("text", x = 1, y = 1, label = "No data") +
             theme_void())
  }

  # 构建频次矩阵
  conditions  <- c("Matching", "NonMatching")
  identities  <- c("Self", "Stranger")
  keys        <- c("f", "j")

  heat_data <- expand.grid(
    Condition = conditions,
    Identity  = identities,
    stringsAsFactors = FALSE
  )

  heat_data$miss <- NA_integer_
  for (k in keys) {
    heat_data[[k]] <- NA_integer_
  }

  for (i in seq_len(nrow(heat_data))) {
    cond  <- heat_data$Condition[i]
    ident <- heat_data$Identity[i]
    # 该条件下的所有正式试次
    sub_all <- formal[formal$Condition == cond & formal$Identity == ident, ]
    # 其中已响应的试次
    sub_resp <- sub_all[sub_all$responded, ]

    n_all  <- nrow(sub_all)
    n_resp <- nrow(sub_resp)
    # miss = 未响应试次数
    heat_data$miss[i] <- n_all - n_resp
    for (k in keys) {
      heat_data[[k]][i] <- sum(sub_resp$Response == k, na.rm = TRUE)
    }
  }

  heat_data$row_label <- paste0(
    substr(heat_data$Condition, 1, 4), "-",
    substr(heat_data$Identity, 1, 3)
  )

  # 转换为长格式
  heat_long <- tidyr::pivot_longer(
    heat_data,
    cols = c("f", "j", "miss"),
    names_to = "Key",
    values_to = "Count"
  )
  heat_long$Key <- factor(heat_long$Key, levels = c("f", "j", "miss"))
  heat_long$row_label <- factor(heat_long$row_label,
                                 levels = rev(unique(heat_data$row_label)))

  ggplot(heat_long, aes(x = Key, y = row_label, fill = Count)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = Count), size = 3.5,
              color = ifelse(heat_long$Count > max(heat_long$Count, na.rm = TRUE) * 0.5,
                             "white", "black")) +
    scale_fill_gradient(low = "#fff9c4", high = "#e65100", na.value = "gray90") +
    labs(title = paste0(title_prefix, "\nMatchKey = ", toupper(match_key)),
         x = "", y = "") +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold", size = 9, hjust = 0.5),
          panel.grid = element_blank(),
          legend.position = "right",
          legend.title = element_text(size = 7))
}

# ===========================================================================
# 7. 群体汇总图表函数
# ===========================================================================

#' Group 汇总: SPE 小提琴图/箱线图
plot_spe_by_group <- function(subject_level) {
  sl <- subject_level
  sl$groupID <- factor(sl$groupID)

  # SPE Accuracy
  p1 <- ggplot(sl, aes(x = groupID, y = SPE_ACC)) +
    geom_violin(fill = "#d7e8ff", alpha = 0.5, draw_quantiles = 0.5) +
    geom_jitter(aes(color = quality), width = 0.2, size = 2, alpha = 0.7) +
    scale_color_manual(values = QUALITY_COLORS, guide = "none") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(x = "Group ID", y = "SPE Accuracy") +
    theme_minimal(base_size = 9) +
    theme(panel.grid.major.x = element_blank())

  # SPE RT
  p2 <- ggplot(sl, aes(x = groupID, y = SPE_RT_ms)) +
    geom_violin(fill = "#d7e8ff", alpha = 0.5, draw_quantiles = 0.5) +
    geom_jitter(aes(color = quality), width = 0.2, size = 2, alpha = 0.7) +
    scale_color_manual(values = QUALITY_COLORS, guide = "none") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(x = "Group ID", y = "SPE RT (ms)") +
    theme_minimal(base_size = 9) +
    theme(panel.grid.major.x = element_blank())

  # Omission Rate
  p3 <- ggplot(sl, aes(x = groupID, y = omission_rate)) +
    geom_violin(fill = "#d7e8ff", alpha = 0.5, draw_quantiles = 0.5) +
    geom_jitter(aes(color = quality), width = 0.2, size = 2, alpha = 0.7) +
    scale_color_manual(values = QUALITY_COLORS) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(x = "Group ID", y = "Omission Rate", color = "Quality") +
    theme_minimal(base_size = 9) +
    theme(panel.grid.major.x = element_blank(),
          legend.position = "top")

  title_gg <- cowplot::ggdraw() +
    cowplot::draw_label("Subject-Level SPE & Quality by Design Group",
                         fontface = "bold", size = 11)
  combined <- cowplot::plot_grid(p1, p2, p3, ncol = 3, align = "h", axis = "bt")
  cowplot::plot_grid(title_gg, combined, ncol = 1, rel_heights = c(0.08, 0.92))
}

#' T×W 设计空间气泡图
plot_design_space <- function(subject_level) {
  group_agg <- subject_level %>%
    group_by(groupID) %>%
    summarise(
      P             = first(P),
      T_ms          = first(T_ms),
      W_ms          = first(W_ms),
      SPE_ACC_mean  = mean(SPE_ACC, na.rm = TRUE),
      SPE_RT_mean   = mean(SPE_RT_ms, na.rm = TRUE),
      n             = n(),
      quality       = first(quality),
      .groups       = "drop"
    )

  ggplot(group_agg, aes(x = T_ms, y = W_ms)) +
    geom_point(aes(size = abs(SPE_RT_mean) * 1.2 + 30,
                   fill = quality),
               shape = 21, alpha = 0.75, stroke = 1.2, color = "white") +
    geom_text(aes(label = paste0("G", groupID)),
              size = 3.5, fontface = "bold", color = "black") +
    scale_fill_manual(values = QUALITY_COLORS) +
    labs(x = "T (ms)", y = "W (ms)",
         title = "Design Space T x W - Group-Level SPE",
         subtitle = "Bubble size proportional to |SPE_RT|",
         fill = "Quality", size = "|SPE_RT|") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold"),
          panel.grid.major = element_line(color = "gray85"))
}

#' 3D 设计空间散点图
plot_3d_design_space <- function(subject_level) {
  sl <- subject_level

  # 使用 scatterplot3d 或 base graphics
  if (!requireNamespace("scatterplot3d", quietly = TRUE)) {
    return(ggplot() +
      annotate("text", x = 1, y = 1,
               label = "3D plot requires 'scatterplot3d' package") +
      theme_void())
  }

  quality_colors <- QUALITY_COLORS[sl$quality]
  quality_colors[is.na(quality_colors)] <- "#888888"

  # 使用 base graphics + scatterplot3d
  # 返回一个 grob
  old_par <- par(no.readonly = TRUE)
  par(mar = c(2, 2, 3, 2))

  s3d <- scatterplot3d::scatterplot3d(
    x = sl$T_ms, y = sl$W_ms, z = sl$SPE_RT_ms,
    color = quality_colors, pch = 16, cex.symbols = 1.2,
    xlab = "T (ms)", ylab = "W (ms)", zlab = "SPE RT (ms)",
    main = "3D Design Space (T x W x SPE_RT)",
    angle = 45, grid = TRUE, box = TRUE
  )

  par(old_par)

  # 无法直接用 cowplot，这里返回 NULL
  return(NULL)
}

# ===========================================================================
# 8. 单个被试完整页面 (一页一图)
# ===========================================================================

#' 绘制单个被试的完整可视化页面
#' 返回 cowplot 组合的 grob
make_subject_page <- function(subj_df, gid, sid, fname, quality_label) {
  formal    <- subj_df[subj_df$stage == "formal", ]
  responded <- formal[formal$responded, ]

  omission <- if (nrow(formal) > 0) (1 - nrow(responded) / nrow(formal)) * 100 else 0
  acc_val  <- if (nrow(responded) > 0) mean(responded$Correct, na.rm = TRUE) * 100 else 0
  mk       <- get_match_key(sid)
  co       <- get_correct_order(sid)

  title_prefix <- paste0("G", gid, "-S", sid)

  # 信息文本
  info_line1 <- paste0(
    "G", gid, "-S", sid, "  |  File: ", fname,
    "  |  Quality: ", toupper(quality_label),
    "  |  MatchKey: ", toupper(mk),
    "  |  CorrectOrder: square->", co$square, ", circle->", co$circle
  )
  info_line2 <- paste0(
    "Formal=", nrow(formal), "  |  Responses=", nrow(responded),
    "  |  Omission=", round(omission, 1), "%",
    "  |  ACC=", round(acc_val, 1), "%"
  )

  # 四个图表
  p1 <- plot_rt_timeseries(responded, title_prefix)

  p2 <- tryCatch(
    plot_rt_histogram(responded, title_prefix),
    error = function(e) {
      ggplot() + annotate("text", x = 1, y = 1, label = paste("RT Hist Error:", e$message)) +
        theme_void()
    }
  )

  subj_summ <- subject_summary_table(subj_df)
  p3 <- tryCatch(
    plot_condition_bars(subj_summ, title_prefix),
    error = function(e) {
      ggplot() + annotate("text", x = 1, y = 1, label = paste("Bar Error:", e$message)) +
        theme_void()
    }
  )

  crf_res <- compute_spe_crf(subj_df, n_quantiles = 5)
  p4 <- tryCatch(
    plot_crf_spe(crf_res$crf_self, crf_res$crf_stranger,
                  crf_res$spe_curve, title_prefix),
    error = function(e) {
      ggplot() + annotate("text", x = 1, y = 1, label = paste("CRF Error:", e$message)) +
        theme_void()
    }
  )

  p5 <- plot_response_heatmap(formal, mk, title_prefix)

  # 组合布局:
  # Row 1: Info header
  # Row 2: RT Timeseries (left) + Response Heatmap (right)
  # Row 3: RT Distribution (left) + Condition Bars (right)
  # Row 4: CRF-SPE (full width)

  info_grob <- grid::textGrob(
    paste(info_line1, info_line2, sep = "\n"),
    x = 0.01, y = 0.5, just = c("left", "center"),
    gp = grid::gpar(fontsize = c(9, 8), col = c("black", "gray40"),
                     fontface = c("bold", "plain"))
  )

  # 使用 cowplot 排列
  top_row <- cowplot::plot_grid(p1, p5, ncol = 2, rel_widths = c(1, 0.5),
                                 align = "h", axis = "bt")
  mid_row <- cowplot::plot_grid(p2, p3, ncol = 2, rel_widths = c(1.2, 1),
                                 align = "h", axis = "bt")

  cowplot::plot_grid(
    cowplot::ggdraw() + cowplot::draw_label(info_line1, fontface = "bold", size = 9, hjust = 0, x = 0.01) +
      cowplot::draw_label(info_line2, size = 8, color = "gray40", hjust = 0, x = 0.01, y = 0.2),
    top_row, mid_row, p4,
    ncol = 1, rel_heights = c(0.05, 0.28, 0.30, 0.37)
  )
}

# ===========================================================================
# 9. 图表说明页面 (PDF 最前面)
# ===========================================================================

make_description_page <- function() {
  # 这是一个纯文本说明页，使用 base R graphics
  # 返回 NULL，在外层手动绘制

  description_text <- c(
    "======================================================================",
    "  Self-Matching Task 全被试数据可视化报告 (V4 - R Language)",
    "  Shape-Label Association Task (Sui et al., 2012)",
    "======================================================================",
    "",
    "【实验范式说明】",
    "  被试需要判断屏幕上出现的图形(方形 square / 圆形 circle)与标签",
    "  (自我 self / 陌生人 stranger)是否匹配。每个被试有独特的匹配规则:",
    "  - 偶数ID被试: square->self, circle->stranger 为 Matching",
    "  - 奇数ID被试: square->stranger, circle->self 为 Matching",
    "  - 匹配键(MatchKey)根据 subjectID 分配: f 键或 j 键",
    "",
    "【实验设计参数】",
    "  P (Practice): 练习试次数量",
    "  T (Stimulus Time): 刺激呈现时间 (秒), 之后刺激消失仅留空白屏",
    "  W (Response Window): 反应窗口时间 (秒), 被试可在此时间内按键反应",
    "  M = T + W: 总可用反应时间",
    "",
    "  共 8 个设计组 (G1-G8):",
    "    G1: P=0,   T=30ms,   W=300ms   (exclude - 极高遗漏率)",
    "    G2: P=0,   T=30ms,   W=600ms   (exclude)",
    "    G3: P=120, T=30ms,   W=600ms   (caution)",
    "    G4: P=120, T=80ms,   W=600ms   (good)",
    "    G5: P=8,   T=100ms,  W=1100ms  (good)",
    "    G6: P=120, T=500ms,  W=1500ms  (good)",
    "    G7: P=0,   T=100ms,  W=1100ms  (good)",
    "    G8: P=120, T=30ms,   W=800ms   (good)",
    "",
    "【每个被试图表说明】",
    "  每个被试占用一页，包含以下 5 个子图:",
    "",
    "  1. RT Timeseries (左上): 反应时随试次的变化",
    "     X轴=trialID (试次序号), Y轴=RT (ms)", 
    "     橙色=Self, 蓝色=Stranger",
    "     解读: 观察学习效应、疲劳效应、异常试次段落",
    "",
    "  2. Response Heatmap (右上): 按键偏好热力图",
    "     展示 Matching/NonMatching x Self/Stranger 四种条件下",
    "     被试按 f 键、j 键和遗漏(miss)的频次",
    "     解读: 判断被试是否正确使用匹配键/不匹配键, 是否有键偏好",
    "",
    "  3. RT Distribution (左中): 反应时分布",
    "     按 Condition (Matching/NonMatching) 分面,",
    "     Self(橙色) vs Stranger(蓝色) 对比",
    "     解读: 比较不同条件下 RT 分布的中心趋势和离散程度",
    "",
    "  4. Condition Bars (右中): 条件分解柱状图",
    "     三个指标: Accuracy(%), RT Mean(ms), Omission(%)",
    "     按 Identity x Condition 四个组合分别展示",
    "     解读: 量化 Self/Stranger 在 Matching/NonMatching 下的行为差异",
    "",
    "  5. CRF & SPE (底部): 条件反应函数与自我优先效应",
    "     左图: CRF - P(按匹配键) 随 RT 分位的变化",
    "           橙色=Self, 蓝色=Stranger, 虚线=0.5 (随机水平)",
    "     右图: CRF-SPE = Self CRF - Stranger CRF",
    "           正值表示 Self 比 Stranger 更倾向按匹配键",
    "           95% CI (灰色区间), 虚线=0 (无SPE)",
    "     解读: SPE 是否在特定 RT 区间更强 (如快反应/慢反应)",
    "",
    "【群体汇总图说明】",
    "  在所有个被试图表之后, 包含:",
    "  6. SPE by Group: 各组 SPE_ACC, SPE_RT, Omission 分布",
    "  7. Design Space T x W: 设计空间中的组级 SPE 气泡图",
    "  8. 3D Design Space: T x W x SPE_RT 三维设计空间",
    "",
    "【数据来源】",
    paste0("  原始数据目录: ", RAW_DIR),
    "  数据清理逻辑: 100% 复现 app_server.py 中的处理流程",
    "  生成脚本: V4/Self_Matching_Task_Visualization_V4.R",
    "",
    "【SPE 指标定义】",
    "  SPE_ACC = ACC_self - ACC_stranger (正值=自我正确率优势)",
    "  SPE_RT_ms = RT_stranger - RT_self (正值=自我反应更快)",
    "  CRF-SPE = P(MatchKey|Self) - P(MatchKey|Stranger) 随RT变化",
    "======================================================================"
  )

  return(description_text)
}

# ===========================================================================
# 10. 主流程：生成 PDF
# ===========================================================================

generate_all_charts_pdf <- function() {
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("  Self-Matching Task 全被试可视化 PDF 生成器 V4 (R)\n")
  cat(paste(rep("=", 60), collapse = ""), "\n\n")

  # ---- 1. 加载全部数据 ----
  cat("[1/5] Loading data...\n")
  all_df <- load_all_raw(RAW_DIR)
  cat(sprintf("  Loaded %d trial records\n", nrow(all_df)))

  # 获取所有 (groupID, subjectID) 组合
  file_manifest <- all_df %>%
    group_by(groupID, subjectID, quality) %>%
    summarise(source_file = first(source_file), .groups = "drop") %>%
    arrange(groupID, subjectID)
  cat(sprintf("  Total subjects: %d\n", nrow(file_manifest)))

  # ---- 2. 计算所有被试级汇总 ----
  cat("\n  Computing subject-level summaries...\n")
  # 使用 group_modify (dplyr >= 0.8.0), fallback to split + lapply
  subject_level <- tryCatch(
    all_df %>%
      group_by(groupID, subjectID) %>%
      group_modify(~ summarize_subject(.x)) %>%
      ungroup(),
    error = function(e) {
      # fallback: split-apply-combine
      message("  group_modify failed, using split-apply-combine fallback...")
      splitted <- split(all_df, list(all_df$groupID, all_df$subjectID), drop = TRUE)
      result_list <- lapply(splitted, function(df_chunk) {
        if (nrow(df_chunk) > 0) summarize_subject(df_chunk) else NULL
      })
      do.call(rbind, result_list[!sapply(result_list, is.null)])
    }
  )
  cat(sprintf("  %d subjects summarized\n", nrow(subject_level)))

  # ---- 3. 打开 PDF ----
  cat("\n[2/5] Opening PDF device...\n")
  pdf(PDF_PATH, width = 14, height = 10, onefile = TRUE,
      title = "Self-Matching Task Visualization V4",
      family = pdf_family)
  # 注意: cowplot 的图需要通过 print() 或 plot() 输出到当前设备

  # ---- 4. 封面页 ----
  cat("[3/5] Generating cover page...\n")
  # 封面使用 base graphics
  op <- par(mar = c(0, 0, 0, 0))
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1))
  text(0.5, 0.6, "Self-Matching Task\n全被试数据可视化报告 V4",
       cex = 2.2, font = 2, col = "#1a237e")
  text(0.5, 0.45, "88 Subjects x 8 Design Groups\nShape-Label Association Task (Sui et al., 2012)",
       cex = 1.1, col = "gray40")
  text(0.5, 0.33, paste0("Generated from RAW data\n", Sys.time()),
       cex = 0.8, col = "gray60")
  text(0.5, 0.25, "R Language Implementation | 100% reproduce app_server.py logic",
       cex = 0.8, col = "gray60")
  par(op)

  # ---- 5. 图表说明页 ----
  cat("  Generating description page...\n")
  desc_lines <- make_description_page()
  op <- par(mar = c(1, 2, 1, 2))
  plot.new()
  plot.window(xlim = c(0, 100), ylim = c(0, 120))
  for (i in seq_along(desc_lines)) {
    y_pos <- 120 - i * 1.8
    if (y_pos < 0) break
    # 判断是否为标题行
    is_heading <- grepl("^【.*】", desc_lines[i]) |
      grepl("^====", desc_lines[i]) |
      grepl("^  Self-Matching", desc_lines[i])
    font_val <- if (is_heading) 2 else 1
    cex_val  <- if (grepl("^====", desc_lines[i])) 0.7 else 0.55
    col_val  <- if (grepl("^====", desc_lines[i])) "gray50" else "black"
    text(1, y_pos, desc_lines[i], pos = 4, cex = cex_val,
         font = font_val, col = col_val)
  }
  par(op)

  # ---- 6. 数据总览页 ----
  cat("  Generating data overview page...\n")
  # Group summary table & distributions
  group_counts <- file_manifest %>% group_by(groupID) %>% summarise(n = n(), .groups = "drop")

  # 直方图
  p_hist_acc <- ggplot(subject_level, aes(x = accuracy)) +
    geom_histogram(bins = 20, fill = "#2196f3", alpha = 0.7, color = "white") +
    geom_vline(aes(xintercept = median(accuracy, na.rm = TRUE)),
               linetype = "dashed", color = "red") +
    labs(title = "Accuracy Distribution", x = "Accuracy", y = "N subjects") +
    theme_minimal(base_size = 9)

  p_hist_omi <- ggplot(subject_level, aes(x = omission_rate)) +
    geom_histogram(bins = 20, fill = "#f44336", alpha = 0.7, color = "white") +
    labs(title = "Omission Rate Distribution", x = "Omission Rate", y = "N subjects") +
    theme_minimal(base_size = 9)

  p_hist_spa <- ggplot(subject_level, aes(x = SPE_ACC)) +
    geom_histogram(bins = 20, fill = "#4caf50", alpha = 0.7, color = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
    labs(title = "SPE Accuracy Distribution", x = "SPE ACC", y = "N subjects") +
    theme_minimal(base_size = 9)

  p_hist_spr <- ggplot(subject_level, aes(x = SPE_RT_ms)) +
    geom_histogram(bins = 20, fill = "#9c27b0", alpha = 0.7, color = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
    labs(title = "SPE RT Distribution", x = "SPE RT (ms)", y = "N subjects") +
    theme_minimal(base_size = 9)

  grid_hist <- cowplot::plot_grid(p_hist_acc, p_hist_omi, p_hist_spa, p_hist_spr,
                                   ncol = 2, align = "h")

  # 箱线图
  sl_g <- subject_level
  sl_g$groupID <- factor(sl_g$groupID)

  p_box_acc <- ggplot(sl_g, aes(x = groupID, y = accuracy)) +
    geom_boxplot(aes(color = groupID), outlier.size = 1.5) +
    scale_color_manual(values = GROUP_PALETTE, guide = "none") +
    labs(title = "Accuracy by Group", x = "Group", y = "Accuracy") +
    theme_minimal(base_size = 9)

  p_box_omi <- ggplot(sl_g, aes(x = groupID, y = omission_rate)) +
    geom_boxplot(aes(color = groupID), outlier.size = 1.5) +
    scale_color_manual(values = GROUP_PALETTE, guide = "none") +
    labs(title = "Omission Rate by Group", x = "Group", y = "Omission Rate") +
    theme_minimal(base_size = 9)

  p_box_spa <- ggplot(sl_g, aes(x = groupID, y = SPE_ACC)) +
    geom_boxplot(aes(color = groupID), outlier.size = 1.5) +
    scale_color_manual(values = GROUP_PALETTE, guide = "none") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = "SPE ACC by Group", x = "Group", y = "SPE ACC") +
    theme_minimal(base_size = 9)

  p_box_spr <- ggplot(sl_g, aes(x = groupID, y = SPE_RT_ms)) +
    geom_boxplot(aes(color = groupID), outlier.size = 1.5) +
    scale_color_manual(values = GROUP_PALETTE, guide = "none") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = "SPE RT by Group", x = "Group", y = "SPE RT (ms)") +
    theme_minimal(base_size = 9)

  grid_box <- cowplot::plot_grid(p_box_acc, p_box_omi, p_box_spa, p_box_spr,
                                  ncol = 4, align = "h")

  title_overview <- cowplot::ggdraw() +
    cowplot::draw_label(paste0("Data Overview - ", nrow(subject_level), " Subjects"),
                         fontface = "bold", size = 13)

  overview_page <- cowplot::plot_grid(
    title_overview, grid_hist, grid_box,
    ncol = 1, rel_heights = c(0.06, 0.47, 0.47)
  )
  print(overview_page)

  # ---- 7. 群体汇总图: SPE 小提琴图 ----
  cat("  Generating group summary plots...\n")
  p_spe_violin <- plot_spe_by_group(subject_level)
  print(p_spe_violin)

  # ---- 8. T×W 设计空间气泡图 ----
  p_design <- plot_design_space(subject_level)
  print(p_design)

  # ---- 9. 遍历每个被试生成个体图表 ----
  cat("\n[4/5] Generating individual subject pages...\n")
  total <- nrow(file_manifest)

  for (idx in seq_len(total)) {
    gid <- as.integer(file_manifest$groupID[idx])
    sid <- as.integer(file_manifest$subjectID[idx])
    fname <- file_manifest$source_file[idx]
    quality_label <- file_manifest$quality[idx]

    cat(sprintf("  [%d/%d] G%d-S%d (%s)", idx, total, gid, sid, quality_label))

    subj <- all_df[all_df$groupID == gid & all_df$subjectID == sid, ]

    if (nrow(subj[subj$stage == "formal", ]) == 0) {
      cat(" - no formal trials, skip\n")
      next
    }

    # 计算 SPE 用于显示
    spe_info <- ""
    crf_res <- tryCatch(
      compute_spe_crf(subj, n_quantiles = 5),
      error = function(e) list(spe_curve = data.frame())
    )
    if (nrow(crf_res$spe_curve) > 0) {
      mean_spe <- mean(crf_res$spe_curve$spe_upper_prop, na.rm = TRUE)
      spe_info <- sprintf(" SPE=%.1f%%", mean_spe * 100)
    }

    # 生成并打印页面
    page <- tryCatch(
      make_subject_page(subj, gid, sid, fname, quality_label),
      error = function(e) {
        ggplot() +
          annotate("text", x = 1, y = 1,
                   label = paste("Error generating page for G", gid, "S", sid, ":", e$message)) +
          theme_void()
      }
    )
    print(page)

    cat(paste0(spe_info, " OK\n"))
  }

  # ---- 10. 关闭 PDF ----
  cat("\n[5/5] Closing PDF...\n")
  dev.off()

  # ---- 输出摘要 ----
  cat("\n", paste(rep("=", 60), collapse = ""), "\n")
  cat("  PDF generation complete!\n")
  cat("  Output: ", PDF_PATH, "\n")
  if (file.exists(PDF_PATH)) {
    size_mb <- file.info(PDF_PATH)$size / (1024 * 1024)
    cat(sprintf("  File size: %.1f MB\n", size_mb))
  }
  cat(paste(rep("=", 60), collapse = ""), "\n")
}

# ===========================================================================
# 11. 入口
# ===========================================================================

# 当直接运行此脚本时自动执行 PDF 生成
# 如果通过 source() 加载, 可手动调用 generate_all_charts_pdf()

if (sys.nframe() == 0) {
  # 直接运行模式
  cat("\n=== Direct execution mode ===\n")
  cat("Starting PDF generation...\n\n")
  generate_all_charts_pdf()
} else {
  cat("\n============================================\n")
  cat("  V4 R Script loaded successfully.\n")
  cat("  Run generate_all_charts_pdf() to start.\n")
  cat("  Or use: Rscript Self_Matching_Task_Visualization_V4.R\n")
  cat("============================================\n\n")
}
