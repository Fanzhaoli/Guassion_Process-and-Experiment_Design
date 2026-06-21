###############################################################################
# shared/utils.R — 共享工具函数
# 供所有 R_Version 子模块 source() 使用
###############################################################################

# ===========================================================================
# 0. 依赖包加载
# ===========================================================================
required_packages <- c("ggplot2", "dplyr", "tidyr", "readr", "scales",
                        "cowplot", "RColorBrewer", "grid", "gridExtra")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.r-project.org")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

# ===========================================================================
# 1. 全局路径
# ===========================================================================
PROJECT_ROOT <- file.path("D:", "GitHub_programe", "GitHub",
                           "Guassion-Process-Experiment-Design")
RAW_DIR      <- file.path(PROJECT_ROOT, "2_Data", "Real_Data", "UnExtact", "raw")
SPE_DIR      <- file.path(PROJECT_ROOT, "2_Data", "Real_Data", "SPE_Database")
R_VERSION_DIR <- file.path(PROJECT_ROOT, "1_Code", "Python_for_Check",
                            "Visualization", "R_Version")

# ===========================================================================
# 2. 实验参数常量 (from app_server.py CONDITIONS)
# ===========================================================================
CONDITIONS <- list(
  "1" = list(P = 0,   T = 0.03, W = 0.3,  label = "G1  | P0_T30_W300"),
  "2" = list(P = 0,   T = 0.03, W = 0.6,  label = "G2  | P0_T30_W600"),
  "3" = list(P = 120, T = 0.03, W = 0.6,  label = "G3  | P120_T30_W600"),
  "4" = list(P = 120, T = 0.08, W = 0.6,  label = "G4  | P120_T80_W600"),
  "5" = list(P = 8,   T = 0.10, W = 1.1,  label = "G5  | P8_T100_W1100"),
  "6" = list(P = 120, T = 0.50, W = 1.5,  label = "G6  | P120_T500_W1500"),
  "7" = list(P = 0,   T = 0.10, W = 1.1,  label = "G7  | P0_T100_W1100"),
  "8" = list(P = 120, T = 0.03, W = 0.8,  label = "G8  | P120_T30_W800"),
  "9" = list(P = 120, T = 0.08, W = 0.8,  label = "G9  | P120_T80_W800")
)

QUALITY_MAP <- c(
  "1" = "exclude", "2" = "exclude", "3" = "caution",
  "4" = "good",    "5" = "good",    "6" = "good",
  "7" = "good",    "8" = "good",    "9" = "good"
)

# ===========================================================================
# 3. 通用主题
# ===========================================================================
theme_spe <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      plot.title = element_text(face = "bold", size = base_size + 1),
      axis.title = element_text(size = base_size - 1),
      axis.text  = element_text(size = base_size - 2),
      legend.position = "bottom",
      legend.title = element_text(size = base_size - 1),
      legend.text  = element_text(size = base_size - 2)
    )
}

# ===========================================================================
# 4. 统计函数：Cohen's d (从 Python app_server 复刻)
# ===========================================================================
cohens_d <- function(group1, group2) {
  # 复刻 Python 版 cohens_d:
  #   n1, n2 = len(group1), len(group2)
  #   var1, var2 = statistics.variance(group1), statistics.variance(group2)
  #   pooled_sd = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
  #   return (mean(group2) - mean(group1)) / pooled_sd
  n1 <- length(group1)
  n2 <- length(group2)
  if (n1 < 2 || n2 < 2) return(NA_real_)
  var1 <- var(group1)
  var2 <- var(group2)
  pooled_sd <- sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
  if (pooled_sd == 0) return(NA_real_)
  (mean(group2) - mean(group1)) / pooled_sd
}

# Pearson 相关系数
pearson_r <- function(x, y) {
  ok <- complete.cases(x, y)
  if (sum(ok) < 3) return(NA_real_)
  cor(x[ok], y[ok], method = "pearson")
}

# 线性回归拟合
linear_fit <- function(x, y) {
  ok <- complete.cases(x, y)
  if (sum(ok) < 2) return(NULL)
  lm(y ~ x, data = data.frame(x = x[ok], y = y[ok]))
}

# ===========================================================================
# 5. 数据加载工具
# ===========================================================================

#' 列出所有原始实验数据文件
list_raw_files <- function() {
  files <- list.files(RAW_DIR, pattern = "^EXP_data_group.*\\.csv$", full.names = TRUE)
  info <- data.frame(
    filename = basename(files),
    fullpath = files,
    stringsAsFactors = FALSE
  )
  # 提取 group 和 subject
  m <- regmatches(info$filename, regexec("group(\\d+)_(\\d+)", info$filename))
  info$group   <- as.integer(sapply(m, `[`, 2))
  info$subject <- as.integer(sapply(m, `[`, 3))
  info[order(info$group, info$subject), ]
}

#' 加载单个原始数据文件
load_file_data <- function(filepath) {
  df <- read.csv(filepath, stringsAsFactors = FALSE,
                 fileEncoding = "UTF-8", check.names = FALSE)
  # 补齐缺失列
  cols_needed <- c("groupID", "subjectID", "stage", "trialID",
                   "P", "T", "W", "Shape", "Label", "CorrectKey",
                   "Response", "RT", "Correct")
  for (cn in setdiff(cols_needed, names(df))) df[[cn]] <- NA
  df
}

#' 加载所有原始数据
load_all_data <- function(group_filter = NULL) {
  files <- list_raw_files()
  if (!is.null(group_filter)) files <- files[files$group %in% group_filter, ]
  all_rows <- lapply(seq_len(nrow(files)), function(i) {
    f <- files[i, ]
    df <- load_file_data(f$fullpath)
    df$file_source <- f$filename
    df
  })
  bind_rows(all_rows)
}

#' 加载 SPE 数据库处理日志
load_spe_log <- function() {
  log_file <- file.path(SPE_DIR, "processing_log.csv")
  if (!file.exists(log_file)) stop("processing_log.csv not found")
  read.csv(log_file, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
}

#' 加载单个 SPE 实验数据
load_spe_file <- function(pair_key) {
  log <- load_spe_log()
  entry <- log[log$Pair_Key == pair_key, ]
  if (nrow(entry) == 0) stop("Pair_Key not found: ", pair_key)
  sp_file <- file.path(SPE_DIR, basename(entry$Output_File))
  if (!file.exists(sp_file)) stop("SPE file not found: ", sp_file)
  read.csv(sp_file, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
}

#' 为 SPE 数据选择 Identity 列
select_identity_column <- function(df, prefer = c("label", "shape")) {
  prefer <- match.arg(prefer)
  cols <- names(df)
  if (prefer == "label") {
    if ("Label_Standardized_Identity" %in% cols) return("Label_Standardized_Identity")
    if ("Shape_Standardized_Identity" %in% cols) return("Shape_Standardized_Identity")
    return(NULL)
  } else {
    if ("Shape_Standardized_Identity" %in% cols) return("Shape_Standardized_Identity")
    if ("Label_Standardized_Identity" %in% cols) return("Label_Standardized_Identity")
    return(NULL)
  }
}

# ===========================================================================
# 6. SPE 计算核心
# ===========================================================================

#' 对标 Python _compute_subject_spe
#' 返回: list(identity_types, per_identity, comparisons, legacy, n_subjects)
compute_subject_spe <- function(df, identity_col, rt_col = "RT_ms",
                                 acc_col = "ACC", condition_filter = "all") {
  # 过滤 Matching 条件
  has_matching <- "Matching" %in% names(df)
  if (condition_filter != "all" && has_matching) {
    df <- df[tolower(df$Matching) == tolower(condition_filter), ]
  }

  # 过滤无效行
  df <- df[!is.na(df[[identity_col]]) & df[[identity_col]] != "" &
            tolower(df[[identity_col]]) != "na", ]
  df[[rt_col]] <- as.numeric(df[[rt_col]])
  df <- df[!is.na(df[[rt_col]]) & df[[rt_col]] > 0, ]

  identity_types <- sort(unique(df[[identity_col]]))

  # 按被试 × Identity 分组
  subj_rt  <- split(df[[rt_col]], list(df$Subject, df[[identity_col]]), drop = TRUE)
  if (!is.null(acc_col) && acc_col %in% names(df)) {
    df[[acc_col]] <- as.numeric(df[[acc_col]])
    subj_acc <- split(df[[acc_col]], list(df$Subject, df[[identity_col]]), drop = TRUE)
  }

  # Per-identity 汇总
  per_identity <- lapply(identity_types, function(ident) {
    rt_vals <- df[[rt_col]][df[[identity_col]] == ident]
    acc_vals <- if (!is.null(acc_col) && acc_col %in% names(df))
      df[[acc_col]][df[[identity_col]] == ident] else numeric(0)
    n_subs <- length(unique(df$Subject[df[[identity_col]] == ident]))
    list(
      rts = rt_vals, accs = acc_vals,
      n_subjects = n_subs,
      mean_rt = if (length(rt_vals)) round(mean(rt_vals), 1) else NULL,
      mean_acc = if (length(acc_vals)) round(mean(acc_vals), 4) else NULL,
      sd_rt = if (length(rt_vals) >= 2) round(sd(rt_vals), 1) else NULL,
      sd_acc = if (length(acc_vals) >= 2) round(sd(acc_vals), 4) else NULL
    )
  })
  names(per_identity) <- identity_types

  # Self vs 每种其他 Identity 的 Cohen's d
  self_key <- if ("Self" %in% identity_types) "Self" else {
    hits <- identity_types[tolower(identity_types) == "self"]
    if (length(hits)) hits[1] else NULL
  }
  comparisons <- list()
  if (!is.null(self_key)) {
    for (other in setdiff(identity_types, self_key)) {
      subj_d_rt  <- c()
      subj_d_acc <- c()
      all_sids <- unique(df$Subject)
      for (sid in all_sids) {
        s_rt <- df[[rt_col]][df$Subject == sid & df[[identity_col]] == self_key]
        o_rt <- df[[rt_col]][df$Subject == sid & df[[identity_col]] == other]
        if (length(s_rt) >= 3 && length(o_rt) >= 3) {
          d_rt <- cohens_d(o_rt, s_rt)  # Other - Self (RT)
          subj_d_rt <- c(subj_d_rt, d_rt)
        }
        if (!is.null(acc_col) && acc_col %in% names(df)) {
          s_ac <- as.numeric(df[[acc_col]][df$Subject == sid & df[[identity_col]] == self_key])
          o_ac <- as.numeric(df[[acc_col]][df$Subject == sid & df[[identity_col]] == other])
          if (length(s_ac) >= 3 && length(o_ac) >= 3) {
            d_acc <- cohens_d(s_ac, o_ac)  # Self - Other (ACC)
            subj_d_acc <- c(subj_d_acc, d_acc)
          }
        }
      }
      comparisons[[other]] <- list(
        spe_rt_d  = if (length(subj_d_rt))  round(mean(subj_d_rt), 4)  else NULL,
        spe_rt_se = if (length(subj_d_rt) >= 2) round(sd(subj_d_rt) / sqrt(length(subj_d_rt)), 4) else NULL,
        spe_acc_d = if (length(subj_d_acc)) round(mean(subj_d_acc), 4) else NULL,
        spe_acc_se = if (length(subj_d_acc) >= 2) round(sd(subj_d_acc) / sqrt(length(subj_d_acc)), 4) else NULL,
        n_valid   = length(subj_d_rt)
      )
    }
  }

  primary_other <- if ("Stranger" %in% names(comparisons)) "Stranger"
                   else if (length(comparisons)) names(comparisons)[1] else NULL

  # Legacy 兼容
  legacy <- list(
    self_rts_all = if (!is.null(self_key)) per_identity[[self_key]]$rts else numeric(0),
    stranger_rts_all = if (!is.null(primary_other)) per_identity[[primary_other]]$rts else numeric(0),
    self_accs_all = if (!is.null(self_key)) per_identity[[self_key]]$accs else numeric(0),
    stranger_accs_all = if (!is.null(primary_other)) per_identity[[primary_other]]$accs else numeric(0),
    d_vals = if (!is.null(primary_other) && !is.null(comparisons[[primary_other]]$spe_rt_d))
      comparisons[[primary_other]]$spe_rt_d else numeric(0),
    acc_d_vals = if (!is.null(primary_other) && !is.null(comparisons[[primary_other]]$spe_acc_d))
      comparisons[[primary_other]]$spe_acc_d else numeric(0),
    n_subjects_valid = if (!is.null(primary_other) && !is.null(comparisons[[primary_other]]$n_valid))
      comparisons[[primary_other]]$n_valid else 0L
  )

  list(
    identity_types     = identity_types,
    per_identity       = per_identity,
    comparisons        = comparisons,
    primary_comparison = primary_other,
    legacy             = legacy,
    n_subjects         = length(unique(df$Subject))
  )
}

# ===========================================================================
# 7. CRF 分箱计算
# ===========================================================================
compute_crf_bins <- function(df, identity_sel = NULL, matching_sel = NULL,
                              n_quantiles = 5, y_mode = c("pMatch", "acc")) {
  y_mode <- match.arg(y_mode)
  if (!is.null(identity_sel)) df <- df[df$Identity %in% identity_sel, ]
  if (!is.null(matching_sel)) df <- df[df$Matching %in% matching_sel, ]
  if (nrow(df) < 10) return(data.frame())

  # 按 RT 排序
  df <- df[order(df$RT_ms), ]
  n <- nrow(df)
  bin_size <- floor(n / n_quantiles)
  bins <- lapply(seq_len(n_quantiles), function(i) {
    start <- (i - 1) * bin_size + 1
    end   <- if (i == n_quantiles) n else i * bin_size
    bin_data <- df[start:end, ]
    rt_mean <- mean(bin_data$RT_ms)
    rt_sem  <- sd(bin_data$RT_ms) / sqrt(nrow(bin_data))

    if (y_mode == "pMatch") {
      n_match <- sum(grepl("^M", bin_data$Matching, ignore.case = TRUE))
      y_val   <- n_match / nrow(bin_data)
    } else {
      n_correct <- sum(bin_data$ACC == 1, na.rm = TRUE)
      y_val     <- n_correct / nrow(bin_data)
    }
    y_sem <- sqrt(y_val * (1 - y_val) / nrow(bin_data))

    data.frame(x = rt_mean, y = y_val, xSEM = rt_sem, ySEM = y_sem,
               n = nrow(bin_data), stringsAsFactors = FALSE)
  })
  bind_rows(bins)
}

# ===========================================================================
# 8. 导出图片辅助
# ===========================================================================
save_plot_png <- function(plot, path, width = 8, height = 5, dpi = 150) {
  ggsave(path, plot, width = width, height = height, dpi = dpi,
         device = "png", bg = "white")
  message("Saved: ", path)
}

save_plot_pdf <- function(plot, path, width = 8, height = 5) {
  ggsave(path, plot, width = width, height = height,
         device = "pdf")
  message("Saved: ", path)
}
