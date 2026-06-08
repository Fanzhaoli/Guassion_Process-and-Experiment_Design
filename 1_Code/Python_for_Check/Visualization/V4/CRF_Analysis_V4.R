###############################################################################
# CRF (Conditional Response Function) 专项分析 V4
# 生成包含所有被试 CRF 曲线的 PDF 报告
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

# Windows 字体
if (.Platform$OS.type == "windows") {
  tryCatch({
    windowsFonts(yahei = windowsFont("Microsoft YaHei"))
  }, error = function(e) {})
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

PDF_PATH <- file.path(OUT_DIR, "CRF_Analysis_V4.pdf")

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
GROUP_PALETTE  <- c("#e91e63", "#2196f3", "#4caf50", "#ff9800",
                     "#9c27b0", "#00bcd4", "#ffeb3b", "#795548", "#607d8b")

COL_SELF     <- "#ff9800"
COL_STRANGER <- "#2196f3"
COL_SPE      <- "#9c27b0"
COL_ACC      <- "#4caf50"
COL_OMISSION <- "#f44336"

# ===========================================================================
# 3. 核心实验规则函数 (100% 复现 app_server.py)
# ===========================================================================
get_match_key <- function(subject_id) {
  c("f", "j", "j", "f")[((subject_id - 1) %% 4) + 1]
}

get_correct_order <- function(subject_id) {
  if (subject_id %% 2 == 0) list(square = "self", circle = "stranger")
  else                       list(square = "stranger", circle = "self")
}

compute_condition <- function(shape, label, subject_id) {
  expected_label <- get_correct_order(subject_id)[[shape]]
  ifelse(label == expected_label, "Matching", "NonMatching")
}

# ===========================================================================
# 4. 数据加载 
# ===========================================================================
parse_file_ids <- function(filename) {
  stem <- sub("\\.csv$", "", filename)
  parts <- strsplit(sub("^EXP_data_group", "", stem), "_")[[1]]
  list(groupID = as.integer(parts[1]), subjectID = as.integer(parts[2]))
}

load_one_file <- function(filepath) {
  fname <- basename(filepath)
  ids <- parse_file_ids(fname)
  gid_from_file <- ids$groupID
  sid_from_file <- ids$subjectID

  df <- read.csv(filepath, stringsAsFactors = FALSE,
                  na.strings = c("NA", "nan", "NaN", ""))

  df$groupID    <- as.integer(df$groupID)
  df$subjectID  <- as.integer(df$subjectID)
  df$trialID    <- as.integer(df$trialID)
  df$Shape      <- trimws(tolower(as.character(df$Shape)))
  df$Label      <- trimws(tolower(as.character(df$Label)))
  df$Response   <- trimws(tolower(as.character(df$Response)))
  df$CorrectKey <- trimws(tolower(as.character(df$CorrectKey)))

  df$stage <- ifelse(is.na(df$stage) | df$stage == "", "formal",
                      as.character(df$stage))

  df$RT      <- suppressWarnings(as.numeric(df$RT))
  df$Correct <- suppressWarnings(as.numeric(df$Correct))

  df$responded <- !is.na(df$RT) & !is.na(df$Response) &
    !(df$Response %in% c("na", "nan", ""))

  subject_id_ref <- sid_from_file
  df$Condition <- sapply(seq_len(nrow(df)), function(i) {
    compute_condition(df$Shape[i], df$Label[i], subject_id_ref)
  })
  df$Identity <- ifelse(df$Label == "self", "Self", "Stranger")
  df$MatchKey <- get_match_key(subject_id_ref)
  df$ResponseIsMatch <- ifelse(df$responded,
                                as.integer(df$Response == df$MatchKey),
                                NA_integer_)

  df$P <- suppressWarnings(as.numeric(df$P))
  df$T <- suppressWarnings(as.numeric(df$T))
  df$W <- suppressWarnings(as.numeric(df$W))
  df$T_ms  <- df$T * 1000
  df$W_ms  <- df$W * 1000
  df$M_ms  <- df$T_ms + df$W_ms
  df$RT_ms <- df$RT * 1000

  qual_val <- QUALITY_MAP[as.character(gid_from_file)]
  if (is.na(qual_val)) qual_val <- "unknown"
  df$quality <- qual_val

  df$source_file <- fname
  df$groupID_from_file   <- gid_from_file
  df$subjectID_from_file <- sid_from_file
  return(df)
}

load_all_raw <- function(raw_dir = RAW_DIR) {
  files <- sort(list.files(raw_dir, pattern = "EXP_data_group.*\\.csv$",
                            full.names = TRUE))
  if (length(files) == 0) stop("No EXP_data_group*.csv found in ", raw_dir)
  message("Loading ", length(files), " files...")
  all_dfs <- lapply(files, load_one_file)
  all_df <- do.call(rbind, all_dfs)
  rownames(all_df) <- NULL
  message("Total rows: ", nrow(all_df))
  return(all_df)
}

# ===========================================================================
# 5. CRF 计算核心函数
# ===========================================================================

#' 计算单个 CRF (Conditional Response Function)
#' 逻辑: 按 RT 升序排列 → 等分 n_quantiles 份 → 每份计算 P(MatchKey)
#' 100% 复现 Python compute_crf
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
      bin        = i,
      n          = n_b,
      rt_mean    = mean(b$RT, na.rm = TRUE),
      rt_mean_ms = mean(b$RT_ms, na.rm = TRUE),
      upper_prop = p,
      sem        = if (n_b > 1) sd_val / sqrt(n_b) else 0,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, bins)
}

#' 计算 CRF-SPE: Self vs Stranger 的 CRF 差异
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

#' 汇总被试级 CRF-SPE 指标
summarize_crf_subject <- function(trials, n_quantiles = 5) {
  res <- compute_spe_crf(trials, n_quantiles)
  spe <- res$spe_curve
  data.frame(
    groupID          = trials$groupID[1],
    subjectID        = trials$subjectID[1],
    quality          = trials$quality[1],
    P                = trials$P[1],
    T_ms             = trials$T_ms[1],
    W_ms             = trials$W_ms[1],
    M_ms             = trials$M_ms[1],
    n_formal         = sum(trials$stage == "formal"),
    n_responded      = sum(trials$responded & trials$stage == "formal", na.rm = TRUE),
    crf_n_quantiles  = n_quantiles,
    spe_mean         = if (nrow(spe) > 0) mean(spe$spe_upper_prop, na.rm = TRUE) else NA,
    spe_sd           = if (nrow(spe) > 0) sd(spe$spe_upper_prop, na.rm = TRUE) else NA,
    spe_max          = if (nrow(spe) > 0) max(spe$spe_upper_prop, na.rm = TRUE) else NA,
    spe_min          = if (nrow(spe) > 0) min(spe$spe_upper_prop, na.rm = TRUE) else NA,
    spe_positive_bins = if (nrow(spe) > 0) sum(spe$spe_upper_prop > 0, na.rm = TRUE) else NA,
    stringsAsFactors = FALSE
  )
}

# ===========================================================================
# 6. CRF 可视化函数
# ===========================================================================

#' 单个被试的 CRF + SPE 曲线 (大图版，更详细的标注)
plot_crf_subject <- function(trials, gid, sid, n_quantiles = 5) {
  res <- compute_spe_crf(trials, n_quantiles)
  crf_s  <- res$crf_self
  crf_st <- res$crf_stranger
  spe    <- res$spe_curve

  mk <- get_match_key(sid)
  co <- get_correct_order(sid)
  title_prefix <- paste0("G", gid, "-S", sid)

  # ---- CRF 面板 ----
  p_crf <- ggplot() +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray50", linewidth = 0.6) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.5, ymax = 1,
             fill = COL_ACC, alpha = 0.04) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = 0.5,
             fill = COL_OMISSION, alpha = 0.04)

  if (nrow(crf_s) > 0) {
    p_crf <- p_crf +
      geom_ribbon(data = crf_s,
                   aes(x = rt_mean_ms, ymin = upper_prop - sem,
                       ymax = upper_prop + sem),
                   fill = COL_SELF, alpha = 0.12) +
      geom_line(data = crf_s, aes(x = rt_mean_ms, y = upper_prop, color = "Self"),
                linewidth = 1.2) +
      geom_point(data = crf_s, aes(x = rt_mean_ms, y = upper_prop, color = "Self"),
                 size = 3)
  }
  if (nrow(crf_st) > 0) {
    p_crf <- p_crf +
      geom_ribbon(data = crf_st,
                   aes(x = rt_mean_ms, ymin = upper_prop - sem,
                       ymax = upper_prop + sem),
                   fill = COL_STRANGER, alpha = 0.12) +
      geom_line(data = crf_st, aes(x = rt_mean_ms, y = upper_prop, color = "Stranger"),
                linewidth = 1.2) +
      geom_point(data = crf_st, aes(x = rt_mean_ms, y = upper_prop, color = "Stranger"),
                 size = 3)
  }

  p_crf <- p_crf +
    scale_color_manual(name = "Identity",
                        values = c("Self" = COL_SELF, "Stranger" = COL_STRANGER)) +
    scale_y_continuous(limits = c(-0.05, 1.05), breaks = seq(0, 1, 0.25)) +
    labs(x = "RT bin mean (ms)", y = "P(Match Key)",
         title = paste0(title_prefix, "  |  MatchKey=", toupper(mk),
                        "  |  square->", co$square, ", circle->", co$circle)) +
    theme_minimal(base_size = 10) +
    theme(legend.position = "top",
          plot.title = element_text(size = 9, face = "bold"),
          panel.grid.major = element_line(color = "gray88"),
          panel.grid.minor = element_blank())

  # ---- SPE 面板 ----
  p_spe <- ggplot() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.6)

  if (nrow(spe) > 0) {
    p_spe <- p_spe +
      geom_ribbon(data = spe,
                   aes(x = rt_mean_ms,
                       ymin = spe_upper_prop - 1.96 * spe_sem,
                       ymax = spe_upper_prop + 1.96 * spe_sem),
                   fill = COL_SPE, alpha = 0.15) +
      geom_line(data = spe, aes(x = rt_mean_ms, y = spe_upper_prop),
                color = COL_SPE, linewidth = 1.2) +
      geom_point(data = spe, aes(x = rt_mean_ms, y = spe_upper_prop),
                 color = COL_SPE, size = 3)
    # 标注每个 bin 的 SPE 值
    if (nrow(spe) <= 8) {
      p_spe <- p_spe +
        geom_text(data = spe,
                   aes(x = rt_mean_ms, y = spe_upper_prop,
                       label = sprintf("%.2f", spe_upper_prop)),
                   vjust = -1.2, size = 3.2, color = COL_SPE)
    }
  }

  # 计算 SPE 统计量用于副标题
  spe_mean <- if (nrow(spe) > 0) mean(spe$spe_upper_prop, na.rm = TRUE) else NA
  spe_subtitle <- if (!is.na(spe_mean)) {
    pos_bins <- sum(spe$spe_upper_prop > 0, na.rm = TRUE)
    sprintf("Mean SPE = %.3f  |  Positive bins = %d/%d  |  Q = %d",
            spe_mean, pos_bins, nrow(spe), n_quantiles)
  } else {
    "Insufficient data for SPE"
  }

  p_spe <- p_spe +
    labs(x = "RT bin mean (ms)", y = "Self - Stranger",
         title = "CRF-SPE", subtitle = spe_subtitle) +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold"),
          plot.subtitle = element_text(size = 8, color = "gray40"),
          panel.grid.major = element_line(color = "gray88"),
          panel.grid.minor = element_blank())

  cowplot::plot_grid(p_crf, p_spe, ncol = 2, align = "h", axis = "bt")
}

#' 群体级平均 CRF 曲线: 按 groupID 聚合
plot_group_crf <- function(all_df, n_quantiles = 5) {
  # 对每个被试计算 CRF, 然后按 group 聚合
  subjects <- all_df %>%
    group_by(groupID, subjectID) %>%
    group_keys()

  all_crf_list <- list()

  for (i in seq_len(nrow(subjects))) {
    gid <- subjects$groupID[i]
    sid <- subjects$subjectID[i]
    subj <- all_df[all_df$groupID == gid & all_df$subjectID == sid, ]
    res <- compute_spe_crf(subj, n_quantiles)

    # Self CRF
    if (nrow(res$crf_self) > 0) {
      res$crf_self$Identity <- "Self"
      res$crf_self$groupID  <- gid
      res$crf_self$subjectID <- sid
      all_crf_list[[length(all_crf_list) + 1]] <- res$crf_self
    }
    # Stranger CRF
    if (nrow(res$crf_stranger) > 0) {
      res$crf_stranger$Identity <- "Stranger"
      res$crf_stranger$groupID  <- gid
      res$crf_stranger$subjectID <- sid
      all_crf_list[[length(all_crf_list) + 1]] <- res$crf_stranger
    }
  }

  if (length(all_crf_list) == 0) {
    return(ggplot() + annotate("text", x = 1, y = 1, label = "No CRF data") + theme_void())
  }

  all_crf <- do.call(rbind, all_crf_list)

  # 按 groupID × Identity 聚合: 对每个 RT bin 位置插值, 然后按 group 取平均
  # 简化方案: 按 bin 号聚合
  group_crf_agg <- all_crf %>%
    group_by(groupID, Identity, bin) %>%
    summarise(
      rt_mean_ms  = mean(rt_mean_ms, na.rm = TRUE),
      upper_prop  = mean(upper_prop, na.rm = TRUE),
      sem         = sd(upper_prop, na.rm = TRUE) / sqrt(n()),
      n_subjects  = n(),
      .groups     = "drop"
    )

  group_crf_agg$groupID <- factor(group_crf_agg$groupID)

  ggplot(group_crf_agg, aes(x = rt_mean_ms, y = upper_prop,
                              color = Identity, fill = Identity)) +
    geom_ribbon(aes(ymin = upper_prop - sem, ymax = upper_prop + sem),
                alpha = 0.1, color = NA) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray50") +
    facet_wrap(~ groupID, ncol = 4, scales = "free_x",
               labeller = labeller(groupID = function(x) {
                 sapply(x, function(g) {
                   cond <- CONDITIONS[[as.character(g)]]
                   if (is.null(cond)) paste0("G", g)
                   else cond$label
                 })
               })) +
    scale_color_manual(values = c("Self" = COL_SELF, "Stranger" = COL_STRANGER)) +
    scale_fill_manual(values = c("Self" = COL_SELF, "Stranger" = COL_STRANGER)) +
    scale_y_continuous(limits = c(-0.05, 1.05)) +
    labs(x = "RT bin mean (ms)", y = "P(Match Key)",
         title = "Group-Level Average CRF Curves",
         subtitle = paste0("Quantiles = ", n_quantiles,
                           " | Ribbon = ±1 SEM across subjects")) +
    theme_minimal(base_size = 9) +
    theme(legend.position = "top",
          plot.title = element_text(face = "bold", size = 13),
          strip.text = element_text(face = "bold", size = 7))
}

#' 群体级平均 SPE 曲线
plot_group_spe <- function(all_df, n_quantiles = 5) {
  subjects <- all_df %>%
    group_by(groupID, subjectID) %>%
    group_keys()

  all_spe_list <- list()

  for (i in seq_len(nrow(subjects))) {
    gid <- subjects$groupID[i]
    sid <- subjects$subjectID[i]
    subj <- all_df[all_df$groupID == gid & all_df$subjectID == sid, ]
    res <- compute_spe_crf(subj, n_quantiles)
    if (nrow(res$spe_curve) > 0) {
      res$spe_curve$groupID   <- gid
      res$spe_curve$subjectID <- sid
      all_spe_list[[length(all_spe_list) + 1]] <- res$spe_curve
    }
  }

  if (length(all_spe_list) == 0) {
    return(ggplot() + annotate("text", x = 1, y = 1, label = "No SPE data") + theme_void())
  }

  all_spe <- do.call(rbind, all_spe_list)

  group_spe_agg <- all_spe %>%
    group_by(groupID, bin) %>%
    summarise(
      rt_mean_ms      = mean(rt_mean_ms, na.rm = TRUE),
      spe_upper_prop  = mean(spe_upper_prop, na.rm = TRUE),
      sem             = sd(spe_upper_prop, na.rm = TRUE) / sqrt(n()),
      n_subjects      = n(),
      .groups         = "drop"
    )

  group_spe_agg$groupID <- factor(group_spe_agg$groupID)

  ggplot(group_spe_agg, aes(x = rt_mean_ms, y = spe_upper_prop)) +
    geom_ribbon(aes(ymin = spe_upper_prop - 1.96 * sem,
                     ymax = spe_upper_prop + 1.96 * sem),
                fill = COL_SPE, alpha = 0.1) +
    geom_line(color = COL_SPE, linewidth = 1) +
    geom_point(color = COL_SPE, size = 2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    facet_wrap(~ groupID, ncol = 4, scales = "free_x",
               labeller = labeller(groupID = function(x) {
                 sapply(x, function(g) {
                   cond <- CONDITIONS[[as.character(g)]]
                   if (is.null(cond)) paste0("G", g)
                   else cond$label
                 })
               })) +
    labs(x = "RT bin mean (ms)", y = "CRF-SPE (Self - Stranger)",
         title = "Group-Level Average CRF-SPE Curves",
         subtitle = paste0("Quantiles = ", n_quantiles,
                           " | Ribbon = ±1.96 SEM (95% CI)")) +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold", size = 13),
          strip.text = element_text(face = "bold", size = 7))
}

#' CRF-SPE 指标在设计空间中的分布
plot_spe_in_design_space <- function(subject_spe_summary) {
  d <- subject_spe_summary
  d$quality_label <- d$quality

  ggplot(d, aes(x = T_ms, y = W_ms)) +
    geom_point(aes(size = abs(spe_mean) * 3 + 2,
                   fill = spe_mean),
               shape = 21, alpha = 0.8, stroke = 0.5, color = "white") +
    geom_text(aes(label = paste0("G", groupID, "-S", subjectID)),
              size = 1.8, alpha = 0.6) +
    scale_fill_gradient2(low = "#2196f3", mid = "white", high = "#ff9800",
                          midpoint = 0, name = "Mean SPE") +
    scale_size_continuous(range = c(2, 12), guide = "none") +
    labs(x = "T (ms)", y = "W (ms)",
         title = "CRF-SPE in Design Space (per Subject)",
         subtitle = "Bubble size ∝ |Mean SPE|, color = SPE direction") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold"),
          panel.grid.major = element_line(color = "gray85"))
}

#' CRF-SPE 多分位数对比 (Q=3,5,7,10)
plot_multi_quantile_spe <- function(trials, gid, sid) {
  quantiles <- c(3, 5, 7, 10)
  title_prefix <- paste0("G", gid, "-S", sid)

  spe_list <- list()
  for (q in quantiles) {
    res <- compute_spe_crf(trials, n_quantiles = q)
    if (nrow(res$spe_curve) > 0) {
      res$spe_curve$Q <- factor(paste0("Q=", q), levels = paste0("Q=", quantiles))
      spe_list[[length(spe_list) + 1]] <- res$spe_curve
    }
  }

  if (length(spe_list) == 0) {
    return(ggplot() + annotate("text", x = 1, y = 1,
                                label = "Insufficient data") + theme_void())
  }

  all_spe <- do.call(rbind, spe_list)

  ggplot(all_spe, aes(x = rt_mean_ms, y = spe_upper_prop, color = Q, fill = Q)) +
    geom_ribbon(aes(ymin = spe_upper_prop - 1.96 * spe_sem,
                     ymax = spe_upper_prop + 1.96 * spe_sem),
                alpha = 0.08, color = NA) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1") +
    labs(x = "RT bin mean (ms)", y = "CRF-SPE",
         title = paste0(title_prefix, ": CRF-SPE across Quantiles"),
         subtitle = "Comparing Q=3, 5, 7, 10 | Coarser Q = fewer bins, more stable") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", size = 11),
          legend.position = "top")
}

# ===========================================================================
# 7. 描述页面
# ===========================================================================
make_description <- function() {
  c(
    "======================================================================",
    "  CRF (Conditional Response Function) 专项分析报告 V4",
    "  Self-Matching Task - Shape-Label Association (Sui et al., 2012)",
    "======================================================================",
    "",
    "【什么是 CRF?】",
    "  CRF (Conditional Response Function) 分析将反应时 (RT) 作为条件变量,",
    "  考察被试的决策倾向如何随反应速度变化。",
    "",
    "  具体做法:",
    "  1. 将被试所有已响应的正式试次按 RT 从小到大排序",
    "  2. 等分为 Q 个 bin (默认为5)",
    "  3. 在每个 RT bin 中, 计算 P(按匹配键) = 匹配键响应次数 / 总响应次数",
    "  4. 对 Self 和 Stranger 条件分别计算, 得到两条 CRF 曲线",
    "  5. CRF-SPE = CRF_Self - CRF_Stranger",
    "",
    "【CRF 曲线的心理学含义】",
    "  - P(MatchKey) > 0.5 → 被试倾向按\"匹配\"键 (DDM 上边界)",
    "  - P(MatchKey) < 0.5 → 被试倾向按\"不匹配\"键 (DDM 下边界)",
    "  - 水平线 0.5 = 随机猜测水平",
    "  - Self 曲线高于 Stranger → 对自我相关信息更倾向匹配判断",
    "",
    "【CRF-SPE 的解读】",
    "  - SPE > 0 → Self 比 Stranger 更倾向按匹配键 (自我优先效应)",
    "  - SPE 随 RT 的变化趋势:",
    "    * 快速反应区间 (左侧) SPE 更强 → 自动化加工优势",
    "    * 慢速反应区间 (右侧) SPE 减弱 → 控制加工补偿",
    "    * 全区间持续正值 → 稳定的自我优先效应",
    "  - 95% CI 不含 0 → 该 RT 区间内 SPE 统计显著",
    "",
    "【与 DDM 建模的关系】",
    "  CRF 的 P(MatchKey) 近似 DDM 中击中上边界的概率。",
    "  SPE in CRF 反映 Self/Stranger 的决策边界偏好差异,",
    "  可映射为 DDM 的 starting point (z) 或 drift rate (v) 参数差异。",
    "",
    "【分位数 Q 的选择】",
    "  Q=3  : 粗粒度, 每 bin 数据多, 曲线稳定但分辨率低",
    "  Q=5  : 默认选择, 平衡粒度与稳定性",
    "  Q=7  : 细粒度, 能捕捉更细致的动态变化",
    "  Q=10 : 最细粒度, 每 bin 数据少, 适合被试内试次多的条件",
    "",
    "【报告结构】",
    "  1. 图表说明页 (本页)",
    "  2. 数据总览 - CRF-SPE 指标分布",
    "  3. 群体级平均 CRF 曲线 (Q=5)",
    "  4. 群体级平均 CRF-SPE 曲线 (Q=5)",
    "  5. CRF-SPE 在设计空间中的分布",
    "  6. 每个被试的个体 CRF 曲线 (一页双图: CRF + SPE)",
    "======================================================================"
  )
}

# ===========================================================================
# 8. 主流程
# ===========================================================================
generate_crf_report <- function() {
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("  CRF 专项分析 PDF 生成器 V4 (R)\n")
  cat(paste(rep("=", 60), collapse = ""), "\n\n")

  # ---- 1. 加载数据 ----
  cat("[1/6] Loading data...\n")
  all_df <- load_all_raw(RAW_DIR)
  cat(sprintf("  Loaded %d trial records\n", nrow(all_df)))

  file_manifest <- all_df %>%
    group_by(groupID, subjectID, quality) %>%
    summarise(source_file = first(source_file), .groups = "drop") %>%
    arrange(groupID, subjectID)
  cat(sprintf("  Total subjects: %d\n", nrow(file_manifest)))

  # ---- 2. 计算每个被试的 CRF-SPE 汇总指标 ----
  cat("[2/6] Computing CRF-SPE summaries...\n")

  subject_spe <- tryCatch(
    all_df %>%
      group_by(groupID, subjectID) %>%
      group_modify(~ summarize_crf_subject(.x, n_quantiles = 5)) %>%
      ungroup(),
    error = function(e) {
      message("  Using fallback split-apply...")
      splitted <- split(all_df, list(all_df$groupID, all_df$subjectID), drop = TRUE)
      res_list <- lapply(splitted, function(chunk) {
        if (nrow(chunk) > 0) summarize_crf_subject(chunk, n_quantiles = 5) else NULL
      })
      do.call(rbind, res_list[!sapply(res_list, is.null)])
    }
  )
  cat(sprintf("  %d subjects with CRF-SPE\n", nrow(subject_spe)))

  # ---- 3. 打开 PDF ----
  cat("\n[3/6] Opening PDF device...\n")
  pdf(PDF_PATH, width = 14, height = 9, onefile = TRUE,
      title = "CRF Analysis V4", family = pdf_family)

  # ---- 封面 ----
  op <- par(mar = c(0, 0, 0, 0))
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1))
  text(0.5, 0.65, "CRF Analysis Report V4",
       cex = 2.5, font = 2, col = "#1a237e")
  text(0.5, 0.50, "Conditional Response Function\nSelf-Matching Task",
       cex = 1.2, col = "gray40")
  text(0.5, 0.35, "CRF = P(MatchKey | RT bin)\nCRF-SPE = CRF_Self - CRF_Stranger",
       cex = 0.9, col = "gray50")
  text(0.5, 0.22, paste0("Generated: ", Sys.time()),
       cex = 0.7, col = "gray60")
  par(op)

  # ---- 图表说明 ----
  cat("[4/6] Generating description page...\n")
  desc <- make_description()
  op <- par(mar = c(1, 2, 1, 2))
  plot.new()
  plot.window(xlim = c(0, 100), ylim = c(0, 120))
  for (i in seq_along(desc)) {
    y_pos <- 120 - i * 1.7
    if (y_pos < 0) break
    is_heading <- grepl("^【", desc[i]) | grepl("^====", desc[i]) |
      grepl("^  CRF", desc[i])
    font_val <- if (is_heading) 2 else 1
    cex_val  <- if (grepl("^====", desc[i])) 0.65 else 0.52
    col_val  <- if (grepl("^====", desc[i])) "gray50" else "black"
    text(1, y_pos, desc[i], pos = 4, cex = cex_val, font = font_val, col = col_val)
  }
  par(op)

  # ---- 数据总览: SPE 分布 ----
  cat("  Generating CRF-SPE overview...\n")

  # SPE 均值分布直方图
  p_spe_hist <- ggplot(subject_spe, aes(x = spe_mean, fill = quality)) +
    geom_histogram(bins = 30, alpha = 0.75, color = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
    scale_fill_manual(values = QUALITY_COLORS) +
    labs(title = "CRF-SPE Mean Distribution across Subjects",
         x = "Mean CRF-SPE", y = "N subjects") +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold"))

  # SPE 正值比例分布
  subject_spe$spe_positive_ratio <- ifelse(
    !is.na(subject_spe$spe_positive_bins) & subject_spe$crf_n_quantiles > 0,
    subject_spe$spe_positive_bins / subject_spe$crf_n_quantiles, NA
  )

  p_spe_pos <- ggplot(subject_spe, aes(x = spe_positive_ratio, fill = quality)) +
    geom_histogram(bins = 11, alpha = 0.75, color = "white", boundary = 0) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    scale_fill_manual(values = QUALITY_COLORS) +
    labs(title = "Proportion of Positive-SPE Bins per Subject",
         x = "Positive bins / Total bins", y = "N subjects") +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold"))

  # SPE 按 Group 箱线图
  subject_spe_g <- subject_spe
  subject_spe_g$groupID <- factor(subject_spe_g$groupID)

  p_spe_box <- ggplot(subject_spe_g, aes(x = groupID, y = spe_mean)) +
    geom_boxplot(aes(color = groupID), outlier.size = 1.2) +
    geom_jitter(aes(color = groupID), width = 0.15, size = 1.5, alpha = 0.5) +
    scale_color_manual(values = GROUP_PALETTE, guide = "none") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = "CRF-SPE Mean by Design Group",
         x = "Group ID", y = "Mean CRF-SPE") +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold"))

  overview_top <- cowplot::plot_grid(p_spe_hist, p_spe_pos, ncol = 2, align = "h")
  overview_page <- cowplot::plot_grid(
    cowplot::ggdraw() + cowplot::draw_label(
      paste0("CRF-SPE Summary - ", nrow(subject_spe), " Subjects (Q=5)"),
      fontface = "bold", size = 13),
    overview_top, p_spe_box,
    ncol = 1, rel_heights = c(0.06, 0.47, 0.47)
  )
  print(overview_page)

  # ---- 群体级平均 CRF 曲线 ----
  cat("  Generating group-level CRF curves...\n")
  p_group_crf <- plot_group_crf(all_df, n_quantiles = 5)
  print(p_group_crf)

  # ---- 群体级平均 SPE 曲线 ----
  cat("  Generating group-level SPE curves...\n")
  p_group_spe <- plot_group_spe(all_df, n_quantiles = 5)
  print(p_group_spe)

  # ---- CRF-SPE 在设计空间中的分布 ----
  cat("  Generating design space SPE map...\n")
  p_design_spe <- plot_spe_in_design_space(subject_spe)
  print(p_design_spe)

  # ---- 每个被试的 CRF 曲线 ----
  cat("\n[5/6] Generating per-subject CRF pages...\n")
  total <- nrow(file_manifest)

  for (idx in seq_len(total)) {
    gid <- as.integer(file_manifest$groupID[idx])
    sid <- as.integer(file_manifest$subjectID[idx])

    subj <- all_df[all_df$groupID == gid & all_df$subjectID == sid, ]
    formal_n <- sum(subj$stage == "formal")
    responded_n <- sum(subj$responded & subj$stage == "formal", na.rm = TRUE)

    if (formal_n == 0) {
      cat(sprintf("  [%d/%d] G%d-S%d - skip (no formal)\n", idx, total, gid, sid))
      next
    }

    # 从汇总表取 SPE 均值
    spe_row <- subject_spe[
      subject_spe$groupID == gid & subject_spe$subjectID == sid, ]
    spe_mean_val <- if (nrow(spe_row) > 0) spe_row$spe_mean[1] else NA

    cat(sprintf("  [%d/%d] G%d-S%d (formal=%d, resp=%d, SPE=%.3f)\n",
                idx, total, gid, sid, formal_n, responded_n,
                ifelse(is.na(spe_mean_val), NA, spe_mean_val)))

    # 主体 CRF 图 (Q=5)
    p_main <- tryCatch(
      plot_crf_subject(subj, gid, sid, n_quantiles = 5),
      error = function(e) {
        ggplot() + annotate("text", x = 1, y = 1, label = paste("Error:", e$message)) +
          theme_void()
      }
    )

    # 多分位数对比图
    p_multi <- tryCatch(
      plot_multi_quantile_spe(subj, gid, sid),
      error = function(e) {
        ggplot() + annotate("text", x = 1, y = 1, label = "Multi-Q error") + theme_void()
      }
    )

    # 组合: 主体 CRF (2/3 宽度) + 多Q对比 (1/3 宽度)
    # 先让主体图占满上方, 多Q在下方
    combined <- cowplot::plot_grid(
      p_main, p_multi,
      ncol = 1, rel_heights = c(0.55, 0.45)
    )
    print(combined)
  }

  # ---- 关闭 PDF ----
  cat("\n[6/6] Closing PDF...\n")
  dev.off()

  cat("\n", paste(rep("=", 60), collapse = ""), "\n")
  cat("  CRF Analysis PDF complete!\n")
  cat("  Output: ", PDF_PATH, "\n")
  if (file.exists(PDF_PATH)) {
    size_mb <- file.info(PDF_PATH)$size / (1024 * 1024)
    cat(sprintf("  File size: %.1f MB\n", size_mb))
  }
  cat(paste(rep("=", 60), collapse = ""), "\n")
}

# ===========================================================================
# 9. 入口
# ===========================================================================
if (sys.nframe() == 0) {
  cat("\n=== CRF Analysis - Direct Execution ===\n\n")
  generate_crf_report()
} else {
  cat("\n============================================\n")
  cat("  CRF Analysis V4 Script loaded.\n")
  cat("  Run generate_crf_report() to start.\n")
  cat("  Or: Rscript CRF_Analysis_V4.R\n")
  cat("============================================\n\n")
}
