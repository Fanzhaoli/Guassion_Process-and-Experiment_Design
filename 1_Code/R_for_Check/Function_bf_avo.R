bf_aov <- function(data, dv='acc') {
  set.seed(1234)
  
  subj_num <- base::unique(data$subj_idx) # 每个被试的编号（按数据中顺序排列）
  n <- base::length(unique(data$subj_idx)) # 获取被试数量
  
  bfs_int <- base::rep(1, length(subj_num))
  bfs_a <- base::rep(1, length(subj_num))
  bfs_diff <- base::rep(1, length(subj_num))
  
  for (i in base::seq_along(subj_num)) {
    if (i == 1) {
      next # 由于一个被试不能正确计算贝叶斯因子，所以当i等于1时，跳过
    }
    # 循环核心修正：筛选“前i个被试”的数据（而非全量数据）
    data_i <- data %>% dplyr::filter(subj_idx %in% subj_num[1:i])
    
    # 使用dv参数构建公式, 基础公式包含随机斜率（为了与JASP一致）
    dv_formula <- stats::as.formula(base::paste(dv, "~ 1 + subj_idx + difficulty:subj_idx + association:subj_idx"))
    
    # 零模型（完全保留你的公式，数据改为筛选后的前i个被试）
    null_model <- BayesFactor::lmBF(
      dv_formula,
      data = data_i,  # 关键：用前i个被试的数据
      whichRandom = "subj_idx",
      rscaleFixed = 0.5,
      rscaleRandom = 1,
      iteration = 10000
    )
    
    # 仅包含 difficulty 主效应（数据改为data_i）
    model_difficulty <- BayesFactor::lmBF(
      update(dv_formula, paste(". ~ . + difficulty")),
      data = data_i,
      whichRandom = "subj_idx",
      rscaleFixed = 0.5,
      rscaleRandom = 1,
      iteration = 10000
    )
    
    # 仅包含 association 主效应（数据改为data_i）
    model_association <- BayesFactor::lmBF(
      update(dv_formula, paste(". ~ . + association")),
      data = data_i,
      whichRandom = "subj_idx",
      rscaleFixed = 0.5,
      rscaleRandom = 1,
      iteration = 10000
    )
    
    # 包含两个主效应（数据改为data_i）
    model_main <- BayesFactor::lmBF(
      update(dv_formula, paste(". ~ . + difficulty + association")),
      data = data_i,
      whichRandom = "subj_idx",
      rscaleFixed = 0.5,
      rscaleRandom = 1,
      iteration = 10000
    )
    
    # 包含主效应和交互作用（数据改为data_i）
    model_int <- BayesFactor::lmBF(
      update(dv_formula, paste(". ~ . + difficulty * association")),
      data = data_i,
      whichRandom = "subj_idx",
      rscaleFixed = 0.5,
      rscaleRandom = 1,
      iteration = 10000
    )
    
    # 核心修正：用extractBF提取BF数值（避免S4类型报错）
    bfs_int[i] <- BayesFactor::extractBF(model_int / model_main)$bf
    bfs_a[i] <- BayesFactor::extractBF(model_main / model_difficulty)$bf
    bfs_diff[i] <- BayesFactor::extractBF(model_main / model_association)$bf
  }
  
  aov_output <- tibble::tibble(
    bfs_a = bfs_a,
    bfs_diff = bfs_diff,
    bfs_int = bfs_int
  )
  
  print(aov_output)
  return(aov_output)
}