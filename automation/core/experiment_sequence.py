"""
实验序列自动生成模块
自动生成符合混合设计 3(设计空间)×2(Matching)×2(Identity) 的实验序列
实现试次随机化、区组平衡和条件分配
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import product


class ExperimentSequence:
    """实验序列生成器

    根据实验设计空间自动生成符合心理学实验规范的试次序列。
    支持混合设计、区组设计和完全随机化三种模式。
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate_mixed_design(self,
                              design_df: pd.DataFrame,
                              n_subjects: int = 30,
                              trials_per_condition: int = 20) -> pd.DataFrame:
        """生成 3×2×2 混合设计实验序列

        3 (设计空间维度: P/T/W 组合) ×
        2 (Matching: match / non-match) ×
        2 (Identity: self / stranger)

        Args:
            design_df: 设计空间数据框 (每行代表一种P/T/W组合)
            n_subjects: 被试数量
            trials_per_condition: 每个条件的试次数

        Returns:
            完整的实验序列 DataFrame
        """
        identity_levels = ['self', 'stranger']
        matching_levels = ['match', 'non-match']

        all_trials = []
        trial_id = 0

        for subj_idx in range(1, n_subjects + 1):
            for _, row in design_df.iterrows():
                P, T, W, M = int(row['P']), int(row['T']), int(row['W']), int(row['M'])
                condition_id = row.get('condition_id', f"P{P}_T{T}_W{W}")

                for identity in identity_levels:
                    for matching in matching_levels:
                        for _ in range(trials_per_condition):
                            trial_id += 1
                            all_trials.append({
                                'trial_id': trial_id,
                                'subject': subj_idx,
                                'P': P, 'T': T, 'W': W, 'M': M,
                                'condition_id': condition_id,
                                'identity': identity,
                                'matching': matching,
                                'condition_key': 1 if identity == 'self' else 0,
                                'matching_key': 1 if matching == 'match' else 0,
                            })

        trial_df = pd.DataFrame(all_trials)

        # 为每个被试的试次随机化顺序
        randomized_trials = []
        for subj in trial_df['subject'].unique():
            subj_trials = trial_df[trial_df['subject'] == subj].copy()
            subj_trials = subj_trials.sample(frac=1, random_state=self.seed + subj).reset_index(drop=True)
            subj_trials['trial_within_subject'] = range(1, len(subj_trials) + 1)
            randomized_trials.append(subj_trials)

        final_df = pd.concat(randomized_trials, ignore_index=True)

        # 添加应答映射键
        self._add_response_keys(final_df)

        return final_df

    def _add_response_keys(self, df: pd.DataFrame):
        """为匹配试次添加正确反应键映射

        参考真实实验设计: matchKey = ['f', 'j', 'j', 'f'] 的循环模式。
        Matching 试次中，self 面孔总是映射到特定键。
        """
        match_keys_f = ['f']  # self 面孔匹配到 'f'
        match_keys_j = ['j']  # self 面孔匹配到 'j'

        # 为每个试次定义正确键
        correct_keys = []
        match_keys = []

        for idx, row in df.iterrows():
            subj_mod = row['subject'] % 4

            if subj_mod in [0, 1]:
                # 这些被试: self='f', stranger='j'
                if row['identity'] == 'self':
                    ck = 'f'
                else:
                    ck = 'j'
            else:
                # 这些被试: self='j', stranger='f'
                if row['identity'] == 'self':
                    ck = 'j'
                else:
                    ck = 'f'

            correct_keys.append(ck)
            match_keys.append(ck)

        df['matchKey'] = correct_keys
        df['CorrectKey'] = match_keys

    def generate_blocked_design(self,
                                design_df: pd.DataFrame,
                                n_subjects: int = 30,
                                trials_per_condition: int = 20,
                                block_size: int = 40) -> pd.DataFrame:
        """生成区组设计实验序列

        按设计条件区组排列试次，每个区组内试次随机化。
        适用于需要控制练习效应扩散的实验设计。

        Args:
            design_df: 设计空间数据框
            n_subjects: 被试数量
            trials_per_condition: 每个条件的试次数
            block_size: 每个区组的试次数

        Returns:
            区组化的实验序列 DataFrame
        """
        full_df = self.generate_mixed_design(
            design_df, n_subjects, trials_per_condition
        )

        blocked_trials = []
        for subj in full_df['subject'].unique():
            subj_df = full_df[full_df['subject'] == subj].copy()
            n_trials = len(subj_df)
            n_blocks = (n_trials + block_size - 1) // block_size

            for block in range(n_blocks):
                start = block * block_size
                end = min(start + block_size, n_trials)
                block_trials = subj_df.iloc[start:end].sample(
                    frac=1, random_state=self.seed + subj * 100 + block
                )
                block_trials['block'] = block + 1
                blocked_trials.append(block_trials)

        result = pd.concat(blocked_trials, ignore_index=True)
        result['trial_within_block'] = result.groupby(
            ['subject', 'block']
        ).cumcount() + 1

        return result

    def generate_latin_square_sequence(self,
                                       design_df: pd.DataFrame,
                                       n_subjects: int = 30,
                                       trials_per_condition: int = 20) -> pd.DataFrame:
        """使用拉丁方设计平衡顺序效应

        适用场景: 当需要对设计条件进行完全平衡的重复测量设计时。
        """
        n_conditions = len(design_df)

        # 构建拉丁方阵
        latin_square = np.zeros((n_conditions, n_conditions), dtype=int)
        for i in range(n_conditions):
            for j in range(n_conditions):
                latin_square[i, j] = (i + j) % n_conditions

        all_trials = []
        trial_id = 0

        for subj_idx in range(1, n_subjects + 1):
            # 每个被试使用不同的行
            order_row = latin_square[(subj_idx - 1) % n_conditions]

            conditions_in_order = [design_df.iloc[idx] for idx in order_row]

            for order_pos, cond in enumerate(conditions_in_order):
                P, T, W, M = int(cond['P']), int(cond['T']), int(cond['W']), int(cond['M'])
                condition_id = cond.get('condition_id', f"P{P}_T{T}_W{W}")

                for identity in ['self', 'stranger']:
                    for matching in ['match', 'non-match']:
                        for _ in range(trials_per_condition):
                            trial_id += 1
                            all_trials.append({
                                'trial_id': trial_id,
                                'subject': subj_idx,
                                'order_position': order_pos + 1,
                                'P': P, 'T': T, 'W': W, 'M': M,
                                'condition_id': condition_id,
                                'identity': identity,
                                'matching': matching,
                                'condition_key': 1 if identity == 'self' else 0,
                                'matching_key': 1 if matching == 'match' else 0,
                            })

        result = pd.DataFrame(all_trials)
        self._add_response_keys(result)
        return result
