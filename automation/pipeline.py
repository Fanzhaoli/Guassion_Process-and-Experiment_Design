"""
自动化 Pipeline 编排引擎

编排实验设计优化的完整流程:
  1. 实验设计空间生成
  2. 实验序列生成
  3. 数据模拟 (Sigmoid + GP + DDM)
  4. DDM 参数估计 (EZ-diffusion)
  5. 效应量计算 (Cohen's d, G-power, BF01)
  6. GP 模型训练与更新
  7. 边界条件探索
  8. 下一轮设计点推荐
  9. 模型比较与报告生成

可以单轮运行或迭代多轮优化。
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Optional, List, Tuple
import time
import json

# 确保可以导入 core 模块
_automation_dir = Path(__file__).resolve().parent
if str(_automation_dir) not in sys.path:
    sys.path.insert(0, str(_automation_dir))

from core.design_space import DesignSpace, DesignSpaceSummary
from core.experiment_sequence import ExperimentSequence
from core.sigmoid_model import PureSigmoidModel, compute_v_s2, compute_a_s2
from core.gp_model import GPHybridModel, GPModel
from core.ddm_engine import DDMSimulator
from core.ez_diffusion import (ez_diffusion, ez_diffusion_from_data,
                                estimate_condition_level_params,
                                subject_level_ez_diffusion)
from core.effect_size import (cohens_d_paired, g_power_analysis,
                               bayes_factor_paired, compute_spe_metrics,
                               compute_condition_level_spe)
from core.generative_model import GenerativeModel, ModelComparison
from core.logger import ExperimentLogger, RunTracker, hash_config


class ExperimentPipeline:
    """实验优化自动化 Pipeline

    编排完整的实验设计→数据生成→模型拟合→优化迭代流程。

    使用方法:
        pipeline = ExperimentPipeline(config)
        results = pipeline.run()
    """

    def __init__(self,
                 config: Dict = None,
                 project_root: Path = None):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(project_root)
        self.config = self._default_config()
        if config:
            self.config.update(config)

        self.log_dir = self.project_root / 'automation' / 'logs'
        self.output_dir = self.project_root / 'automation' / 'output'
        self.figure_dir = self.project_root / '3_Figures' / 'automation_output'
        self.report_dir = self.project_root / '4_Reports'

        for d in [self.output_dir, self.figure_dir, self.report_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = None
        self.run_tracker = RunTracker(self.log_dir)
        self.generative_model = None

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'experiment_name': 'spe_iteration',

            # 设计空间配置
            'design': {
                'P_values': [0, 4, 8, 16, 32, 64, 120, 150],
                'T_values': [10, 30, 50, 80, 100, 200, 350, 500],
                'W_values': [200, 300, 400, 600, 800, 1000, 1200, 1500],
                'lhs_points': 50,
            },

            # 实验序列配置
            'experiment': {
                'n_subjects': 30,
                'trials_per_condition': 20,
                'design_type': 'mixed',  # 'mixed', 'blocked', 'latin_square'
                'block_size': 40,
            },

            # 模型参数配置
            'model': {
                'w_gp': 0.5,
                'v_noise': 0.3,
                'a_noise': 0.25,
                'kernel_type': 'rbf',
            },

            # 迭代配置
            'iteration': {
                'n_rounds': 5,
                'exploration_weight': 0.3,
                'use_real_data': True,
                'real_data_path': None,  # 自动使用默认路径
            },

            # 种子
            'seed': 42,
        }

    def run(self) -> Dict:
        """执行完整的自动化流程

        Returns:
            dict: 包含所有运行结果的字典
        """
        self.logger = ExperimentLogger(self.log_dir, self.config['experiment_name'])
        self.logger.log("=" * 60, 'INFO')
        self.logger.log("实验优化自动化流程启动", 'INFO')
        self.logger.log(f"配置: {json.dumps(self._serializable_config(), indent=2, ensure_ascii=False)}", 'INFO')

        np.random.seed(self.config['seed'])

        try:
            # Stage 1: 初始化组件
            self.logger.snapshot('init', {'status': 'starting'})
            self._init_components()

            # Stage 2: 生成实验设计空间
            design_df = self._stage_generate_design()

            # Stage 3: 生成实验序列
            trial_sequence = self._stage_generate_sequence(design_df)

            # Stage 4: 数据准备 (加载真实数据)
            real_data = self._stage_load_real_data()

            # Stage 5: 初始化生成模型并训练
            self._stage_train_models(design_df, real_data)

            # Stage 6: 生成模拟数据
            synthetic_data = self._stage_generate_synthetic(design_df)

            # Stage 7: 效应量分析
            effect_results = self._stage_effect_analysis(synthetic_data, real_data)

            # Stage 8: 迭代优化与边界探索
            optimization_results = self._stage_iterate_optimization(design_df, real_data)

            # Stage 9: 模型比较
            comparison_results = self._stage_model_comparison(design_df, real_data)

            # Stage 10: 生成报告
            final_summary = self._stage_generate_report(
                design_df, synthetic_data, real_data,
                effect_results, optimization_results, comparison_results,
            )

            self.logger.finalize(success=True)
            return final_summary

        except Exception as e:
            self.logger.record_exception(e)
            self.logger.finalize(success=False)
            raise

    def _init_components(self):
        """初始化所有组件"""
        self.logger.log("初始化组件...", 'INFO')

        self.design_space = DesignSpace(seed=self.config['seed'])
        self.sequence_gen = ExperimentSequence(seed=self.config['seed'])
        self.ddm_simulator = DDMSimulator(seed=self.config['seed'])

        self.generative_model = GenerativeModel(
            w_gp=self.config['model']['w_gp'],
            v_noise=self.config['model']['v_noise'],
            a_noise=self.config['model']['a_noise'],
            seed=self.config['seed'],
        )

        self.logger.log("组件初始化完成", 'INFO')

    def _stage_generate_design(self) -> pd.DataFrame:
        """Stage 2: 生成实验设计空间"""
        self.logger.log("生成实验设计空间...", 'INFO')
        stage_start = time.time()

        design_cfg = self.config['design']
        design_df = self.design_space.generate_grid(
            P_values=design_cfg['P_values'],
            T_values=design_cfg['T_values'],
            W_values=design_cfg['W_values'],
        )

        lhs_df = self.design_space.generate_lhs(design_cfg['lhs_points'])

        summary = DesignSpaceSummary.summarize(design_df)
        self.logger.snapshot('design_generation', {
            'n_grid_conditions': len(design_df),
            'n_lhs_points': len(lhs_df),
            'summary': summary,
        })

        self.logger.save_dataframe(design_df, 'design_grid')
        self.logger.save_dataframe(lhs_df, 'design_lhs')

        self.logger.log(f"设计空间生成完成: {len(design_df)} 个网格点, "
                        f"{len(lhs_df)} 个LHS点 "
                        f"(耗时 {time.time()-stage_start:.1f}s)", 'INFO')

        return design_df

    def _stage_generate_sequence(self, design_df: pd.DataFrame) -> pd.DataFrame:
        """Stage 3: 生成实验序列"""
        self.logger.log("生成实验序列...", 'INFO')
        stage_start = time.time()

        exp_cfg = self.config['experiment']
        design_type = exp_cfg['design_type']

        if design_type == 'blocked':
            sequence = self.sequence_gen.generate_blocked_design(
                design_df, exp_cfg['n_subjects'],
                exp_cfg['trials_per_condition'], exp_cfg['block_size'],
            )
        elif design_type == 'latin_square':
            sequence = self.sequence_gen.generate_latin_square_sequence(
                design_df, exp_cfg['n_subjects'],
                exp_cfg['trials_per_condition'],
            )
        else:
            sequence = self.sequence_gen.generate_mixed_design(
                design_df, exp_cfg['n_subjects'],
                exp_cfg['trials_per_condition'],
            )

        total_trials = len(sequence)
        n_subjects = sequence['subject'].nunique()

        self.logger.snapshot('sequence_generation', {
            'total_trials': total_trials,
            'n_subjects': n_subjects,
            'design_type': design_type,
        })

        self.logger.save_dataframe(sequence, 'experiment_sequence')

        self.logger.log(f"实验序列生成完成: {total_trials} 试次, {n_subjects} 被试 "
                        f"(耗时 {time.time()-stage_start:.1f}s)", 'INFO')

        return sequence

    def _stage_load_real_data(self) -> pd.DataFrame:
        """Stage 4: 加载真实实验数据"""
        self.logger.log("加载真实实验数据...", 'INFO')
        stage_start = time.time()

        real_data_path = self.config['iteration'].get('real_data_path')
        if real_data_path is None:
            real_data_path = self.project_root / '2_Data' / 'Real_Data' / 'EXP_data_combined.csv'

        real_data_path = Path(real_data_path)

        if real_data_path.exists():
            real_data = pd.read_csv(real_data_path)

            # 根据 AGENTS.md 的说明，T和W在真实数据中是秒，需转换为 ms
            if 'T' in real_data.columns:
                t_max = real_data['T'].max()
                if t_max < 60:  # 判断是秒级
                    real_data['T'] = (real_data['T'] * 1000).astype(int)
                    real_data['W'] = (real_data['W'] * 1000).astype(int)
                    self.logger.log("T/W 已从秒转换为毫秒", 'INFO')

            # 添加 label 列
            if 'label' not in real_data.columns and 'condition' in real_data.columns:
                real_data['label'] = real_data['condition'].map({1: 'self', 0: 'stranger'})

            self.logger.snapshot('real_data_loaded', {
                'path': str(real_data_path),
                'n_rows': len(real_data),
                'columns': list(real_data.columns),
            })

            self.logger.log(f"真实数据加载完成: {len(real_data)} 行 "
                            f"(耗时 {time.time()-stage_start:.1f}s)", 'INFO')
            return real_data
        else:
            self.logger.log(f"真实数据文件不存在: {real_data_path}, 跳过", 'WARN')
            return None

    def _stage_train_models(self, design_df: pd.DataFrame, real_data: pd.DataFrame):
        """Stage 5: 训练模型"""
        self.logger.log("训练生成模型...", 'INFO')
        stage_start = time.time()

        use_real = self.config['iteration']['use_real_data'] and real_data is not None

        if use_real:
            self.generative_model.fit_to_real_data(real_data, design_df)
            self.logger.log("模型已用真实数据进行训练", 'INFO')
        else:
            self.logger.log("跳过真实数据训练 (使用默认参数)", 'INFO')

        self.logger.snapshot('model_training', {
            'use_real_data': use_real,
            'is_trained': self.generative_model.is_trained,
        })

        self.logger.log(f"模型训练完成 (耗时 {time.time()-stage_start:.1f}s)", 'INFO')

    def _stage_generate_synthetic(self, design_df: pd.DataFrame) -> pd.DataFrame:
        """Stage 6: 生成模拟行为数据"""
        self.logger.log("生成模拟行为数据...", 'INFO')
        stage_start = time.time()

        exp_cfg = self.config['experiment']
        synthetic = self.generative_model.generate_dataset(
            design_df,
            n_subjects=exp_cfg['n_subjects'],
            trials_per_condition=exp_cfg['trials_per_condition'],
        )

        self.logger.save_dataframe(synthetic, 'synthetic_behavior_data')

        self.logger.snapshot('synthetic_generation', {
            'n_rows': len(synthetic),
            'n_subjects': synthetic['subject'].nunique(),
            'n_omissions': synthetic['omission'].sum() if 'omission' in synthetic.columns else 0,
        })

        self.logger.log(f"模拟数据生成完成: {len(synthetic)} 行 "
                        f"(耗时 {time.time()-stage_start:.1f}s)", 'INFO')

        return synthetic

    def _stage_effect_analysis(self, synthetic: pd.DataFrame,
                                real_data: pd.DataFrame) -> Dict:
        """Stage 7: 效应量分析"""
        self.logger.log("效应量分析...", 'INFO')
        stage_start = time.time()

        results = {}

        # 被试层级 SPE 分析 (生成数据)
        synth_valid = synthetic[synthetic['RT'].notna() & (synthetic['RT'] > 0)]
        if len(synth_valid) > 0:
            subj_spe = subject_level_ez_diffusion(synth_valid)
            if 'SPE_ms' in subj_spe.columns:
                spe_values = subj_spe['SPE_ms'].dropna().values
                if len(spe_values) > 3:
                    # 单样本 t 检验 SPE 是否显著
                    from scipy import stats
                    t, p = stats.ttest_1samp(spe_values, 0)
                    d = np.mean(spe_values) / np.std(spe_values, ddof=1)

                    # Bayes Factor
                    self_rt_vals = synth_valid[synth_valid['label']=='self'].groupby('subject')['RT'].mean().values if 'label' in synth_valid.columns else synth_valid[synth_valid['Label']=='self'].groupby('subject')['RT'].mean().values
                    stranger_rt_vals = synth_valid[synth_valid['label']=='stranger'].groupby('subject')['RT'].mean().values if 'label' in synth_valid.columns else synth_valid[synth_valid['Label']=='stranger'].groupby('subject')['RT'].mean().values
                    bf = bayes_factor_paired(self_rt_vals, stranger_rt_vals)

                    # G-power
                    power = g_power_analysis(d, len(spe_values), test_type='paired')

                    results['synthetic_SPE'] = {
                        'SPE_ms_mean': float(np.mean(spe_values)),
                        'SPE_ms_std': float(np.std(spe_values, ddof=1)),
                        'cohens_d': float(d),
                        't_stat': float(t),
                        'p_value': float(p),
                        'n_subjects': len(spe_values),
                        'power': power['power'],
                        'BF01': bf['BF01'],
                        'BF10': bf['BF10'],
                        'interpretation': bf['interpretation'],
                    }

        # 真实数据效应量 (如果可用)
        if real_data is not None and len(real_data) > 0:
            real_valid = real_data[real_data['RT'].notna() & (real_data['RT'] > 0)]
            if len(real_valid) > 0 and 'label' in real_valid.columns:
                self_subjects = real_valid[real_valid['label']=='self'].groupby('subject')['RT'].mean().values
                stranger_subjects = real_valid[real_valid['label']=='stranger'].groupby('subject')['RT'].mean().values

                min_len = min(len(self_subjects), len(stranger_subjects))
                if min_len > 3:
                    d_result = cohens_d_paired(self_subjects[:min_len], stranger_subjects[:min_len])
                    bf_result = bayes_factor_paired(self_subjects[:min_len], stranger_subjects[:min_len])

                    results['real_SPE'] = {
                        'SPE_ms': float(np.mean(self_subjects[:min_len] - stranger_subjects[:min_len]) * 1000),
                        'cohens_d': d_result['d'],
                        'p_value': d_result['p_value'],
                        'BF01': bf_result['BF01'],
                        'interpretation': bf_result['interpretation'],
                    }

        self.logger.snapshot('effect_analysis', results)

        # 保存
        with open(self.logger.run_dir / 'effect_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.log(f"效应量分析完成 (耗时 {time.time()-stage_start:.1f}s)", 'INFO')
        return results

    def _stage_iterate_optimization(self, design_df: pd.DataFrame,
                                     real_data: pd.DataFrame) -> Dict:
        """Stage 8: 迭代优化与边界探索"""
        self.logger.log("迭代优化探索...", 'INFO')
        stage_start = time.time()

        iter_cfg = self.config['iteration']
        n_rounds = iter_cfg['n_rounds']

        # 边界条件探索
        boundaries = self.generative_model.explore_spe_boundaries()

        # GP 不确定性分析
        if self.generative_model.is_trained:
            design_points = design_df.copy()
            Pn, Tn, Wn = self.design_space.normalize_df(design_points)
            X_eval = np.column_stack([Pn, Tn, Wn])

            mu_v, std_v = self.generative_model.gp_hybrid.gp_v.predict(X_eval, return_std=True)
            mu_a, std_a = self.generative_model.gp_hybrid.gp_a.predict(X_eval, return_std=True)

            uncertainty = {
                'v_gp_mean_std': float(np.mean(std_v)),
                'v_gp_max_std': float(np.max(std_v)),
                'a_gp_mean_std': float(np.mean(std_a)),
                'a_gp_max_std': float(np.max(std_a)),
            }
        else:
            uncertainty = {}

        # 识别 SPE 效应最强的区域
        strong_spe = boundaries['strong_spe_region']
        n_strong = len(strong_spe) if hasattr(strong_spe, '__len__') else 0

        optimization = {
            'n_rounds': n_rounds,
            'uncertainty': uncertainty,
            'n_strong_spe_regions': n_strong,
            'spe_boundaries': {
                'strong': len(boundaries['boundary_df'][boundaries['boundary_df']['spe_region'] == 'strong']),
                'moderate': len(boundaries['boundary_df'][boundaries['boundary_df']['spe_region'] == 'moderate']),
                'weak': len(boundaries['boundary_df'][boundaries['boundary_df']['spe_region'] == 'weak']),
                'none': len(boundaries['boundary_df'][boundaries['boundary_df']['spe_region'] == 'none']),
            },
        }

        self.logger.snapshot('optimization', optimization)

        self.logger.log(f"迭代优化完成 (耗时 {time.time()-stage_start:.1f}s)", 'INFO')
        return optimization

    def _stage_model_comparison(self, design_df: pd.DataFrame,
                                real_data: pd.DataFrame) -> Dict:
        """Stage 9: 模型比较"""
        self.logger.log("模型比较...", 'INFO')
        stage_start = time.time()

        if real_data is None or len(real_data) == 0:
            self.logger.log("无真实数据, 跳过模型比较", 'INFO')
            return {}

        pure_model = PureSigmoidModel(seed=self.config['seed'])
        comparison = ModelComparison(pure_model, self.generative_model)

        comp_df = comparison.compare_condition_predictions(design_df, real_data)
        summary = comparison.summary(comp_df)

        self.logger.save_dataframe(comp_df, 'model_comparison')

        self.logger.snapshot('model_comparison', summary)
        self.logger.log(f"模型比较完成 (耗时 {time.time()-stage_start:.1f}s)", 'INFO')

        return summary

    def _stage_generate_report(self,
                                design_df: pd.DataFrame,
                                synthetic: pd.DataFrame,
                                real_data: pd.DataFrame,
                                effect_results: Dict,
                                optimization: Dict,
                                comparison: Dict) -> Dict:
        """Stage 10: 生成最终报告"""
        self.logger.log("生成最终报告...", 'INFO')
        stage_start = time.time()

        report = {
            'run_id': self.logger.run_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_summary': {
                'design_space_size': len(design_df),
                'n_subjects': self.config['experiment']['n_subjects'],
                'w_gp': self.config['model']['w_gp'],
            },

            # 设计空间摘要
            'design_summary': DesignSpaceSummary.summarize(design_df),

            # 生成数据统计
            'synthetic_summary': {
                'n_rows': len(synthetic),
                'mean_rt_ms': float(synthetic[synthetic['RT'].notna()]['RT'].mean() * 1000)
                if 'RT' in synthetic.columns else None,
                'omission_rate': float(synthetic['omission'].mean())
                if 'omission' in synthetic.columns else None,
            },

            # 效应量分析
            'effect_analysis': effect_results,

            # 优化结果
            'optimization': optimization,

            # 模型比较
            'model_comparison': comparison,

            # 运行统计
            'run_stats': {
                'total_elapsed_s': round(time.time() - self.logger.start_time, 1),
                'python_version': sys.version,
            },

            'all_outputs_path': str(self.logger.run_dir),
        }

        # 保存报告
        report_path = self.logger.run_dir / 'final_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # 生成 Markdown 报告
        md_report = self._generate_markdown_report(report)
        md_path = self.logger.run_dir / 'final_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)

        # 注册到追踪器
        self.run_tracker.register_run(report)

        self.logger.log(f"报告生成完成 (耗时 {time.time()-stage_start:.1f}s)", 'INFO')
        print(f"\n{'='*60}")
        print(f"实验完成!")
        print(f"报告位置: {report_path}")
        print(f"Markdown报告: {md_path}")
        print(f"所有输出: {self.logger.run_dir}")
        print(f"{'='*60}")

        return report

    def _generate_markdown_report(self, report: Dict) -> str:
        """生成 Markdown 格式的实验报告"""
        md = []
        md.append("# SPE 实验设计自动化迭代报告\n")
        md.append(f"**Run ID**: `{report.get('run_id', 'N/A')}`\n")
        md.append(f"**时间**: {report.get('timestamp', 'N/A')}\n\n")

        md.append("## 设计空间摘要\n")
        ds = report.get('design_summary', {})
        if ds:
            md.append(f"- 条件总数: {ds.get('n_conditions', 'N/A')}\n")
            for dim in ['P', 'T', 'W', 'M']:
                info = ds.get(dim, {})
                md.append(f"- **{dim}**: 范围 [{info.get('min', '?')}, {info.get('max', '?')}], "
                           f"唯一值: {info.get('unique', '?')}\n")
        md.append("\n")

        md.append("## 效应量分析\n")
        eff = report.get('effect_analysis', {})
        spe_synth = eff.get('synthetic_SPE', {})
        if spe_synth:
            md.append("### 模拟数据 SPE\n")
            md.append(f"- SPE均值: {spe_synth.get('SPE_ms_mean', 'N/A'):.2f} ms\n")
            md.append(f"- Cohen's d: {spe_synth.get('cohens_d', 'N/A'):.3f}\n")
            md.append(f"- p-value: {spe_synth.get('p_value', 'N/A'):.4f}\n")
            md.append(f"- Statistical Power: {spe_synth.get('power', 'N/A'):.3f}\n")
            md.append(f"- BF01: {spe_synth.get('BF01', 'N/A'):.3f}\n")
            md.append(f"- 解释: {spe_synth.get('interpretation', 'N/A')}\n\n")

        spe_real = eff.get('real_SPE', {})
        if spe_real:
            md.append("### 真实数据 SPE\n")
            md.append(f"- Cohen's d: {spe_real.get('cohens_d', 'N/A'):.3f}\n")
            md.append(f"- BF01: {spe_real.get('BF01', 'N/A'):.3f}\n")
            md.append(f"- 解释: {spe_real.get('interpretation', 'N/A')}\n\n")

        md.append("## 模型比较\n")
        mc = report.get('model_comparison', {})
        if mc:
            md.append(f"- Hybrid GP 平均 RMSE: {mc.get('hybrid_mean_rmse', 'N/A'):.4f}\n")
            md.append(f"- Pure Sigmoid 平均 RMSE: {mc.get('pure_mean_rmse', 'N/A'):.4f}\n")
            md.append(f"- Hybrid 更优的比例: {mc.get('hybrid_better_pct', 'N/A'):.1f}%\n")
            md.append(f"- 改进的 p-value: {mc.get('improvement_p_value', 'N/A'):.4f}\n\n")

        md.append("## 运行统计\n")
        md.append(f"- 总耗时: {report.get('run_stats', {}).get('total_elapsed_s', 'N/A')} 秒\n")
        md.append(f"- 输出目录: `{report.get('all_outputs_path', 'N/A')}`\n")

        return ''.join(md)

    def _serializable_config(self) -> Dict:
        """生成可序列化的配置"""
        cfg = self.config.copy()
        return cfg
