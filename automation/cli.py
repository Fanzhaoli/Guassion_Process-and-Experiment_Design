"""
命令行入口 - 一键启动自动化迭代流程

用法:
  python cli.py                           默认配置运行 (从 automation/ 目录运行)
  python -m automation.cli                从项目根目录运行
  python cli.py --rounds 10              指定迭代轮数
  python cli.py --config config.json     从JSON文件加载配置
  python cli.py --quick                  快速模式 (小规模验证)
  python cli.py --profile research       研究级配置
"""

import sys
import json
import argparse
from pathlib import Path

# 设置项目根目录
# 优先使用模块运行时 Python 的工作目录；fallback 到 cli.py 所在目录的父目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 如果当前 sys.path 的第一个元素不是项目根目录，则插入
if not any(str(PROJECT_ROOT) == p for p in sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config_from_file(config_path: str) -> dict:
    """从 JSON 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='SPE 实验设计自动化迭代环境',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python -m automation.cli                          默认配置运行
  python -m automation.cli --quick                  快速验证模式 (小规模)
  python -m automation.cli --profile research       研究级配置 (完整规模)
  python -m automation.cli --config my_config.json  从配置文件加载
  python -m automation.cli --rounds 10 --subjects 50 自定义参数
        """
    )

    parser.add_argument(
        '--config', '-c', type=str, default=None,
        help='JSON 配置文件路径'
    )
    parser.add_argument(
        '--quick', '-q', action='store_true',
        help='快速验证模式 (小规模: 8被试, 5条件, 8试次/条件)'
    )
    parser.add_argument(
        '--profile', '-p', type=str, default='research',
        choices=['quick', 'standard', 'research'],
        help='运行配置级别 (quick|standard|research)'
    )
    parser.add_argument(
        '--rounds', '-r', type=int, default=None,
        help='优化迭代轮数'
    )
    parser.add_argument(
        '--subjects', '-s', type=int, default=None,
        help='被试数量'
    )
    parser.add_argument(
        '--name', '-n', type=str, default='spe_iteration',
        help='实验名称'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--no-real-data', action='store_true',
        help='不使用真实数据 (仅模拟)'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default=None,
        help='输出目录 (覆盖默认)'
    )

    return parser.parse_args()


def get_profile_config(profile: str) -> dict:
    """获取不同配置级别的预设参数"""
    profiles = {
        'quick': {
            'design': {
                'P_values': [0, 64, 120],
                'T_values': [30, 100, 500],
                'W_values': [300, 600, 1500],
                'lhs_points': 10,
            },
            'experiment': {
                'n_subjects': 8,
                'trials_per_condition': 8,
                'design_type': 'mixed',
                'block_size': 20,
            },
            'iteration': {
                'n_rounds': 2,
                'exploration_weight': 0.3,
                'use_real_data': True,
            },
        },
        'standard': {
            'design': {
                'P_values': [0, 4, 8, 16, 32, 64, 120, 150],
                'T_values': [10, 30, 50, 80, 100, 200, 350, 500],
                'W_values': [200, 300, 400, 600, 800, 1000, 1200, 1500],
                'lhs_points': 50,
            },
            'experiment': {
                'n_subjects': 30,
                'trials_per_condition': 20,
                'design_type': 'mixed',
                'block_size': 40,
            },
            'iteration': {
                'n_rounds': 5,
                'exploration_weight': 0.3,
                'use_real_data': True,
            },
        },
        'research': {
            'design': {
                'P_values': [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 120, 150],
                'T_values': [10, 20, 30, 50, 80, 100, 150, 200, 280, 350, 420, 500, 600],
                'W_values': [200, 300, 400, 500, 600, 800, 1000, 1200, 1500],
                'lhs_points': 100,
            },
            'experiment': {
                'n_subjects': 50,
                'trials_per_condition': 30,
                'design_type': 'mixed',
                'block_size': 40,
            },
            'iteration': {
                'n_rounds': 10,
                'exploration_weight': 0.3,
                'use_real_data': True,
            },
        },
    }
    return profiles.get(profile, profiles['standard'])


def main():
    """主入口"""
    args = parse_args()

    print("=" * 60)
    print("  SPE 实验设计自动化迭代环境")
    print("  Gaussian Process + DDM 实验优化 Pipeline")
    print("=" * 60)
    print()

    # 构建配置
    config = {}

    # 1. 加载预设配置
    profile_cfg = get_profile_config(args.profile)
    config.update(profile_cfg)

    # 2. 加载外部配置文件
    if args.config:
        file_cfg = load_config_from_file(args.config)
        config.update(file_cfg)

    # 3. 命令行参数覆盖
    if args.quick:
        quick_cfg = get_profile_config('quick')
        config.update(quick_cfg)
        print("[快速模式] 使用最小配置进行快速验证")

    if args.rounds is not None:
        config.setdefault('iteration', {})['n_rounds'] = args.rounds

    if args.subjects is not None:
        config.setdefault('experiment', {})['n_subjects'] = args.subjects

    if args.no_real_data:
        config.setdefault('iteration', {})['use_real_data'] = False

    # 4. 基础配置
    config['experiment_name'] = args.name
    config['seed'] = args.seed

    # 显示配置摘要
    print(f"实验名称: {args.name}")
    print(f"配置级别: {args.profile}")
    if 'design' in config:
        ds = config['design']
        print(f"设计空间: P({len(ds.get('P_values',[]))}) × T({len(ds.get('T_values',[]))}) × W({len(ds.get('W_values',[]))})")
    print(f"被试数: {config.get('experiment',{}).get('n_subjects', 'N/A')}")
    print(f"迭代轮数: {config.get('iteration',{}).get('n_rounds', 'N/A')}")
    print(f"随机种子: {args.seed}")
    print()

    # 执行 Pipeline
    from automation.pipeline import ExperimentPipeline

    pipeline = ExperimentPipeline(config=config, project_root=PROJECT_ROOT)

    # 运行
    results = pipeline.run()

    print()
    print("=" * 60)
    print("  自动化流程完成!")
    print(f"  报告目录: {pipeline.logger.run_dir}")

    # 输出关键发现
    eff = results.get('effect_analysis', {})
    spe = eff.get('synthetic_SPE', {})
    if spe:
        print(f"  SPE均值: {spe.get('SPE_ms_mean', 'N/A'):.2f} ms")
        print(f"  Cohen's d: {spe.get('cohens_d', 'N/A'):.3f}")
        print(f"  Power: {spe.get('power', 'N/A'):.3f}")
        print(f"  BF01: {spe.get('BF01', 'N/A'):.3f}")

    mc = results.get('model_comparison', {})
    if mc:
        print(f"  GP混合模型RMSE: {mc.get('hybrid_mean_rmse', 'N/A'):.4f}")
        print(f"  纯Sigmoid RMSE: {mc.get('pure_mean_rmse', 'N/A'):.4f}")

    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
