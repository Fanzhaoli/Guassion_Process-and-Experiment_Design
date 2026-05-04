"""
可重复性日志与结果追溯系统

功能:
  1. 实验参数全量记录 (日志文件 + JSON 快照)
  2. 中间结果自动保存
  3. 版本化输出目录
  4. 实验运行追溯
"""

import json
import time
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib
import pickle


class ExperimentLogger:
    """实验日志与追溯系统

    记录实验运行的完整状态:
      - 参数配置
      - 运行时间戳
      - 中间结果路径
      - 错误追踪
    """

    def __init__(self, log_dir: Path, experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            self.run_id = f"{experiment_name}_{timestamp}"
        else:
            self.run_id = f"run_{timestamp}"

        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.run_dir / 'experiment.log'
        self.snapshot_file = self.run_dir / 'snapshots.jsonl'
        self.config_file = self.run_dir / 'config.json'
        self.results_file = self.run_dir / 'results_summary.json'

        self.start_time = time.time()
        self.snapshots = []
        self.errors = []
        self.warnings = []

        self._log_header()

    def _log_header(self):
        """写入日志文件头"""
        header = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': sys.platform,
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(header, f, indent=2, ensure_ascii=False)

    def log(self, message: str, level: str = 'INFO'):
        """记录运行日志

        Args:
            message: 日志消息
            level: 日志级别 (INFO, WARN, ERROR, DEBUG)
        """
        timestamp = datetime.now().isoformat()
        elapsed = time.time() - self.start_time

        entry = {
            'timestamp': timestamp,
            'elapsed_s': round(elapsed, 3),
            'level': level,
            'message': message,
        }

        if level == 'ERROR':
            self.errors.append(entry)
        elif level == 'WARN':
            self.warnings.append(entry)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] [{level}] ({elapsed:.1f}s) {message}\n")

    def snapshot(self, stage: str, data: Dict[str, Any]):
        """保存中间结果快照

        Args:
            stage: 阶段名称 (如 'design_generation', 'ddm_fitting')
            data: 阶段产出的关键数据
        """
        snapshot_entry = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_s': round(time.time() - self.start_time, 3),
            'stage': stage,
            'data': self._make_serializable(data),
        }
        self.snapshots.append(snapshot_entry)

        with open(self.snapshot_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(snapshot_entry, ensure_ascii=False) + '\n')

    def save_dataframe(self, df: 'pd.DataFrame', name: str, format: str = 'csv'):
        """保存 DataFrame 到运行目录

        Args:
            df: 要保存的 DataFrame
            name: 文件名 (不含扩展名)
            format: 格式 ('csv' 或 'parquet')
        """
        import pandas as pd

        if format == 'csv':
            path = self.run_dir / f"{name}.csv"
            df.to_csv(path, index=False)
        elif format == 'parquet':
            path = self.run_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = self.run_dir / f"{name}.{format}"
            df.to_csv(path, index=False)

        self.log(f"数据已保存: {path}", 'INFO')
        return path

    def save_model(self, model: Any, name: str):
        """保存模型对象

        Args:
            model: 可 pickle 的模型对象
            name: 模型名称
        """
        path = self.run_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        self.log(f"模型已保存: {path}", 'INFO')
        return path

    def save_figure(self, fig, name: str, dpi: int = 200):
        """保存 matplotlib 图表"""
        path = self.run_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        self.log(f"图表已保存: {path}", 'INFO')
        return path

    def record_exception(self, exc: Exception):
        """记录异常信息"""
        tb = traceback.format_exc()
        self.log(f"异常: {exc}\n{tb}", 'ERROR')

    def finalize(self, success: bool = True):
        """完成实验运行, 保存汇总结果

        Args:
            success: 是否成功完成
        """
        elapsed_total = time.time() - self.start_time

        summary = {
            'run_id': self.run_id,
            'success': success,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_elapsed_s': round(elapsed_total, 1),
            'total_elapsed_min': round(elapsed_total / 60, 1),
            'n_snapshots': len(self.snapshots),
            'n_errors': len(self.errors),
            'n_warnings': len(self.warnings),
            'output_dir': str(self.run_dir),
            'errors': self.errors[-10:] if self.errors else [],
            'warnings': self.warnings[-10:] if self.warnings else [],
        }

        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.log(f"实验完成. 耗时: {elapsed_total:.1f}s. 成功: {success}", 'INFO')
        return summary

    def _make_serializable(self, data: Any) -> Any:
        """确保数据可JSON序列化"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_serializable(v) for v in data]
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)


class RunTracker:
    """跨运行的实验追踪

    追踪多次实验运行之间的进度和结果比较。
    """

    def __init__(self, tracker_dir: Path):
        self.tracker_dir = Path(tracker_dir)
        self.tracker_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.tracker_dir / 'all_runs.json'

    def register_run(self, summary: Dict):
        """注册新的实验运行"""
        runs = self._load_runs()
        runs.append(summary)

        with open(self.runs_file, 'w', encoding='utf-8') as f:
            json.dump(runs, f, indent=2, ensure_ascii=False)

    def get_run_history(self) -> list:
        """获取运行历史"""
        return self._load_runs()

    def get_latest_run(self) -> Optional[Dict]:
        """获取最近一次运行"""
        runs = self._load_runs()
        return runs[-1] if runs else None

    def _load_runs(self) -> list:
        if self.runs_file.exists():
            with open(self.runs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []


def hash_config(config: Dict) -> str:
    """对配置进行哈希, 用于追溯唯一性"""
    config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
