import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, request, jsonify, send_file, send_from_directory
from model_engine import run_model, compute_sweep, MODEL_VERSIONS

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_ROOT = BASE_DIR / '2_Data' / 'Generate_Data'
FIG_ROOT = BASE_DIR / '3_Figures'

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/', methods=['GET'])
def index():
    return send_from_directory(str(BASE_DIR), 'SPE_Visualizer.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    models = []
    for key, info in MODEL_VERSIONS.items():
        models.append({
            'key': key,
            'name': info['name'],
            'source': info['source'],
            'description': info['description'],
        })
    return jsonify({'models': models})

@app.route('/api/run', methods=['POST'])
def run():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体为空'}), 400

        model_key = data.get('model', 'sigmoid_ddm')
        params = data.get('params', {})

        if model_key not in MODEL_VERSIONS:
            return jsonify({
                'error': f'未知模型: {model_key}',
                'available': list(MODEL_VERSIONS.keys()),
            }), 400

        # 构建输出路径
        subfolder = f'Generate_Data_{model_key}'
        data_dir = DATA_ROOT / subfolder
        fig_dir = FIG_ROOT / subfolder

        data_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        result = run_model(model_key, params, str(data_dir), str(fig_dir))

        return jsonify({
            'success': True,
            'model': model_key,
            'model_name': MODEL_VERSIONS[model_key]['name'],
            'csv_path': result['csv_path'],
            'csv_filename': Path(result['csv_path']).name,
            'summary': result['summary'],
            'figures': result['figures'],
            'figure_filenames': [Path(f).name for f in result['figures']],
            'n_rows': result['n_rows'],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<path:filepath>', methods=['GET'])
def download_file(filepath):
    full_path = Path(filepath)
    if not full_path.is_absolute():
        full_path = BASE_DIR / filepath
    if not full_path.exists():
        return jsonify({'error': '文件不存在'}), 404
    return send_file(str(full_path), as_attachment=True)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/sweep', methods=['POST'])
def sweep():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体为空'}), 400

        # 支持单个模型或两个模型对比
        models = data.get('models', [])
        if isinstance(models, str):
            models = [models]
        if not models:
            models = ['sigmoid_ddm']

        sweep_var = data.get('sweep_var', 'T')
        w_gp = data.get('w_gp', 0.5)
        seed = data.get('seed', 42)

        # fixed_params 可以是一个（共用）或多个（每个模型不同）
        fixed_params_list = data.get('fixed_params', [{}])
        if isinstance(fixed_params_list, dict):
            fixed_params_list = [fixed_params_list] * len(models)
        while len(fixed_params_list) < len(models):
            fixed_params_list.append(fixed_params_list[-1])

        results = []
        for i, model_key in enumerate(models):
            if model_key not in MODEL_VERSIONS:
                return jsonify({'error': f'未知模型: {model_key}'}), 400
            fp = fixed_params_list[i] if i < len(fixed_params_list) else fixed_params_list[-1]
            curve = compute_sweep(model_key, sweep_var, fp, w_gp=w_gp, seed=seed)
            results.append(curve)

        return jsonify({
            'success': True,
            'sweep_var': sweep_var,
            'curves': results,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('=' * 60)
    print('SPE 模型后端服务启动')
    print('=' * 60)
    print(f'数据输出目录: {DATA_ROOT}')
    print(f'图表输出目录: {FIG_ROOT}')
    print(f'可用模型: {list(MODEL_VERSIONS.keys())}')
    print('=' * 60)
    app.run(host='127.0.0.1', port=5100, debug=True)
