import http.server
import json
import csv
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

RAW_DIR = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\2_Data\Real_Data\UnExtact\raw")
SCRIPT_DIR = Path(__file__).parent
HTML_FILE = SCRIPT_DIR / "visualization_app.html"

CONDITIONS = {
    1: {"P": 0, "T": 0.03, "W": 0.3, "label": "G1 | P0_T30_W300"},
    2: {"P": 0, "T": 0.03, "W": 0.6, "label": "G2 | P0_T30_W600"},
    3: {"P": 120, "T": 0.03, "W": 0.6, "label": "G3 | P120_T30_W600"},
    4: {"P": 120, "T": 0.08, "W": 0.6, "label": "G4 | P120_T80_W600"},
    5: {"P": 8, "T": 0.1, "W": 1.1, "label": "G5 | P8_T100_W1100"},
    6: {"P": 120, "T": 0.5, "W": 1.5, "label": "G6 | P120_T500_W1500"},
    7: {"P": 0, "T": 0.1, "W": 1.1, "label": "G7 | P0_T100_W1100"},
    8: {"P": 120, "T": 0.03, "W": 0.8, "label": "G8 | P120_T30_W800"},
    9: {"P": 120, "T": 0.08, "W": 0.8, "label": "G9 | P120_T80_W800"},
}


def get_pairing_rules(subject_id):
    mod_result = subject_id % 4
    rules = {
        0: {"square": {"self": "f", "stranger": "j"}, "circle": {"self": "j", "stranger": "f"}},
        1: {"square": {"self": "j", "stranger": "f"}, "circle": {"self": "f", "stranger": "j"}},
        2: {"square": {"self": "j", "stranger": "f"}, "circle": {"self": "f", "stranger": "j"}},
        3: {"square": {"self": "f", "stranger": "j"}, "circle": {"self": "j", "stranger": "f"}},
    }
    return rules[mod_result]


def get_match_key(subject_id):
    match_keys = ['f', 'j', 'j', 'f']
    index = (subject_id - 1) % 4
    return match_keys[index]


def get_correct_order(subject_id):
    if subject_id % 2 == 0:
        return {"square": "self", "circle": "stranger"}
    else:
        return {"square": "stranger", "circle": "self"}


def compute_condition(shape, label, subject_id):
    correct_order = get_correct_order(subject_id)
    expected_label = correct_order[shape]
    return "Matching" if label == expected_label else "NonMatching"


def compute_experiment_params(subject_id, group_id):
    match_key = get_match_key(subject_id)
    mismatch_key = 'j' if match_key == 'f' else 'f'
    correct_order = get_correct_order(subject_id)
    cond = CONDITIONS.get(group_id, {"P": None, "T": None, "W": None, "label": f"G{group_id}"})
    return {
        "subjectID": subject_id,
        "groupID": group_id,
        "matchKey": match_key,
        "mismatchKey": mismatch_key,
        "correctOrder": correct_order,
        "P": cond["P"],
        "T": cond["T"],
        "W": cond["W"],
        "groupLabel": cond["label"],
    }


def list_files():
    files = []
    for f in sorted(RAW_DIR.glob("EXP_data_group*.csv")):
        parts = f.stem.replace("EXP_data_group", "").split("_")
        group_id = int(parts[0])
        subject_id = int(parts[1])
        cond = CONDITIONS.get(group_id, {"P": None, "T": None, "W": None, "label": f"G{group_id}"})
        files.append({
            "name": f.name,
            "groupID": group_id,
            "subjectID": subject_id,
            "P": cond["P"],
            "T": cond["T"],
            "W": cond["W"],
            "groupLabel": cond["label"],
        })
    return {"files": files, "total": len(files)}


def load_file_data(filename):
    fpath = RAW_DIR / filename
    if not fpath.exists():
        return {"error": f"File {filename} not found"}
    rows = []
    stats = {"total": 0, "formal": 0, "practice": 0, "responses": 0, "correct": 0}
    group_id = None
    subject_id = None
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if group_id is None:
                group_id = int(row['groupID'])
                subject_id = int(row['subjectID'])
            stage = row.get('stage', 'formal')
            stats["total"] += 1
            if stage == 'formal':
                stats["formal"] += 1
            else:
                stats["practice"] += 1

            rt_val = row.get('RT', 'NA')
            resp_val = row.get('Response', 'NA')
            has_resp = resp_val not in ['NA', 'nan', ''] and rt_val not in ['NA', 'nan', '']
            if has_resp and stage == 'formal':
                stats["responses"] += 1
                if int(float(row.get('Correct', 0))) == 1:
                    stats["correct"] += 1

            condition = compute_condition(
                str(row['Shape']).strip(),
                str(row['Label']).strip(),
                subject_id
            ) if subject_id else "Unknown"

            # 确定该被试的匹配键
            match_key = get_match_key(subject_id) if subject_id else 'f'
            # ResponseIsMatch: 被试的响应是否按了"匹配"键 (DDM上边界)
            response_is_match = resp_val.strip().lower() == match_key if has_resp else None

            try:
                rt = float(rt_val) if rt_val not in ['NA', 'nan', ''] else None
            except ValueError:
                rt = None
            try:
                corr = int(float(row['Correct'])) if row['Correct'] not in ['NA', 'nan', ''] else None
            except ValueError:
                corr = None

            rows.append({
                "groupID": int(row['groupID']),
                "subjectID": int(row['subjectID']),
                "stage": stage,
                "trialID": int(row['trialID']),
                "P": float(row['P']),
                "T": float(row['T']),
                "W": float(row['W']),
                "Shape": str(row['Shape']).strip(),
                "Label": str(row['Label']).strip(),
                "CorrectKey": str(row['CorrectKey']).strip(),
                "Response": resp_val,
                "RT": rt,
                "Correct": corr,
                "Condition": condition,
                "Identity": "Self" if str(row['Label']).strip() == 'self' else "Stranger",
                "MatchKey": match_key,
                "ResponseIsMatch": response_is_match,
            })

    omission_rate = (stats["formal"] - stats["responses"]) / stats["formal"] * 100 if stats["formal"] > 0 else 0
    accuracy = stats["correct"] / stats["responses"] * 100 if stats["responses"] > 0 else 0

    return {
        "filename": filename,
        "groupID": group_id,
        "subjectID": subject_id,
        "stats": {**stats, "omissionRate": round(omission_rate, 1), "accuracy": round(accuracy, 1)},
        "trials": rows,
    }


def load_all_data(group_filter=None):
    """Load all data, optionally filtered by group(s).
    group_filter: int (single group), list[int] (multiple), or None (all)."""
    all_trials = []
    summary = []
    for f in sorted(RAW_DIR.glob("EXP_data_group*.csv")):
        parts = f.stem.replace("EXP_data_group", "").split("_")
        gid = int(parts[0])
        if group_filter is not None:
            if isinstance(group_filter, (list, tuple, set)):
                if gid not in group_filter:
                    continue
            elif gid != group_filter:
                continue
        data = load_file_data(f.name)
        if "error" not in data:
            all_trials.extend(data["trials"])
            summary.append({
                "filename": f.name,
                "groupID": gid,
                "subjectID": int(parts[1]),
                **data["stats"],
            })
    return {"trials": all_trials, "summary": summary}


def verify_file_data(filename):
    """校验数据文件是否符合实验逻辑"""
    fpath = RAW_DIR / filename
    if not fpath.exists():
        return {"error": f"File {filename} not found"}

    errors = []
    group_id = None
    subject_id = None
    total_trials = 0
    total_formal = 0
    total_responses = 0
    total_correct = 0
    correct_key_errors = 0
    correct_column_errors = 0

    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if group_id is None:
                group_id = int(row['groupID'])
                subject_id = int(row['subjectID'])

            total_trials += 1
            stage = row.get('stage', 'formal')
            if stage == 'formal':
                total_formal += 1

            shape = str(row['Shape']).strip()
            label = str(row['Label']).strip()
            response = str(row['Response']).strip()
            data_correct_key = str(row['CorrectKey']).strip()
            trial_id = row['trialID']

            # 计算预期的 CorrectKey
            pairing = get_pairing_rules(subject_id)
            expected_correct_key = pairing[shape][label]

            if expected_correct_key != data_correct_key:
                correct_key_errors += 1
                errors.append({
                    "trialID": trial_id,
                    "shape": shape,
                    "label": label,
                    "type": "CorrectKey错误",
                    "expected": expected_correct_key,
                    "got": data_correct_key,
                    "response": response,
                    "detail": f"trialID={trial_id}: shape={shape}, label={label}, Expected CorrectKey={expected_correct_key}, Got={data_correct_key}"
                })

            # 计算预期的 Correct 值
            if response not in ['NA', 'nan', ''] and stage == 'formal':
                total_responses += 1
                expected_correct = 1 if response == expected_correct_key else 0
                try:
                    data_correct = int(float(row['Correct']))
                except (ValueError, KeyError):
                    data_correct = None

                if expected_correct != data_correct:
                    correct_column_errors += 1
                    errors.append({
                        "trialID": trial_id,
                        "shape": shape,
                        "label": label,
                        "type": "Correct列错误",
                        "expected": expected_correct,
                        "got": data_correct,
                        "response": response,
                        "detail": f"trialID={trial_id}: shape={shape}, label={label}, response={response}, Expected Correct={expected_correct}, Got={data_correct}"
                    })

                if data_correct == 1:
                    total_correct += 1

    cond = CONDITIONS.get(group_id, {"P": None, "T": None, "W": None, "label": f"G{group_id}"})
    match_key = get_match_key(subject_id)
    mismatch_key = 'j' if match_key == 'f' else 'f'
    correct_order = get_correct_order(subject_id)

    omission_rate = (total_formal - total_responses) / total_formal * 100 if total_formal > 0 else 0
    accuracy = total_correct / total_responses * 100 if total_responses > 0 else 0

    return {
        "filename": filename,
        "passed": len(errors) == 0,
        "totalErrors": len(errors),
        "correctKeyErrors": correct_key_errors,
        "correctColumnErrors": correct_column_errors,
        "subjectID": subject_id,
        "groupID": group_id,
        "stats": {
            "total": total_trials,
            "formal": total_formal,
            "responses": total_responses,
            "correct": total_correct,
            "omissionRate": round(omission_rate, 1),
            "accuracy": round(accuracy, 1),
        },
        "experimentParams": {
            "P": cond["P"],
            "T": cond["T"],
            "W": cond["W"],
            "matchKey": match_key,
            "mismatchKey": mismatch_key,
            "correctOrder": f"square↔{correct_order['square']}, circle↔{correct_order['circle']}",
        },
        "errors": errors[:200],  # 最多返回200条错误
        "errorCount": len(errors),
    }


def simulate_trial(subject_id, shape, label):
    pairing = get_pairing_rules(subject_id)
    correct_key = pairing[shape][label]
    condition = compute_condition(shape, label, subject_id)
    match_key = get_match_key(subject_id)
    return {
        "subjectID": subject_id,
        "shape": shape,
        "label": label,
        "correctKey": correct_key,
        "condition": condition,
        "matchKey": match_key,
        "mismatchKey": 'j' if match_key == 'f' else 'f',
        "expectedResponse": f"{'匹配' if condition == 'Matching' else '不匹配'}: 按 {correct_key.upper()} 键",
    }


class AppHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        try:
            if path == '/api/files':
                self._json(list_files())
            elif path == '/api/data/all':
                gf = params.get('group', [None])[0]
                if gf and ',' in gf:
                    groups = [int(g.strip()) for g in gf.split(',') if g.strip()]
                    self._json(load_all_data(groups))
                else:
                    self._json(load_all_data(int(gf) if gf else None))
            elif path == '/api/data/file':
                fn = params.get('name', [None])[0]
                if fn:
                    self._json(load_file_data(unquote(fn)))
                else:
                    self._error(400, "Missing name parameter")
            elif path == '/api/experiment/params':
                sid = int(params.get('subject', [1])[0])
                gid = int(params.get('group', [1])[0])
                self._json(compute_experiment_params(sid, gid))
            elif path == '/api/experiment/trial':
                sid = int(params.get('subject', [1])[0])
                shape = params.get('shape', ['square'])[0]
                label = params.get('label', ['self'])[0]
                self._json(simulate_trial(sid, shape, label))
            elif path.startswith('/api/verify'):
                fn = params.get('name', [None])[0]
                if fn:
                    self._json(verify_file_data(unquote(fn)))
                else:
                    self._error(400, "Missing name parameter")
            elif path == '/api/health':
                self._json({"status": "ok", "message": "Server is running"})
            elif path == '/' or path == '' or path == '/index.html':
                self._serve_html()
            else:
                self._serve_static(path.lstrip('/'))
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            self._error(500, str(e))

    def _serve_html(self):
        try:
            with open(HTML_FILE, 'r', encoding='utf-8') as f:
                html = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except FileNotFoundError:
            self._error(500, f"HTML file not found at {HTML_FILE}")

    def _serve_static(self, rel_path):
        file_path = SCRIPT_DIR / rel_path
        if not file_path.exists() or not file_path.is_file():
            self._error(404, f"Not found: {rel_path}")
            return
        content_types = {
            '.js': 'application/javascript',
            '.css': 'text/css',
            '.html': 'text/html',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.ico': 'image/x-icon',
        }
        ext = file_path.suffix.lower()
        content_type = content_types.get(ext, 'application/octet-stream')
        with open(file_path, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def _json(self, data):
        body = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg):
        body = json.dumps({"error": msg}).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {args[0]}")


def main():
    port = 8899
    for attempt in range(3):
        try:
            server = http.server.HTTPServer(('0.0.0.0', port), AppHandler)
            break
        except OSError:
            port += 1
    print()
    print("=" * 60)
    print(f"  Experiment Data Visualization Server")
    print(f"  Local:  http://localhost:{port}")
    print(f"  Health: http://localhost:{port}/api/health")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)
    print(f"  HTML file: {HTML_FILE}")
    print(f"  Data dir:  {RAW_DIR}")
    print("=" * 60)
    print()
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == '__main__':
    main()
