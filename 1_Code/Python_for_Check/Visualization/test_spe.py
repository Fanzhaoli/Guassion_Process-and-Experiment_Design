import csv, statistics, math, time
from pathlib import Path
from collections import defaultdict

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2: return 0.0
    m1, m2 = statistics.mean(group1), statistics.mean(group2)
    v1, v2 = statistics.variance(group1), statistics.variance(group2)
    pooled_sd = math.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    if pooled_sd == 0: return 0.0
    return (m1 - m2) / pooled_sd

SPE_DIR = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\2_Data\Real_Data\SPE_Database")
log_file = SPE_DIR / "processing_log.csv"

t0 = time.time()
with open(log_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"Loaded {len(rows)} experiments in {time.time()-t0:.2f}s")

# Test first experiment
row = rows[0]
output_file = row.get('Output_File', '')
sp_file = SPE_DIR / Path(output_file).name
print(f"Processing: {row['Pair_Key']} -> {sp_file.name}")

t1 = time.time()
with open(sp_file, 'r', encoding='utf-8-sig') as f:
    data_rows = list(csv.DictReader(f))
print(f"Loaded {len(data_rows)} rows in {time.time()-t1:.2f}s")

identity_col = None; rt_col = None; acc_col = None
for col in data_rows[0].keys():
    if col == "Label_Standardized_Identity":
        identity_col = col
        break
    if identity_col is None and col in ('Label_Origin_Identity', 'Shape_Standardized_Identity'):
        identity_col = col
for col in data_rows[0].keys():
    if col in ('RT_ms', 'RT_sec'):
        if rt_col is None or col == 'RT_ms': rt_col = col
    if col == 'ACC': acc_col = col
print(f"Identity: {identity_col}, RT: {rt_col}, ACC: {acc_col}")

t2 = time.time()
subj_data = defaultdict(lambda: {'Self': [], 'Stranger': []})
for r in data_rows:
    sid = r.get('Subject', '')
    identity = r.get(identity_col, '')
    try: rt_val = float(r.get(rt_col, ''))
    except: continue
    if identity == 'Self':
        subj_data[sid]['Self'].append(rt_val)
    elif identity == 'Stranger':
        subj_data[sid]['Stranger'].append(rt_val)
print(f"Grouped {len(subj_data)} subjects in {time.time()-t2:.2f}s")

d_vals = []
for sid in subj_data:
    self_rts = subj_data[sid]['Self']
    stranger_rts = subj_data[sid]['Stranger']
    if len(self_rts) >= 3 and len(stranger_rts) >= 3:
        d_vals.append(cohens_d(self_rts, stranger_rts))
spe_rt_d = statistics.mean(d_vals) if d_vals else None
print(f"Valid: {len(d_vals)} subjects, SPE RT d = {spe_rt_d}")
print(f"Total: {time.time()-t0:.2f}s")
