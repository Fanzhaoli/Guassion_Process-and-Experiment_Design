import pandas as pd
from pathlib import Path

RAW = Path(r'd:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\2_Data\Real_Data\UnExtact\raw')
csvs = sorted(RAW.glob('EXP_data_group*.csv'))

print('=== 逐被试: CorrectKey for Match vs Mismatch pairs ===')
print()

for f in csvs:
    df = pd.read_csv(f)
    df_f = df[df['stage'] == 'formal'].dropna(subset=['CorrectKey'])
    if len(df_f) == 0:
        continue
    gid = int(df_f['groupID'].iloc[0])
    sid = int(df_f['subjectID'].iloc[0])
    parity = 'odd' if sid % 2 == 1 else 'even'
    
    # Odd subjects: Match = (circle,self)+(square,stranger)
    # Even subjects: Match = (square,self)+(circle,stranger)
    if sid % 2 == 1:
        ck_match1 = df_f[(df_f['Shape']=='circle') & (df_f['Label']=='self')]['CorrectKey'].iloc[0]
        ck_match2 = df_f[(df_f['Shape']=='square') & (df_f['Label']=='stranger')]['CorrectKey'].iloc[0]
        ck_mismatch1 = df_f[(df_f['Shape']=='circle') & (df_f['Label']=='stranger')]['CorrectKey'].iloc[0]
        ck_mismatch2 = df_f[(df_f['Shape']=='square') & (df_f['Label']=='self')]['CorrectKey'].iloc[0]
    else:
        ck_match1 = df_f[(df_f['Shape']=='square') & (df_f['Label']=='self')]['CorrectKey'].iloc[0]
        ck_match2 = df_f[(df_f['Shape']=='circle') & (df_f['Label']=='stranger')]['CorrectKey'].iloc[0]
        ck_mismatch1 = df_f[(df_f['Shape']=='circle') & (df_f['Label']=='self')]['CorrectKey'].iloc[0]
        ck_mismatch2 = df_f[(df_f['Shape']=='square') & (df_f['Label']=='stranger')]['CorrectKey'].iloc[0]
    
    match_keys_same = (ck_match1 == ck_match2)
    mismatch_keys_same = (ck_mismatch1 == ck_mismatch2)
    match_vs_mismatch = 'DIFFER' if ck_match1 != ck_mismatch1 else 'SAME!'
    
    print(f'G{gid:>1} S{sid:>2}  {parity:>4}  |  Match: {ck_match1} {ck_match2}  Mismatch: {ck_mismatch1} {ck_mismatch2}  |  Match_keys_same={match_keys_same}  Mismatch_keys_same={mismatch_keys_same}  Match_vs_Mismatch={match_vs_mismatch}')
