import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = Path(r"d:\GitHub_programe\GitHub\Guassion-Process-Experiment-Design\2_Data\Real_Data\UnExtact\raw")

CONDITIONS = {
    1: {"P": 0, "T": 0.03, "W": 0.3},
    2: {"P": 0, "T": 0.03, "W": 0.6},
    3: {"P": 120, "T": 0.03, "W": 0.6},
    4: {"P": 120, "T": 0.08, "W": 0.6},
    5: {"P": 8, "T": 0.1, "W": 1.1},
    6: {"P": 120, "T": 0.5, "W": 1.5},
    7: {"P": 0, "T": 0.1, "W": 1.1},
    8: {"P": 120, "T": 0.03, "W": 0.8},
    9: {"P": 120, "T": 0.08, "W": 0.8},
}

QUALITY_MAP = {
    1: "exclude", 2: "exclude", 3: "caution",
    4: "good", 5: "good", 6: "good", 7: "good", 8: "good",
}


def get_pairing_rules(subjectID):
    mod_result = subjectID % 4
    rules = {
        0: {"square": {"self": "f", "stranger": "j"}, "circle": {"self": "j", "stranger": "f"}},
        1: {"square": {"self": "j", "stranger": "f"}, "circle": {"self": "f", "stranger": "j"}},
        2: {"square": {"self": "j", "stranger": "f"}, "circle": {"self": "f", "stranger": "j"}},
        3: {"square": {"self": "f", "stranger": "j"}, "circle": {"self": "j", "stranger": "f"}},
    }
    return rules[mod_result]


def get_match_key(subjectID):
    match_keys = ['f', 'j', 'j', 'f']
    index = (subjectID - 1) % 4
    return match_keys[index]


def get_correct_order(subjectID):
    if subjectID % 2 == 0:
        return {"shape": {"square": "self", "circle": "stranger"}}
    else:
        return {"shape": {"square": "stranger", "circle": "self"}}


def compute_correct_key(shape, label, subjectID):
    pairing = get_pairing_rules(subjectID)
    return pairing[shape][label]


def compute_condition(shape, label, subjectID):
    correct_order = get_correct_order(subjectID)
    expected_label = correct_order["shape"][shape]
    if label == expected_label:
        return "Matching"
    else:
        return "NonMatching"


def compute_is_correct(shape, label, response, subjectID):
    correct_key = compute_correct_key(shape, label, subjectID)
    if pd.isna(response) or response == 'NA':
        return np.nan
    return 1 if str(response).strip() == correct_key else 0


def verify_file(filepath):
    df = pd.read_csv(filepath)
    errors = []
    groupID = df['groupID'].iloc[0]
    subjectID = df['subjectID'].iloc[0]

    cond = CONDITIONS.get(groupID, None)
    if cond is None:
        errors.append(f"Unknown groupID={groupID}")

    for idx, row in df.iterrows():
        shape = str(row['Shape']).strip()
        label = str(row['Label']).strip()
        response = str(row['Response']).strip()
        trialID = row['trialID']

        expected_correct_key = compute_correct_key(shape, label, subjectID)
        data_correct_key = str(row['CorrectKey']).strip()

        if expected_correct_key != data_correct_key:
            errors.append(
                f"trialID={trialID}: shape={shape}, label={label}, "
                f"Expected CorrectKey={expected_correct_key}, Got={data_correct_key}"
            )

        if response not in ['NA', 'nan', '']:
            expected_correct = compute_is_correct(shape, label, response, subjectID)
            data_correct = row['Correct']
            if not pd.isna(expected_correct) and expected_correct != data_correct:
                errors.append(
                    f"trialID={trialID}: shape={shape}, label={label}, response={response}, "
                    f"Expected Correct={expected_correct}, Got={data_correct}"
                )

    match_key = get_match_key(subjectID)
    mismatch_key = 'j' if match_key == 'f' else 'f'
    conditions = {
        "groupID": groupID,
        "subjectID": subjectID,
        "P": cond["P"] if cond else None,
        "T": cond["T"] if cond else None,
        "W": cond["W"] if cond else None,
        "match_key": match_key,
        "mismatch_key": mismatch_key,
        "correct_order": f"square↔{get_correct_order(subjectID)['shape']['square']}, "
                         f"circle↔{get_correct_order(subjectID)['shape']['circle']}",
    }

    return errors, conditions


def generate_report():
    all_files = sorted(RAW_DIR.glob("EXP_data_group*.csv"))
    total_files = len(all_files)
    total_errors = 0
    total_trials = 0
    results = []

    for fpath in all_files:
        print(f"Verifying: {fpath.name} ...", end=" ")
        errors, cond = verify_file(fpath)
        df = pd.read_csv(fpath)
        n_trials = len(df)

        formal_df = df[df['stage'] == 'formal']
        n_formal = len(formal_df)
        n_response = formal_df[formal_df['Response'].notna() & (formal_df['Response'] != 'NA')].shape[0]
        n_correct = formal_df[formal_df['Correct'] == 1].shape[0]
        omission_rate = (n_formal - n_response) / n_formal * 100 if n_formal > 0 else 0
        acc = n_correct / n_response * 100 if n_response > 0 else 0

        total_trials += n_trials
        total_errors += len(errors)
        status = "OK" if not errors else f"{len(errors)} ERRORS"
        print(f"trials={n_trials}, formal={n_formal}, omission={omission_rate:.1f}%, ACC={acc:.1f}%, {status}")

        gid = cond["groupID"]
        sid = cond["subjectID"]
        quality = QUALITY_MAP.get(gid, "unknown")
        results.append({
            "file": fpath.name,
            "groupID": gid,
            "subjectID": sid,
            "P": cond["P"],
            "T": cond["T"],
            "W": cond["W"],
            "n_formal": n_formal,
            "omission_rate": omission_rate,
            "accuracy": acc,
            "quality": quality,
            "match_key": cond["match_key"],
            "correct_order": cond["correct_order"],
            "errors": len(errors),
            "status": status,
        })

    print(f"\n{'='*80}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {total_files}")
    print(f"Total trials: {total_trials}")
    print(f"Total errors: {total_errors}")
    print(f"Overall status: {'PASS' if total_errors == 0 else 'FAIL'}")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(Path(__file__).parent / "verification_results.csv", index=False)
    print(f"\nDetailed results saved to verification_results.csv")

    group_summary = summary_df.groupby("groupID").agg({
        "n_formal": "sum",
        "omission_rate": "mean",
        "accuracy": "mean",
        "errors": "sum",
    }).reset_index()
    group_summary["quality"] = group_summary["groupID"].map(QUALITY_MAP)
    print(f"\n{'='*80}")
    print(f"GROUP-LEVEL SUMMARY")
    print(f"{'='*80}")
    print(group_summary.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    generate_report()
