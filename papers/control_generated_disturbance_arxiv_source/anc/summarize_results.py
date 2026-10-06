"""Print manuscript summaries from the recorded experiment CSVs.

Run: python3 anc/summarize_results.py
Uses only the Python standard library. Makes no model/API calls.
"""
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent

def read(name):
    with (HERE / name).open(newline='') as f:
        return list(csv.DictReader(f))

def mean(rows, field):
    return sum(float(r[field]) for r in rows) / len(rows)

def fraction(rows, field):
    return sum(r[field].lower() == 'true' for r in rows) / len(rows)

print('Table 1: persistent agent, mechanism comparison')
rows = read('induced_robustness_mechanisms.csv')
for mechanism in ('RESULT_INJECTION', 'ARGUMENT_CORRUPTION'):
    for label in ('blind', 'aware_k3.0'):
        selected = sorted([r for r in rows if r['agent'] == 'persistent' and r['mechanism'] == mechanism and r['label'] == label], key=lambda r: float(r['alpha']))
        print(mechanism, label, '; '.join(f"alpha={r['alpha']}: success={float(r['success_rate']):.5f}, g={float(r['mean_g']):.5f}" for r in selected))

print('\nTable 2: baseline ladder')
rows = read('induced_robustness_baselines.csv')
for agent in ('persistent', 'impulsive'):
    for label in ('ungoverned', 'governor_alpha0', 'cost_blind_alpha2', 'cost_aware_alpha2', 'fixed_weak_alpha2'):
        r = next(r for r in rows if r['agent'] == agent and r['label'] == label)
        apply = '--' if label == 'ungoverned' else r['mean_apply_rate']
        print(agent, label, 'success=' + r['success_rate'], 'apply=' + apply)

gemini = read('gemini_induced_checkpoint.csv')
assert len(gemini) == 576
keys = {(r['task_id'], r['mode'], float(r['alpha']), int(r['repeat'])) for r in gemini}
assert len(keys) == len(gemini), 'Duplicate episode identifiers'
assert len({r['task_id'] for r in gemini}) == 6

print('\nTable 3: all SIX tasks; 48 episodes per cell')
for alpha in (0, 1, 2):
    for mode in ('blind', 'aware_k1.5', 'aware_k3.0', 'aware_k6.0'):
        selected = [r for r in gemini if float(r['alpha']) == alpha and r['mode'] == mode]
        assert len(selected) == 48
        print(f"alpha={alpha} {mode}: n={len(selected)} success={fraction(selected, 'success'):.5f} g={mean(selected, 'mean_regulation_intensity'):.5f} blocks={mean(selected, 'hard_blocks'):.5f}")

print('\nTable 4: all six tasks, alpha=2; 8 episodes per cell')
for task in ('T002_wrong_folder', 'E003_confident_wrong_report', 'C001_wrong_subdir', 'E002_wrong_dir_and_ext', 'C002_wrong_ext_dir', 'C003_wrong_root'):
    for mode in ('blind', 'aware_k3.0'):
        selected = [r for r in gemini if r['task_id'] == task and float(r['alpha']) == 2 and r['mode'] == mode]
        assert len(selected) == 8
        successes = sum(r['success'].lower() == 'true' for r in selected)
        answered = sum(r['answered'].lower() == 'true' for r in selected)
        print(f"{task} {mode}: successes={successes}/8 answered={answered}/8 blocks={mean(selected, 'hard_blocks'):.3f}")

print('\nSection 5.2: RF-only k=6 intensity endpoints, alpha 0 to 2')
rows = read('cost_aware_aggregate.csv')
for agent in ('impulsive', 'persistent'):
    for alpha in (0, 2):
        selected = [r for r in rows if r['agent_type'] == agent and r['ordering'] == 'RF' and int(r['max_steps']) == 16 and float(r['alpha']) == alpha and r['mode'] == 'aware_k6.0']
        assert len(selected) == 1, (agent, alpha, selected)
        print(agent, alpha, selected[0]['mean_regulation_intensity'])
