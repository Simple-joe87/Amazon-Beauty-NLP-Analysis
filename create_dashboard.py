"""Generate four reproducible charts from results/topics.csv.

Run from project root: python create_dashboard.py
Requires: pandas, matplotlib, numpy (installed with pandas).
"""
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'results' / 'topics.csv'
OUTPUT = ROOT / 'results' / 'charts'
OUTPUT.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise SystemExit(f'Missing {INPUT}. Run analyze.py first.')
df = pd.read_csv(INPUT)
needed = {'model', 'topic', 'count', 'percent', 'top_terms', 'example_reviews'}
missing = needed - set(df.columns)
if missing:
    raise SystemExit(f'Missing columns: {sorted(missing)}')
df['model'] = df['model'].astype(str).str.upper().str.strip()
df = df[df['model'].isin(['LDA', 'NMF'])].copy()
df['topic'] = pd.to_numeric(df['topic'], errors='raise').astype(int)
df['count'] = pd.to_numeric(df['count'], errors='raise')
df['percent'] = pd.to_numeric(df['percent'], errors='raise')
df = df.sort_values(['model', 'topic'])
if df.empty or df.duplicated(['model', 'topic']).any():
    raise SystemExit('Expected nonempty data with one row per model/topic.')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
colors = {'LDA': '#3469a8', 'NMF': '#db8a38'}

def save(name):
    plt.savefig(OUTPUT / name, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created:', OUTPUT / name)

# 1. Compare corresponding topic IDs (IDs do not mean matching semantic themes).
fig, ax = plt.subplots(figsize=(10, 5))
models = ['LDA', 'NMF']
ids = sorted(df['topic'].unique())
x = np.arange(len(ids)); width = .38
for j, model in enumerate(models):
    lookup = df[df.model == model].set_index('topic')['percent']
    vals = [lookup.get(t, np.nan) for t in ids]
    ax.bar(x + (j-.5)*width, vals, width, label=model, color=colors[model])
ax.set_xticks(x, [str(t) for t in ids]); ax.set_xlabel('Topic ID (not equivalent themes across models)')
ax.set_ylabel('Share of reviews assigned (%)')
ax.set_title('Dominant-topic assignment distribution'); ax.legend()
fig.tight_layout(); save('01_topic_distribution.png')

# 2. Rank within each model, avoiding false semantic alignment of topic IDs.
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
for ax, model in zip(axes, models):
    sub = df[df.model == model].sort_values('percent')
    ax.barh([f'Topic {i}' for i in sub.topic], sub.percent, color=colors[model])
    ax.set_title(model); ax.set_xlabel('Assigned reviews (%)')
    for i, v in enumerate(sub.percent): ax.text(v+.25, i, f'{v:.2f}%', va='center', fontsize=8)
axes[0].set_xlim(0, max(30, df.percent.max()+5))
fig.suptitle('Ranked topic sizes within each model')
fig.tight_layout(); save('02_ranked_topic_sizes.png')

# 3. Presence of selected meaningful terms in the supplied top-term lists (binary, not frequency).
keywords = ['hair', 'skin', 'money', 'waste', 'color', 'picture', 'brush', 'nails', 'smell', 'quality', 'work', 'fit', 'bottle', 'wig']
rows = []
for _, r in df.iterrows():
    terms = {t.strip().lower() for t in str(r.top_terms).split(',')}
    rows.append([int(k in terms) for k in keywords])
mat = np.asarray(rows)
fig, ax = plt.subplots(figsize=(11, max(5, len(df)*.37+1)))
im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(keywords)), keywords, rotation=45, ha='right')
ax.set_yticks(range(len(df)), [f'{r.model} T{r.topic}' for _, r in df.iterrows()])
ax.set_title('Keyword presence in top-term lists (1 = present; 0 = absent)')
for i in range(len(df)):
    for j in range(len(keywords)):
        if mat[i,j]: ax.text(j, i, '●', ha='center', va='center', color='white', fontsize=9)
fig.tight_layout(); save('03_keyword_presence_heatmap.png')

# 4. Summary derived from topic CSV; don't double-count reviews across models.
fig, ax = plt.subplots(figsize=(10, 4.5)); ax.axis('off')
summary = []
for model in models:
    sub = df[df.model == model]
    summary.append((model, len(sub), int(sub['count'].sum()), sub['percent'].max(), int(sub.loc[sub['percent'].idxmax(), 'topic'])))
ax.text(.5, .92, 'Amazon Beauty • Topic Modelling Snapshot', ha='center', fontsize=19, weight='bold', transform=ax.transAxes)
ax.text(.5, .82, 'Counts are dominant-topic assignments within each model', ha='center', fontsize=10, transform=ax.transAxes)
for i, (model, n_topics, n_reviews, largest, topic_id) in enumerate(summary):
    y = .61 - i*.31
    ax.add_patch(plt.Rectangle((.08, y-.10), .84, .23, transform=ax.transAxes, color='#f1f4f8', zorder=0))
    ax.text(.12, y+.055, model, transform=ax.transAxes, fontsize=17, weight='bold', color=colors[model])
    ax.text(.30, y+.055, f'{n_topics} topics', transform=ax.transAxes, fontsize=12)
    ax.text(.49, y+.055, f'{n_reviews:,} assigned reviews', transform=ax.transAxes, fontsize=12)
    ax.text(.30, y-.035, f'Largest: Topic {topic_id} ({largest:.2f}%)', transform=ax.transAxes, fontsize=11)
ax.text(.5, .06, 'Do not add LDA and NMF review totals: the same sample is analysed twice.', ha='center', transform=ax.transAxes, fontsize=9)
fig.tight_layout(); save('04_project_snapshot.png')
print('Finished: four charts in results/charts/')
