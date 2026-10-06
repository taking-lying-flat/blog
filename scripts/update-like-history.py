"""Collect real article HEART reactions and render the README's daily history."""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def fetch_counts(mapping):
    ids = [record['id'] for record in mapping.values()]
    payload = {'query': '''query($ids: [ID!]!) {
      nodes(ids: $ids) { ... on Discussion { id reactions(content: HEART) { totalCount } } }
    }''', 'variables': {'ids': ids}}
    result = subprocess.run(['gh', 'api', 'graphql', '--input', '-'], input=json.dumps(payload),
                            text=True, capture_output=True, check=True)
    response = json.loads(result.stdout)
    if response.get('errors'):
        raise RuntimeError('GitHub returned errors; preserving the previous snapshot.')
    nodes = response['data']['nodes']
    if any(not node or 'reactions' not in node for node in nodes):
        raise ValueError('Missing discussion; preserving the previous snapshot.')
    counts = {node['id']: node['reactions']['totalCount'] for node in nodes}
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError('Invalid reaction count.')
    return {slug: counts[record['id']] for slug, record in mapping.items()}


def update_snapshot(history, counts, date):
    snapshot = {'date': date, 'total': sum(counts.values()), 'posts': counts}
    snapshots = [s for s in history.get('snapshots', []) if s['date'] != date] + [snapshot]
    return {'version': 1, 'metric': 'GitHub discussion HEART reactions',
            'timezone': 'Asia/Shanghai', 'snapshots': sorted(snapshots, key=lambda s: s['date'])}


def render_chart(history, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'blog-like-history', 'font.size': 10})
    snapshots = history['snapshots']
    dates = [datetime.fromisoformat(s['date']) for s in snapshots]
    totals = [s['total'] for s in snapshots]
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=120)
    fig.patch.set_facecolor('#fcfcfe'); ax.set_facecolor('#fcfcfe')
    fig.subplots_adjust(left=.085, right=.97, bottom=.19, top=.7)
    fig.text(.085, .88, 'Article Likes', fontsize=21, weight='bold', color='#263348')
    fig.text(.085, .80, 'Daily snapshots · GitHub heart reactions', fontsize=10, color='#6d788b')
    fig.text(.97, .87, f'{totals[-1]:,}', fontsize=27, weight='bold', color='#b55070', ha='right')
    fig.text(.97, .80, f'Updated {snapshots[-1]["date"]} · Asia/Shanghai', fontsize=9, color='#6d788b', ha='right')
    ax.plot(dates, totals, color='#bb5b7c', linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(dates, totals, color='#bb5b7c', alpha=.09)
    ax.set_ylim(-.05, max(1, max(totals) * 1.2))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    if len(dates) == 1:
        ax.set_xticks(dates)
        ax.set_xlim(mdates.date2num(dates[0]) - .5, mdates.date2num(dates[0]) + .5)
        ax.annotate('History starts here', (dates[0], totals[0]), xytext=(12, 17),
                    textcoords='offset points', color='#6d788b', fontsize=9)
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.grid(axis='y', color='#e6eaf0', linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(colors='#6d788b', length=0, pad=10)
    ax.set_ylabel('Likes', color='#6d788b', labelpad=12)
    fig.savefig(output, format='svg', metadata={'Date': None, 'Title': 'Article like history'})
    plt.close(fig)


def main():
    mapping = json.loads((ROOT / 'site/like-discussions.json').read_text())
    counts = fetch_counts(mapping)
    date = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    history_path = ROOT / 'data/likes-history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    history = update_snapshot(history, counts, date)
    (ROOT / 'docs').mkdir(exist_ok=True); (ROOT / 'data').mkdir(exist_ok=True)
    # Render successfully before replacing any published data.
    chart = ROOT / 'docs/likes-history.tmp.svg'
    render_chart(history, chart)
    chart.replace(ROOT / 'docs/likes-history.svg')
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'data/article-likes.json').write_text(json.dumps(history['snapshots'][-1], ensure_ascii=False, indent=2) + '\n')
    print(f'Recorded {sum(counts.values())} likes across {len(counts)} articles on {date}.')


if __name__ == '__main__': main()
