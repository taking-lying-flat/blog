"""Provision public article counters. Owner token stays outside the repository.

Usage: BLOG_LIKES_OWNER_TOKEN=... python scripts/setup-article-likes.py
API: https://github.com/kako-jun/nostalgic/blob/main/docs/user-guide/services/like.md
"""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'https://api.nostalgic.llll-ll.com/like'


def call(action, payload):
    result = subprocess.run([
        'curl', '--fail', '--silent', '--show-error', '--max-time', '60',
        '-H', 'Content-Type: application/json', '--data-binary', '@-',
        ENDPOINT + '?action=' + action,
    ], input=json.dumps(payload), text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    if not data.get('success'):
        raise RuntimeError('Counter service rejected the request.')
    return data


def main():
    token = os.environ['BLOG_LIKES_OWNER_TOKEN']
    posts = json.loads((ROOT / 'site/posts.json').read_text())
    items = [{'id': 'tlf-blog-' + post['slug'],
              'url': 'https://taking-lying-flat.github.io/blog/posts/' + post['slug'] + '/'}
             for post in posts]
    for start in range(0, len(items), 100):
        call('batchCreate', {'token': token, 'items': items[start:start + 100]})
    records = call('batchLookup', {'token': token, 'urls': [item['url'] for item in items]})['data']
    by_url = {record['url']: record for record in records}
    mapping = {}
    for post, item in zip(posts, items):
        record = by_url[item['url']]
        if not record.get('exists') or not record.get('authorized'):
            raise RuntimeError('Counter ownership could not be verified.')
        mapping[post['slug']] = record['id']
    config = {'endpoint': ENDPOINT, 'posts': mapping}
    (ROOT / 'site/likes-config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Configured {len(mapping)} article counters.')


if __name__ == '__main__':
    main()
