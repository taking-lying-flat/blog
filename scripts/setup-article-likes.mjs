import { readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const query = (query, variables = {}) => {
  const result = JSON.parse(execFileSync('gh', ['api', 'graphql', '--input', '-'], {
    input: JSON.stringify({ query, variables }), encoding: 'utf8',
  }));
  if (result.errors) throw new Error(JSON.stringify(result.errors));
  return result.data;
};
const posts = JSON.parse(await readFile(path.join(root, 'site/posts.json'), 'utf8'));
const existing = new Map();
let cursor = null;
let repository;
do {
  repository = query(`query($cursor: String) {
    repository(owner: "taking-lying-flat", name: "blog") {
      id discussionCategories(first: 20) { nodes { id slug } }
      discussions(first: 100, after: $cursor) {
        nodes { id number url body } pageInfo { hasNextPage endCursor }
      }
    }
  }`, { cursor }).repository;
  for (const discussion of repository.discussions.nodes) {
    const slug = discussion.body.match(/<!-- blog-like: ([a-z0-9-]+) -->/)?.[1];
    if (slug) {
      if (existing.has(slug)) throw new Error(`Duplicate discussion for ${slug}`);
      existing.set(slug, discussion);
    }
  }
  cursor = repository.discussions.pageInfo.hasNextPage ? repository.discussions.pageInfo.endCursor : null;
} while (cursor);
const category = repository.discussionCategories.nodes.find(c => c.slug === 'announcements');
if (!category) throw new Error('Enable repository Discussions before running setup.');
const mapping = {};
for (const post of posts) {
  let discussion = existing.get(post.slug);
  if (!discussion) {
    let title = post.title ?? post.slug;
    try {
      const html = await readFile(path.join(root, 'site/dist/posts', post.slug, 'index.html'), 'utf8');
      title = html.match(/<title>(.*?) · Blog<\/title>/)?.[1]?.replaceAll('&amp;', '&') ?? title;
    } catch { /* New posts may not have been built yet. */ }
    const body = `[阅读文章：${title}](https://taking-lying-flat.github.io/blog/posts/${post.slug}/)\n\n点击这段正文下方的 ❤️，即可为文章点赞；再次点击可以取消。需要先登录 GitHub。\n\n点赞数按日汇总到仓库 README 的历史曲线中。\n\n<!-- blog-like: ${post.slug} -->`;
    discussion = query(`mutation($input: CreateDiscussionInput!) {
      createDiscussion(input: $input) { discussion { id number url } }
    }`, { input: { repositoryId: repository.id, categoryId: category.id, title: `❤️ ${title}`, body } }).createDiscussion.discussion;
    console.log(`Created like entry: ${post.slug}`);
  }
  mapping[post.slug] = { id: discussion.id, number: discussion.number, url: discussion.url };
}
await writeFile(path.join(root, 'site/like-discussions.json'), JSON.stringify(mapping, null, 2) + '\n');
console.log(`Mapped ${posts.length} articles.`);
