const categories = [
  { id: 'reports', title: '技术报告' },
  { id: 'llm', title: '大模型' },
  { id: 'token-mixer', title: 'Token Mixer' },
  { id: 'infra', title: 'AI Infra' },
  { id: 'policy', title: '策略优化' },
  { id: 'generative', title: '生成模型' },
  { id: 'code', title: '源码解读' },
  { id: 'issues', title: 'GitHub Issue' },
] as const;
type Category = typeof categories[number]['id'];
export type ArticleEntry = {
  file: `../papers/${string}/lakes/${string}.lake` | `../code/lakes/${string}.lake`;
  slug: string;
  category: Category;
  categoryOrder?: number;
  date: `${number}-${number}-${number}`;
  description: string;
  tags: string[];
  aliases?: string[];
  downloads?: string[];
};

export interface SearchRecord { slug: string; text: string }
export interface CategoryGroup<T> { id: Category; title: string; posts: T[] }

export function orderCategories<T extends ArticleEntry>(posts: T[]) {
  const groups = new Map<Category, CategoryGroup<T>>(
    categories.map(category => [category.id, { ...category, posts: [] }]),
  );
  for (const post of posts) {
    const group = groups.get(post.category);
    if (!group) throw new Error(`Invalid article category: ${post.slug}`);
    if (post.categoryOrder !== undefined && (!Number.isInteger(post.categoryOrder) || post.categoryOrder < 1)) {
      throw new Error(`Invalid category reading order: ${post.slug}`);
    }
    group.posts.push(post);
  }
  for (const group of groups.values()) {
    group.posts.sort((a, b) => (a.categoryOrder ?? Number.MAX_SAFE_INTEGER) - (b.categoryOrder ?? Number.MAX_SAFE_INTEGER));
  }
  return groups.values();
}
