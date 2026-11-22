import { copyFile, cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

export async function copySiteAssets(root: string, output: string) {
  const target = path.join(output, 'assets');
  const primer = path.join(root, 'node_modules/@primer/primitives');
  await cp(path.join(root, 'css'), target, { recursive: true });
  await cp(path.join(root, '../background'), target, { recursive: true });
  await copyFile(path.join(import.meta.dirname, '../reader.js'), path.join(target, 'reader.js'));
  for (const mode of ['light', 'dark']) {
    await copyFile(path.join(primer, `dist/css/functional/themes/${mode}-tritanopia.css`),
      path.join(target, `github-${mode}-tritanopia.css`));
  }
  const fontFiles = path.join(target, 'fonts/files');
  await mkdir(fontFiles, { recursive: true });
  await mkdir(path.join(target, 'licenses'), { recursive: true });
  await copyFile(path.join(root, '../LICENSE'), path.join(target, 'licenses/Blog.txt'));
  for (const [name, stylesheet, published, license] of [
    ['jetbrains-mono', 'index.css', 'index.css', 'fonts/LICENSE'],
    ['noto-sans-sc', 'wght.css', 'noto-sans-sc.css', 'licenses/NotoSansSC.txt'],
    ['noto-serif-sc', 'wght.css', 'noto-serif-sc.css', 'licenses/NotoSerifSC.txt'],
  ] as const) {
    const directory = path.join(root, 'node_modules/@fontsource-variable', name);
    const css = await readFile(path.join(directory, stylesheet), 'utf8');
    const files = new Set<string>();
    for (const [, file] of css.matchAll(/url\(\.\/files\/([^)]+)\)/g)) {
      if (file) files.add(file);
    }
    await Promise.all([...files].map(file => copyFile(path.join(directory, 'files', file), path.join(fontFiles, file))));
    await writeFile(path.join(target, 'fonts', published), css);
    await copyFile(path.join(directory, 'LICENSE'), path.join(target, license));
  }
  const englishFonts = path.join(root, 'node_modules/@fontsource/source-serif-4');
  for (const weight of [400, 700]) {
    for (const style of ['normal', 'italic']) {
      const file = `source-serif-4-latin-${weight}-${style}.woff2`;
      await copyFile(path.join(englishFonts, 'files', file), path.join(target, 'fonts/files', file));
    }
  }
  await copyFile(path.join(englishFonts, 'LICENSE'), path.join(target, 'licenses/SourceSerif4.txt'));
  await copyFile(path.join(primer, 'LICENSE'), path.join(target, 'licenses/Primer.txt'));
  await copyFile(path.join(root, 'node_modules/@mathjax/src/LICENSE'), path.join(target, 'licenses/MathJax.txt'));
  const mathFont = JSON.parse(await readFile(path.join(root, 'node_modules/@mathjax/mathjax-newcm-font/package.json'), 'utf8'));
  await writeFile(path.join(target, 'licenses/NewCM.txt'),
    `${mathFont.name} ${mathFont.version}\n${mathFont.repository.url}\nLicense: ${mathFont.license}\n\n` +
    await readFile(path.join(root, 'node_modules/@mathjax/src/LICENSE'), 'utf8'));
  await copyFile(path.join(root, 'node_modules/@shikijs/themes/LICENSE'), path.join(target, 'licenses/Shiki.txt'));
  await copyFile(path.join(root, 'licenses/GitHubTheme.txt'), path.join(target, 'licenses/GitHubTheme.txt'));
}
