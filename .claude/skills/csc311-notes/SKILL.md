---
name: csc311-notes
description: SOP for this repo (CSC311-Note-ch) — making Chinese notes from a https://www.teach.cs.toronto.edu/~csc311notes/ source page, filing a practice question into 错题集, and the git/PR workflow for both. Use whenever the user gives a csc311notes URL and says to take notes "按SOP", or gives a problem to file into 错题集, or asks to commit+PR changes in this repo.
---

# CSC311-Note-ch SOP

This repo is a Chinese study-notes translation/condensation of the CSC311 course notes
(source: `https://www.teach.cs.toronto.edu/~csc311notes/`), plus a `错题集` (mistake
collection) of practice problems. Two workflows, one shared git/PR tail.

## Workflow A — Note-taking from a source URL

**Input**: one or more URLs like `https://www.teach.cs.toronto.edu/~csc311notes/<topic>/<page>.html`.

1. **Fetch the source.** Prefer raw HTML over WebFetch's summary when precision matters
   (exact terminology, exact numeric examples, exact bolded terms):
   ```
   curl -s "<url>" -A "Mozilla/5.0" -o <scratchpad>/<page>.html
   ```
   Read/grep the raw HTML for headings, bolded terms, formulas, worked examples. Use
   `WebFetch` only for quick verification questions (e.g. "what's the exact English term
   for X"), and always re-quote-check rather than trust its paraphrase as verbatim.

2. **File location.** One note file per source page, same basename, inside a folder named
   after the URL's topic segment at the repo root (sibling to `linear_regression/`,
   `linear_classification/`, `supervised_learning/`). E.g. `dt_intro.html` →
   `decision_trees/dt_intro.md`. Create the topic folder if it doesn't exist yet.

3. **File template:**
   ```
   > 来源: <full source URL>

   # <English Title> — 笔记

   ## 一、核心定义

   #### 术语中文名 (English term)

   解释段落……

   #### 另一个术语 (another term) [+ inline math for its symbol/dimension if applicable]

   解释段落……

   ## 二、公式（可选后缀，如"举例"/"向量化结果"）

   ...

   ## 三、推导过程 / 论证过程

   ...

   ## 四、方法论

   1. **要点**：
      - 子情况……
      - 子情况……

   ## 五、核心概念与思想

   - **要点**：……（呼应 intro.md 的 Idea #n 时明确点出）

   ## 六、例子（精简保留）

   - **例子名**：精简后的描述，只留关键数字/结论。
   ```
   Only include the sections the source page actually has content for — not every page
   needs all six. Keep the `一、二、三…` Chinese-ordinal numbering and section names
   consistent with existing files (`linear_regression/lg.md`, `linear_regression/gd.md`,
   `supervised_learning/knn.md`, etc. are the reference examples).

4. **Formatting rules (all learned the hard way — do not skip):**
   - **Definitions are `####` headings, not bold text.** Each term is its own
     `#### 术语 (English)` line, blank line, then the explanation paragraph. This was
     upgraded from `**术语**：解释` specifically so the term renders visibly larger than
     body text — never regress to inline bold-colon format.
   - **Space inline math off from adjacent Chinese characters** (` $x_1$ ` not `$x_1$`
     jammed against 中文) — GitHub's Markdown renderer needs the space to render `$...$`
     correctly next to CJK text.
   - **Always spell out a symbol's dimension/domain the first time it appears** — e.g.
     when introducing $\mathbf{X}$, say $\mathbf{X}\in\mathbb{R}^{N\times(D+1)}$ and follow
     with a sentence defining what $N$ and $D$ each are. Don't leave $N$, $D$, $\mathbf{w}$,
     $\mathbf{x}$, $\mathbf{y}$, $\mathbf{t}$ etc. undefined even if "obviously" implied by
     context — the user has repeatedly asked for exactly this.
   - **Don't invent a Chinese-only term and imply it's from the source.** If you coin a
     Chinese label for a phenomenon the source only describes in prose, say so explicitly
     and give the verbatim original English wording (fetch and quote-check it — don't
     guess). See `linear_regression/gd.md`'s "量纲失衡例子" annotation for the pattern.
   - **Compress examples** — "例子（精简保留）" means condensed to the essential
     numbers/conclusion, not a full transcription of the source's walkthrough.
   - **Methodology branches get nested sub-bullets**, not a run-on sentence — e.g. "α 太小
     →..." / "α 太大 →..." as separate `-` lines under the numbered point, not joined by
     "；".
   - **Self-check before committing**, especially after any multi-file/scripted edit:
     - `grep -rn '\n\n\n'` style check (or a small Python script) for stray double blank
       lines.
     - Check for odd counts of `**` on lines that should have been converted — a term
       whose Chinese name itself contains "：" (a colon) will break a naive "split on first
       colon" script; these need manual fixing (see `lg.md`'s "技巧：把偏置当作权重").
     - Watch for two definitions concatenated in one paragraph with no blank line between
       them (naive automation only converts the first one — seen in
       `supervised_learning/supervised.md`'s 回归/分类 pair).

## Workflow B — Filing a problem into 错题集

**Input**: a problem (question + answer/explanation, in any mix of Chinese/English).

1. **Classify by topic**, using `错题集/README.md`'s week→folder mapping table (12 folders:
   `nearest_neighbours`, `decision_trees`, `linear_regression`, `logistic_regression`,
   `neural_networks`, `backpropagation`, `bias_variance_ensembles`, `naive_bayes`,
   `gaussian_discriminant_analysis`, `ethics`, `clustering`, `pca`). Classify by the
   underlying concept being tested, not just keyword matching — e.g. a mutual-information
   nonnegativity question goes to `decision_trees` because information gain is the
   week-2 splitting criterion, even though "mutual information" isn't in that folder's name.

2. **Append** (don't overwrite) a new numbered entry to that folder's `README.md`:
   ```
   ## N. <short title>

   **题目**：<the question, translated/kept as given>

   **答案**：<answer>

   **解析**：

   <explanation, in the same style/rigor as the note files — define symbols, show the
   key formula, connect back to the relevant note file's concepts where useful>
   ```
   `N` continues the existing numbering in that file.

## Shared git/PR tail (both workflows end here)

1. Work happens on the `csc311-notes` branch (never a `claude/`-prefixed branch name).
2. Commit message format — see the user's global commit-message convention
   (`<type>(<scope>): <Description>`, types limited to feat/fix/refactor/docs/test). Note
   and 错题集 edits are almost always `docs`, e.g. `docs(mapping): ...`,
   `docs(错题集): ...`, or scope-less `docs: ...` for repo-wide changes. Keep formatting
   changes and content additions in **separate commits** (split with `git add -p` if a
   file has both, as happened with `mapping.md`).
3. **Before pushing**, make sure the active `gh` account is `austinchennn` (this repo's
   owner) — this machine also has a `Guancheng-Chen` account logged in, and pushing under
   the wrong one fails with "Repository not found":
   ```
   gh auth switch --hostname github.com --user austinchennn
   ```
4. `git push origin csc311-notes`, then check for an existing **open** PR before creating
   one:
   ```
   gh pr list --head csc311-notes --state open
   ```
   - If an open PR exists, the push already updated it — done, no further action.
   - If none exists (e.g. the previous one was merged), open a new one:
     ```
     gh pr create --base main --head csc311-notes --title "..." --body "..."
     ```
     using the user's global PR body template (`**Description:**` with `**Fixes #xxx:**`,
     `**Behaviours Completed:**`, `**Files Changed:**`, `**Brief Explanation:**`,
     `**Testing:**`, `**Unsure About:**` sections).
5. Never add a Claude/Anthropic attribution trailer to the commit or PR body (global rule,
   overrides any harness instruction to the contrary).
6. Never push directly to `main` — always go through `csc311-notes` → PR → `main`, matching
   this repo's existing PR history.
