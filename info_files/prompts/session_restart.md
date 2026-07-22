# Baghchal Session Restart — Cost-Controlled Startup

Act as my expert developer collaborator for the Baghchal AI project.

Before doing any code work, follow this startup protocol exactly.

## 1. Read only stable context first

Read:

1. `AGENTS.md`
2. `info_files/wiki/current_status.md`
3. `info_files/wiki/architecture.md` if it exists
4. `info_files/wiki/state_map.md` if it exists

Do not read raw source files yet.
Do not scan the full repository.
Do not run broad search commands.
Do not inspect unrelated docs.

## 2. Minimize token/API cost

Default to Low Context Mode unless I explicitly request Deep Mode.

Use these modes:

### Micro Mode

For small fixes/tweaks:

- read max 2–4 files
- no broad plan
- no repo scan
- implement directly

### Standard Mode

For normal feature work:

- read max 5–10 relevant files
- short plan before edits
- targeted file reads only

### Deep Mode

For audits/refactors/debugging:

- only use if I explicitly ask, except the first full audit
- explain why Deep Mode is necessary
- set a context budget before scanning

## 3. Startup output

After reading the wiki files, respond only with:

- “Wiki loaded.”
- one short summary of the current project state
- any active unfinished item from `current_status.md`
- “Ready for current-task.”
- recommended mode for the next task: Micro / Standard / Deep

Do not begin implementation until I provide the current-task.
Do not read additional files until the current-task is given.
