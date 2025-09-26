# Project Instructions

## Conversation style
- Be brief and on-point, avoid unnecessary blabbing.
- Expand on topics only when asked to elaborate, explain, give examples etc.

## Scope

## Coding Standards
- Style: Keep code idempotent and explicit. Use small, composable functions.
- Adhere to Pyright standards, address the warnings and errors.
- Avoid numbering things in the comments.
- Always add a new line at the end of any file.

## Development
- Always run the code before and after any changes to confirm the new functionality and avoid introducing breaking changes.
- Print portions of introduced dataframes and variables to confirm the program works as expected.
- Use Docker for running the code, installing packages etc.

## Security & Secrets
- Never hardcode tokens or secrets. Read from env and document required keys.

## When In Doubt
- Prefer small, incremental changes and add tests.
- Document behavior in docstrings/comments near the change. Keep comments concise and actionable.
