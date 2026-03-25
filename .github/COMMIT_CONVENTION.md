# Commit Message Convention

When committing to this repository, follow these rules:

1. **Describe the effect, not the method.**
   - Good: `improve MSM throughput for k=19-21`
   - Bad: `switch from approach A to approach B in kernel`

2. **Keep messages short** (50 chars title, optional body).
   - Good: `fix GPU fallback threshold`
   - Bad: `adjust GPU_MIN_SIZE from 1<<12 to 1<<13 because profiling showed...`

3. **Never mention internal algorithm details** in commit messages.
   Commit history is public. Treat every message as if a competitor will read it.

4. **Use imperative mood**: `add`, `fix`, `improve`, `update`, `remove`.
