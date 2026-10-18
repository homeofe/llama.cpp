# Round 11 stopped by owner

The owner requested the machine/thread back and shorter tests on 2026-10-18 at 12:34 UTC. Do not automatically resume the old test queue. Follow the owner's next task.

Read `NOTES-round11.md` for the preserved evidence and unexecuted work. Qwen is complete at +1.68%/+1.86%; Ling has one matched pair at +1.40%/+1.47% plus an unpaired async pass. No async code adopted. Quality, CUDA timeline, 32K and panel checks remain unexecuted. No gateway candidate is validated.

Sources and worktree CUDA runtime are restored; the live gateway and system settings are unchanged. There are no measurement processes or GPU lease. See `task-logs/round11/stopped-state.json`. Prototype and raw evidence are retained locally. Do not run `sequence.py` unchanged: it names a historical PID and its signal-handler wait needs replacing before reuse.
