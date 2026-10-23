# docs_index

Drop plain-text or markdown excerpts of API notes and gotchas for your pinned
simulator version here (.md, .txt, .py, .rst) -- see `mujoco_notes.md` and
`newton_notes.md` for the current ones. SimSmith chunks and retrieves from
these files and feeds them to the generate/repair prompts as reference
material. Do not mix simulator versions, and do not drop in a full working
solution script here -- these are API notes for the model to work from, not
answers to copy.

Optional (Isaac Sim only): `isaac_api_allowlist.txt`, one module path per
line (for example `isaacsim.core.api`), to reject scripts importing modules
that do not exist in your pinned release.
