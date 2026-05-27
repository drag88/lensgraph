"""Back-compat shim — the visual eval runner lives at
``eval.runners.run_visual_eval`` because it defines an eval run and
writes ``eval_runs`` / ``eval_results``. Prefer invoking the canonical
module path:

    uv run python -m eval.runners.run_visual_eval [args...]

This shim exists so older ``python -m scripts.run_visual_eval`` calls
keep working. No logic lives here — adding any would duplicate the
implementation.
"""

from __future__ import annotations

from eval.runners.run_visual_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
