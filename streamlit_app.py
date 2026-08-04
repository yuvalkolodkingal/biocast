"""Deployment entry point for the design studio.

Locally the app is run as `PYTHONPATH=. streamlit run biocast/gui/app.py`. No host
sets PYTHONPATH for you, and `streamlit run` puts only the *script's* directory on
sys.path — so run from `biocast/gui/` the `biocast` package is invisible and the app
dies on import. This file sits at the repository root, which makes the root the
directory Streamlit adds, and every hosted target (Hugging Face Space, Cloud Run,
Streamlit Community Cloud) can point straight at it.
"""
import sys
from pathlib import Path

# explicit rather than relying on sys.path[0], so this also works under runners that
# invoke the file by absolute path from another working directory
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biocast.gui.app import main  # noqa: E402  (import must follow the path fix)

main()
