import sys
from pathlib import Path

# Add the project root so `import spy_1dte_vol_pipeline` works from any CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))
