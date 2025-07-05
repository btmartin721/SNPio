# snpio/utils/plot_queue.py

from typing import Any, Dict, List

# Global plot queue – all queue_* methods write here
queued_plots: List[Dict[str, Any]] = []
