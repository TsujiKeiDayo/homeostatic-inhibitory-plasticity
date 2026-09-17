"""Shared defaults for the five saved-result notebooks. No computation here."""
from experiment_settings import EXPERIMENT_ROOT, RESULTS_ROOT, SAVE_FIGURE_PNG, SAVE_FIGURE_PDF

OUTPUT = EXPERIMENT_ROOT / "analysis" / "notebook_figures_v2"
# None selects each model's target by seed-median final classifier validation loss.
# A dictionary overrides display targets only; HP eta selection stays unchanged.
# DISPLAY_TARGETS = {"rec": .17, "ff": .55, "thresh": 1.55}
DISPLAY_TARGETS = None

# Viewing is the default. Enable saving in the notebook's export cells.
SAVE_NUMBERS = False
SAVE_PNG = SAVE_FIGURE_PNG
SAVE_PDF = SAVE_FIGURE_PDF
SHOW_FIGURES = True
EXPORT_NAME = "all_v1"  # Choose a new name for each complete export.
