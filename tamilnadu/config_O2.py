"""
config.py — paths for the Objective-2 Tamil Nadu pipeline.

This tree lived in a subfolder called `rajmethod/` until 2026-10-30. The name
described the METHOD's origin, not the data, and read as though the project
were about Rajasthan -- it is not; every input, design and result here is
Tamil Nadu. Since this directory is the whole working pipeline, it was
promoted to the project root and the older implementation retired to
archive/old_implementation/.

This tree is Manvitha's ACTUAL Rajasthan implementation
(Github/Objective---2/objective2-rajasthan), copied in and pointed at Tamil
Nadu's frozen inputs. It replaced a hand-port that had been written from her
.docx report's prose: a comparison against the real source found ten material
physics divergences, the largest being that her collector is a MASS-FLOW loop
(flow rate sets the loop delta-T and the stratified return node, so it acts on
stratification) while the hand-port had modelled it as a pure heat source. A
flow-rate optimisation built on the wrong one of those is meaningless, so the
code itself is adopted rather than re-derived.

What differs from her tree, deliberately and only here:
  * STATE = "tamilnadu", with her state-config format filled from Tamil Nadu's
    own Objective 1 output (regimes, PCM shortlists, mains temperatures).
  * tank.volume_L = 100 and collector.area_m2 = 2.5 (she uses 50 L / 1.5 m2).
    The tank size is a project instruction; the collector is scaled with it to
    keep 40 L/m2, the low end of IS 12976:2023's 40-100 L/m2 band -- her
    33.3 L/m2 sits below that band, which her own report flags as a caveat.
    Her absolute kWh figures are therefore NOT directly comparable to ours;
    the algorithms, gates and pass/fail criteria are.
  * data/ is a copy of Tamil Nadu's frozen inputs laid out the way her
    io_utils expects (data/objective1, data/weather, data/demand).

Everything else -- src/, scripts/, the gates, the metric definitions -- is hers
unchanged, so a divergence in results is a divergence in inputs, not in method.
"""

from pathlib import Path

STATE = "tamilnadu"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR                     # this IS the project root since the 2026-10-30 promotion

DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"

OBJ1_FROZEN_DIR = DATA_DIR / "objective1"
WEATHER_DIR = DATA_DIR / "weather"
DEMAND_DIR = DATA_DIR / "demand"
MANIFEST_FILE = OBJ1_FROZEN_DIR / "manifest.json"

ALL_DIRS = [DATA_DIR, CONFIGS_DIR, RESULTS_DIR, DOCS_DIR,
            OBJ1_FROZEN_DIR, WEATHER_DIR, DEMAND_DIR]


def ensure_dirs():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
