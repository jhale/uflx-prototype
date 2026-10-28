#!/bin/sh
# Runs every test under coverage.py and reports the lines and branches of the passes that never ran.
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
rm -f .coverage .coverage.*
for t in test_actions.py test_more.py test_random.py test_full.py test_gateaux.py test_pipeline.py test_integrals.py test_fields.py test_facets.py test_surface.py; do
    $PY -m coverage run -p --branch --include="gateaux.py,passes.py,derivatives.py,integrals.py,pullbacks.py,algebra.py" "$t" > /dev/null
done
$PY -m coverage combine -q
$PY -m coverage report --show-missing
