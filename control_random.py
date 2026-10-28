# Negative controls, each of which should change values.
import runpy
import sys

import core
import passes

mode = sys.argv[1]
if mode == "no-rename":
    core.rename = lambda node, name, avoid: name
    script = ["test_random.py", "2000"]
elif mode == "sequential-tuple":
    # Binding the members of a tuple entry one at a time captures a later member's function.
    def sequential(e, mapping):
        for name, value in mapping.items():
            e = core.substitute(e, {name: value})
        return e

    passes.substitute = sequential
    y0, y1 = core.Fn(core.Space("V"), "y0"), core.Fn(core.Space("V"), "y1")
    form = core.Form(core.Prod(y0, y1), ((y0, y1),))
    out = passes.apply_actions(core.Action(form, 0, (y1, core.Num(2))))
    print("binding (y0, y1) to (y1, 2) gives", out, "instead of y1 * 2")
    script = ["test_full.py", "500"]
sys.argv = script
runpy.run_path(script[0], run_name="__main__")
