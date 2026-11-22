"""Five local thermal contributions; the library binds cells separately for each call.

This constructs expressions, not a three-cell discretisation or a UFCx bundle.
The current UFCx adapter does not yet support the mixed interface arguments.
"""

import basix

from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import mesh, pull_back, push_forward, transformation_law
from uflx.nodes import EuclideanSpace, ScalarType
from uflx_basix.basix_nodes import BasixCell, BasixSpace
from uflx_basix.nodes import Space, StaticTopology
from uflx_basix.sides import static_side_map

K = BasixCell(basix.CellType.triangle.value)
I = BasixCell(basix.CellType.interval.value)

# One triangle role per material. Vertex labels describe reference correspondence.
# No overall three-cell topology is needed.
#
#    air volume       liquid volume       solid volume
#        Ta                Tl                  Ts
#
#    air -- Sal -- liquid       liquid -- Sls -- solid
#
# The liquid role can receive different concrete cells in the two interface calls.
Ta = StaticTopology((K,), ((0, 1, 2),))
Tl = StaticTopology((K,), ((1, 3, 2),))
Ts = StaticTopology((K,), ((3, 4, 2),))
Sal = StaticTopology((I,), ((1, 2),))
Sls = StaticTopology((I,), ((3, 2),))

# Abstract vector coordinate fields. The library supplies compatible traces
# for each interface invocation; declaring P1 geometry does not enforce that.
Xa = BasixSpace("Xa", Ta, basix.ElementFamily.P.value, 1, 0, ScalarType.real, (2,))
Xl = BasixSpace("Xl", Tl, basix.ElementFamily.P.value, 1, 0, ScalarType.real, (2,))
Xs = BasixSpace("Xs", Ts, basix.ElementFamily.P.value, 1, 0, ScalarType.real, (2,))
xa, xl, xs = function(Xa, "xa"), function(Xl, "xl"), function(Xs, "xs")
Da, Dl, Ds = mesh(xa, EuclideanSpace(2)), mesh(xl, EuclideanSpace(2)), mesh(xs, EuclideanSpace(2))
Eal = mesh(pull_back(xa, static_side_map(Ta, Sal, 0)), EuclideanSpace(2))
Els = mesh(pull_back(xl, static_side_map(Tl, Sls, 0)), EuclideanSpace(2))

# Local scalar temperature bases: three DOFs per input role.
# Assembled continuity is specified by library gather/scatter maps.
Va = BasixSpace("Va", Ta, basix.ElementFamily.P.value, 1, 0, ScalarType.real, ())
Vl = BasixSpace("Vl", Tl, basix.ElementFamily.P.value, 1, 0, ScalarType.real, ())
Vs = BasixSpace("Vs", Ts, basix.ElementFamily.P.value, 1, 0, ScalarType.real, ())
ua, va = function(Va, "ua"), function(Va, "va")
ul, vl = function(Vl, "ul"), function(Vl, "vl")
us, vs = function(Vs, "us"), function(Vs, "vs")
law = transformation_law((), 0)
Ua, Wa = push_forward(ua, Da, law), push_forward(va, Da, law)
Ul, Wl = push_forward(ul, Dl, law), push_forward(vl, Dl, law)
Us, Ws = push_forward(us, Ds, law), push_forward(vs, Ds, law)

# Conductivities and interface thermal conductances: positive runtime parameters.
C = Space("parameters", (), (), ScalarType.real)
ka, kl, ks = function(C, "ka"), function(C, "kl"), function(C, "ks")
hal, hls = function(C, "hal"), function(C, "hls")

# Three independent volume contributions.
a_air = form(integral(ka * (gradient(Ua, Da) @ gradient(Wa, Da)), Da), (va, ua))
a_liquid = form(integral(kl * (gradient(Ul, Dl) @ gradient(Wl, Dl)), Dl), (vl, ul))
a_solid = form(integral(ks * (gradient(Us, Ds) @ gradient(Ws, Ds)), Ds), (vs, us))

# Two interface contributions, with independent material temperatures.
# A positive h*(U_left-U_right) denotes heat transfer from left to right.
ma, ml = static_side_map(Da, Eal, 0), static_side_map(Dl, Eal, 0)
nl, ns = static_side_map(Dl, Els, 0), static_side_map(Ds, Els, 0)
jump_Ual = pull_back(Ua, ma) - pull_back(Ul, ml)
jump_Wal = pull_back(Wa, ma) - pull_back(Wl, ml)
jump_Uls = pull_back(Ul, nl) - pull_back(Us, ns)
jump_Wls = pull_back(Wl, nl) - pull_back(Ws, ns)
a_air_liquid = form(integral(hal * jump_Ual * jump_Wal, Eal), ((va, vl), (ua, ul)))
a_liquid_solid = form(integral(hls * jump_Uls * jump_Wls, Els), ((vl, vs), (ul, us)))

# Library bindings (conceptual; mesh field identities remain outside UFLx):
#
# contribution       liquid input        library restriction
# a_liquid           ul, xl              selected liquid volume cell
# a_air_liquid       ul, xl              liquid neighbour of air-interface facet
# a_liquid_solid     ul, xl              liquid neighbour of solid-interface facet
#
# Each invocation gathers its own values and scatters its own output DOFs.
# Reusing ul and xl does not require these selected cells to be identical.
#
# Five integration contributions update seven assembled material blocks:
#
#                 air  liquid solid
#       air       [ *     *     0 ]
#       liquid    [ *     *     * ]
#       solid     [ 0     *     * ]
