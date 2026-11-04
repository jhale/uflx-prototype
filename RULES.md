# Rewrite rules

The specification owns the formulas. This table records their implementation
locations and records differences found during the reorganisation.

`rules/` contains mathematical rules and applicability checks. `passes/` selects
rules, runs traversal and performs cleanup. A rule returns `None` when it does
not match. Returning the input retains it; a replacement goes through type
completion and traversal. Substitution keeps its separate binding-aware walk.

Paths below are relative to `src/uflx_spec_interpreter/_src/`.

| Rules | Implementation | Main tests in `test/` |
| --- | --- | --- |
| ST01 | `rules/structural.py::_structural_fields`, `_commute` | `test_actions.py`, `test_tensor_gateaux.py`, `test_broadcast.py`, `test_rule_groups.py` |
| ST02 | `passes/actions.py::_rule` | `test_actions.py`, `test_mixed.py` |
| ST03–ST05 | `rules/structural.py::_action_retained`; shared with `passes/contracts.py` | `test_action_spatial.py`, `test_core_normal_form.py` |
| ST06 | `nodes/forms.py::Action._validate` | `test_signatures.py` |
| ST07, ST10–ST11 | `passes/gradients.py`, `passes/pullbacks.py` | `test_action_spatial.py` |
| ST08 | `rules/structural.py::_gradient` | `test_gradients.py`, `test_rule_groups.py` |
| ST09 | `rules/structural.py::_gradient_component` | `test_gradients.py` |
| ST13–ST16 | `passes/arithmetic.py::expand_arithmetic`; final step of `lower` | `test_tensor_arithmetic.py` |
| ST12 | `rules/structural.py::_gradient_remains`; shared with `passes/contracts.py` | `test_algorithms/rewriting.py`, `test_piola_applicability.py` |
| L01–L04 | `rules/linearity.py::_linear_fields`, `rules/structural.py::_commute` | `test_gateaux.py`, `test_tensor_gateaux.py`, `test_broadcast.py` |
| L05, L07 | `passes/actions.py::_rule` | `test_actions.py`, `test_typed_zero.py` |
| L10 | `rules/differentiation.py::_deriv` selects `_zero_like` | `test_derivative_nodes/validation.py`, `test_typed_zero.py` |
| L11 | `passes/gradients.py::_rule` selects a typed component zero | `test_gradients.py`, `test_rule_groups.py` |
| D01–D10, D12 | `rules/differentiation.py::_derivative_of_node` | `test_derivative_algebra.py`, `test_tensor_gateaux.py`, `test_full.py` |
| D11, D15 | `rules/differentiation.py::_deriv`; value occurrence through `algorithms/substitution.py::_occurs` | `test_gateaux.py`, `test_derivative_nodes/validation.py`, `test_rule_groups.py` |
| D13 | `passes/derivatives.py::apply_derivatives`; operands first in `algorithms/rewriting.py::_rewrite` | `test_lowering_recipe.py`, `test_single_derivative.py` |
| D16 | `passes/gradients.py::apply_gradients` selects compositions from `calculus.py` | `test_gradients.py`, `test_piola_applicability.py` |
| D17 | `rules/differentiation.py::_coordinate_partial` | `test_gradients.py`, `test_rule_groups.py` |
| D19–D20 | `rules/differentiation.py::_derivative_of_node`, before D03–D04 | `test_rule_groups.py` |
| SB16–SB17 | `rules/binding.py::_bind`; uses shared substitution | `test_actions.py`, `test_controls.py` |
| CC01 | `rules/coordinates.py::_chain_partial` | `test_pull_back_gradients.py` |
| CC06–CC07 | `rules/coordinates.py::_piola_law`; shared by rewriting and contracts | `test_piola_applicability.py` |

The remaining groups use the following locations. Substitution stays in its
existing module because it needs a different walk from expression rewriting.

| Rules | Implementation | Main tests in `test/` |
| --- | --- | --- |
| SH01 | Mathematical admissibility is a precondition; oracle comparisons check defined examples | `test_random.py`, `test_full.py` |
| SH02–SH07 | `algorithms/rewriting.py::_complete_indices`, `_promote`, `_zero_like` | `test_index_completion.py`, `test_rule_groups.py` |
| SH08 | Promotion in `rules/binding.py` and `rules/differentiation.py`; completion before rebuilding parents | `test_gateaux.py`, `test_index_completion.py` |
| SH09–SH11 | `algorithms/traversal.py::_map_values`, node `value_fields` | `test_lowering_passes/contracts.py`, `test_static_type.py` |
| SH12 | `algorithms/rewriting.py::_rewrite`, `rules/pullback_algebra.py::_composed` | `test_lowering_recipe.py`, `test_pull_back_gradients.py` |
| SH13 | Prepared geometry remains a supplier precondition | `test_lowering_passes/contracts.py` |
| SH14, SH18 | Mathematical independence is a precondition; core recipes use the conservative checks below | `test_derivative_nodes/validation.py` |
| SH15 | `rules/differentiation.py::_deriv`, `algorithms/substitution.py::_occurs` | `test_derivative_nodes/validation.py` |
| SH16 | Domain check in `passes/gradients.py::_rule` | `test_gradients.py` |
| SH17 | Slot selection in `passes/actions.py::_rule`; unchanged operands in `rules/linearity.py::_linear_fields`, `rules/structural.py::_commute` | `test_actions.py`, `test_mixed.py` |
| SH19–SH21 | `algorithms/rewriting.py::_rewrite` | `test_algorithms/rewriting.py`, `test_lowering_recipe.py` |
| SH22–SH23 | Pass cleanup and `passes/lowering.py`; see findings below | `test_algorithms/rewriting.py`, `test_lowering_passes/contracts.py` |
| SH24 | Component recursion in `passes/gradients.py`; no derivative-request node | `test_gradients.py`, `test_rule_groups.py` |
| SB01–SB04 | Node `value_fields`, `scope`, `bound` and `labels` | `test_binding_identities.py`, `test_static_type.py` |
| SB05–SB10 | `algorithms/substitution.py::_substitute`, `_freshen`, `_substituted`; see findings below | `test_random.py`, `test_controls.py` |
| SB11–SB12 | `algorithms/substitution.py::_rename`, `algorithms/names.py::_renamed_label` | `test_binding_identities.py` |
| SB13–SB15 | `algorithms/substitution.py::_substitute`, `_substituted` and reconstruction | `test_spec_revision.py`, `test_binding_identities.py` |
| SB18 | `rules/differentiation.py::_deriv`, `algorithms/substitution.py::_substitute_values` | `test_derivative_nodes/validation.py`, `test_lowering_recipe.py` |
| SB19 | `rules/pullback_algebra.py::_pointwise` | `test_pull_back_gradients.py` |
| SB20–SB22 | Occurrence queries in `algorithms/substitution.py`; see findings below | `test_binding_identities.py`, `test_derivative_nodes/validation.py` |
| PB01–PB02 | Selection in `passes/pullbacks.py`, shared type completion | `test_points.py`, `test_broadcast.py` |
| PB03–PB05 | `rules/pullback_algebra.py::_composed`, `_pullback_remains` and pass selection | `test_pull_back_gradients.py`, `test_surface.py` |
| PB06 | Explicit operand-class selection in `passes/pullbacks.py`; `rules/pullback_algebra.py::_pointwise` | `test_broadcast.py`, `test_pullback_contractions.py` |
| CC02, CC04 | `rules/coordinates.py::_transformed` and pass selection | `test_push_forwards.py`, `test_pullback_contractions.py` |
| CC03, CC05 | Retention in `passes/pullbacks.py` and `rules/pullback_algebra.py`; see findings below | `test_piola_applicability.py` |
| CC08–CC09 | `rules/coordinates.py::_piola_law`, `_transformed`; selection in `passes/pullbacks.py` | `test_piola_applicability.py` |
| R01–R03 | `rules/restriction.py::_side_cell`, `_side_maps`, `_normal` and pass selection | `test_facets.py`, `test_surface.py` |
| I01 | `rules/integration.py::_mesh_integral`, `_piece_integral` | `test_integrals.py`, `test_runtime_extent.py` |

All 107 current rule IDs are accounted for. This is a source audit, not a proof
of conformance. Existing tests cover the examples listed; the findings below
need dedicated follow-up changes.

| Rules | Remaining difference |
| --- | --- |
| SB05–SB10, SB20–SB22 | `_occurs` chooses the construction walk for indices and the value walk for functions. Capture avoidance therefore does not yet have a distinct construction-occurrence query for functions stored in geometry or spaces. Directional dependence should continue to use the value walk. |
| CC03 | `expand_pull_backs` still expands ordinary divergence and curl through gradients. The current table instead retains these reads, leaving their expansion to `apply_gradients`. |
| PB01–PB03, CC02–CC05 | The pull-back retention predicate is broader than rule selection. For example, it accepts a push-forward read on a matching cell even though CC02 rewrites that read. It also accepts some domain-free side reads that PB01 rewrites. Predicate and selection need one applicability decision. |
| SH23 | `lower` checks its final contract but raises `TypeError` for unsupported output, whereas the current recipe specifies `ValueError` for an unchanged invalid state. Individual passes do not all check their declared output contract. Cycle detection is implemented. |

`test_boundaries.py` prevents mathematical rules from importing pass recipes or
output contracts. Strict Pyright checks only the interpreter package.
