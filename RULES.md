# Rewrite rules

The specification owns the formulas. This table records their implementation
locations and records differences found during the reorganisation.

`rules/` contains mathematical rules and applicability checks. `passes/` selects
rules, runs traversal and performs cleanup. A rule returns `None` when it does
not match. Returning the input retains it; a replacement goes through type
completion and traversal. Substitution keeps its separate binding-aware walk.

Paths below are relative to `src/uflx/_src/`.

| Rules | Implementation | Main tests in `test/` |
| --- | --- | --- |
| ST01 | `rules/structural.py::structural_fields`, `commute` | `test_actions.py`, `test_tensor_derivatives.py`, `test_broadcast.py`, `test_rule_groups.py` |
| ST02 | `passes/applications.py::_expand` | `test_actions.py`, `test_mixed.py` |
| ST03–ST04 | `rules/structural.py::application_retained`; shared with `passes/contracts.py` | `test_action_spatial.py`, `test_core_normal_form.py` |
| ST06 | `forms.py::apply`, `nodes/forms.py::Apply._validate` | `test_signatures.py` |
| ST07, ST10–ST11 | `passes/gradients.py`, `passes/pullbacks.py` | `test_action_spatial.py` |
| ST08 | `rules/structural.py::retain_gradient` | `test_gradients.py`, `test_rule_groups.py` |
| ST09 | `rules/structural.py::gradient_component` | `test_gradients.py` |
| ST13–ST16 | `passes/arithmetic.py::expand_arithmetic`; final step of `lower` | `test_tensor_arithmetic.py` |
| ST12 | `rules/structural.py::gradient_remains`; shared with `passes/contracts.py` | `test_rewriting.py`, `test_piola_applicability.py` |
| L01–L04 | `rules/linearity.py::linear_fields`, `rules/structural.py::commute` | `test_single_derivative.py`, `test_tensor_derivatives.py`, `test_broadcast.py` |
| L05, L07 | `passes/applications.py::_expand` | `test_actions.py`, `test_typed_zero.py` |
| L10 | `rules/differentiation.py::deriv` selects `zero_like` | `test_derivative_validation.py`, `test_typed_zero.py` |
| L11 | `passes/gradients.py::_rule` selects a typed component zero | `test_gradients.py`, `test_rule_groups.py` |
| D01–D10, D12 | `rules/differentiation.py::derivative_of_node` | `test_derivative_algebra.py`, `test_tensor_derivatives.py`, `test_full.py` |
| D11, D15 | `rules/differentiation.py::deriv`; value occurrence through `algorithms/substitution.py::occurs` | `test_single_derivative.py`, `test_derivative_validation.py`, `test_rule_groups.py` |
| D13 | `passes/derivatives.py::apply_derivatives`; operands first in `algorithms/rewriting.py::rewrite` | `test_lowering_recipe.py`, `test_single_derivative.py` |
| D16 | `passes/gradients.py::apply_gradients` selects compositions from `calculus.py` | `test_gradients.py`, `test_piola_applicability.py` |
| D17 | `rules/differentiation.py::coordinate_partial` | `test_gradients.py`, `test_rule_groups.py` |
| D19–D20 | `rules/differentiation.py::derivative_of_node`, before D03–D04 | `test_rule_groups.py` |
| SB16–SB17 | `algorithms/form_binding.py::partial`, `passes/applications.py::_expand` and shared substitution | `test_actions.py`, `test_controls.py` |
| B01–B02 | `passes/applications.py::_expand`, `rules/differentiation.py::derivative_of_node` | `test_current_language.py` |
| EO01 (extension) | `extensions/external_operators.py` | `test_current_language.py` |
| CC01 | `rules/coordinates.py::chain_partial` | `test_pull_back_gradients.py` |
| CC06–CC07 | `rules/coordinates.py::piola_law`; shared by rewriting and contracts | `test_piola_applicability.py` |

The remaining groups use the following locations. Substitution stays in its
existing module because it needs a different walk from expression rewriting.

| Rules | Implementation | Main tests in `test/` |
| --- | --- | --- |
| SH01 | Mathematical admissibility is a precondition; oracle comparisons check defined examples | `test_random.py`, `test_full.py` |
| SH02–SH07 | `algorithms/rewriting.py::complete_indices`, `promote`, `zero_like` | `test_index_completion.py`, `test_rule_groups.py` |
| SH08 | Promotion in `algorithms/form_binding.py` and `rules/differentiation.py`; completion before rebuilding parents | `test_single_derivative.py`, `test_index_completion.py` |
| SH09–SH11 | `algorithms/traversal.py::map_values`, node `value_fields` | `test_lowering_contracts.py`, `test_static_type.py` |
| SH12 | `algorithms/rewriting.py::rewrite`, `rules/pullback_algebra.py::composed` | `test_lowering_recipe.py`, `test_pull_back_gradients.py` |
| SH13 | Prepared geometry remains a supplier precondition | `test_lowering_contracts.py` |
| SH14, SH18 | Mathematical independence is a precondition; core recipes use the conservative checks below | `test_derivative_validation.py` |
| SH15 | `rules/differentiation.py::deriv`, `algorithms/substitution.py::occurs` | `test_derivative_validation.py` |
| SH16 | Domain check in `passes/gradients.py::_rule` | `test_gradients.py` |
| SH17 | Slot selection in `passes/applications.py::_expand`; unchanged operands in `rules/linearity.py::linear_fields`, `rules/structural.py::commute` | `test_actions.py`, `test_mixed.py` |
| SH19–SH21 | `algorithms/rewriting.py::rewrite` | `test_rewriting.py`, `test_lowering_recipe.py` |
| SH22–SH23 | Pass cleanup and `passes/lowering.py`; see findings below | `test_rewriting.py`, `test_lowering_contracts.py` |
| SH24 | Component recursion in `passes/gradients.py`; no derivative-request node | `test_gradients.py`, `test_rule_groups.py` |
| SB01–SB04 | Node `value_fields`, `scope`, `bound` and `labels` | `test_binding_identities.py`, `test_static_type.py` |
| SB05–SB10 | `algorithms/substitution.py::substitute`, `freshen`, `_substituted`; see findings below | `test_random.py`, `test_controls.py` |
| SB11–SB12 | `algorithms/substitution.py::_rename`, `algorithms/substitution.py::_renamed_label` | `test_binding_identities.py` |
| SB13–SB15 | `algorithms/substitution.py::substitute`, `_substituted` and reconstruction | `test_spec_revision.py`, `test_binding_identities.py` |
| SB18 | `rules/differentiation.py::deriv`, `algorithms/substitution.py::substitute_values` | `test_derivative_validation.py`, `test_lowering_recipe.py` |
| SB19 | `rules/pullback_algebra.py::pointwise` | `test_pull_back_gradients.py` |
| SB20–SB22 | Occurrence queries in `algorithms/substitution.py`; see findings below | `test_binding_identities.py`, `test_derivative_validation.py` |
| PB01–PB02 | Selection in `passes/pullbacks.py`, shared type completion | `test_points.py`, `test_broadcast.py` |
| PB03–PB05 | `rules/pullback_algebra.py::composed`, `pullback_remains` and pass selection | `test_pull_back_gradients.py`, `test_surface.py` |
| PB06 | Explicit operand-class selection in `passes/pullbacks.py`; `rules/pullback_algebra.py::pointwise` | `test_broadcast.py`, `test_pullback_contractions.py` |
| CC02, CC04 | `rules/coordinates.py::transformed` and pass selection | `test_push_forwards.py`, `test_pullback_contractions.py` |
| CC03, CC05 | Retention in `passes/pullbacks.py` and `rules/pullback_algebra.py`; see findings below | `test_piola_applicability.py` |
| CC08–CC09 | `rules/coordinates.py::piola_law`, `transformed`; selection in `passes/pullbacks.py` | `test_piola_applicability.py` |
| R01–R03 | `rules/restriction.py::side_cell`, `side_maps`, `pull_back_conormal` and pass selection | `test_facets.py`, `test_surface.py` |
| I01 | `rules/integration.py::mesh_integral`, `reference_integral` | `test_integrals.py`, `test_runtime_extent.py` |

Construction-dependency and value-occurrence queries in
`algorithms/substitution.py` serve capture avoidance and directional differentiation.
`expand_pull_backs` retains ordinary divergence and curl reads for the gradient pass.
The pull-back grammar describes permitted representations. Rewrite selection applies
the exclusions and precedence in the specification.
Stable invalid core output and unsupported extension operations have separate
diagnostics in `passes/lowering.py`.

`test_boundaries.py` prevents mathematical rules from importing pass recipes or
output contracts. Strict Pyright checks only the interpreter package.
