# Test Maintenance Notes

ProbeFlow's test suite should protect scientific correctness, parsing, provenance,
public CLI/API behavior, and release-safety contracts. Prefer compact contract
tests over many tiny implementation-detail assertions.

This cleanup removed or consolidated:

- GUI refactor scaffolding tests that only checked private widget layout,
  module extraction paths, or compatibility re-export details.
- Micro-tests that asserted one field at a time for display, ROI, provenance,
  and processing-state serialization contracts.
- Dialog `__new__` private-method tests where helper-level or user-facing
  workflow coverage already protects the behavior.

Keep tests when they exercise units, coordinates, data lineage, NaN handling,
file readers/writers, provenance sidecars, scientific processing, or documented
public behavior. Avoid adding tests whose only purpose is to pin private names,
old layout choices, or completed refactor seams.

Happy-path audit categories:

- Protective: keep tests that would catch realistic regressions in scientific
  values, physical units, coordinates, parser behavior, provenance, export file
  contents, NaN handling, invalid input handling, or public API behavior.
- Weak but salvageable: rewrite tests that only check an object exists, a path
  was reached, or a single field has a routine value so they assert an invariant
  or an edge case in the same test.
- Redundant: merge repeated one-field checks when a single contract test can
  protect the same behavior.
- Delete: remove tests that only check implementation plumbing, private module
  placement, trivial construction, labels, or format existence without checking
  format content.
