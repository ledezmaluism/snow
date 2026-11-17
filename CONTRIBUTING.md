# Contributing to snow

## Setup

```bash
git clone https://github.com/ledezmaluism/snow
cd snow
pip install -e ".[test,tutorials]"
pytest
```

The suite takes a few seconds. All tests must pass before merging.

## Branches

Development happens on `refactoring`; `master` holds the last stable state.
Open pull requests against `refactoring`.

## Tests

Tests live in `test/`, one file per module. `test/conftest.py` puts the
repository root on `sys.path` (so the tests also run without installing),
seeds numpy's random generator before every test, and provides two grids and
a fast waveguide (`make_ext_waveguide`, a spline of bulk MgO:LN) for
propagation tests. The geometric waveguide model is slower, so use it only
where the geometry matters.

Prefer tests that check physics over tests that pin numbers: energy
conservation, analytic results (GVD broadening of a Gaussian, exponential
loss, quadratic SHG at low conversion), dispersion relations, round trips.
Where a regression value is pinned, say in a comment where it came from.

### Known bugs are expected failures

Every known bug has a GitHub issue and a test marked

```python
@pytest.mark.xfail(strict=True, reason="#41: what is wrong")
```

The test describes the correct behaviour, so it fails today. When you fix the
bug, the test passes. `strict=True` reports that as an error until you delete
the marker, so a fix cannot land without its test being turned on. Close the
issue in the same pull request ("Fixes #41").

If a fix changes a pinned regression value, update the value, keep the old
one in a comment, and mention the change in the commit message.

To see the actual failures behind the markers:

```bash
pytest --runxfail
```

## Bugs

Report bugs as GitHub issues with:

- **What**: the function and the wrong behaviour;
- **Why it matters**: which results are affected, with numbers if possible;
- **Reproduce**: a minimal snippet;
- **Proposed fix**, if you have one;
- **Test**: the test that covers it.

## Style

- SI units everywhere; say the unit in docstrings.
- numpy-style docstrings for public functions and classes.
- Keep the conventions in [docs/physics.md](docs/physics.md); update that page
  when the model changes.
