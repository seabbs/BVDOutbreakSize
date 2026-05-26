using JET: test_opt
using BVDOutbreakSize

# Type-stability spot check on the gamma CDF rrule plumbing. Kept here
# (in an isolated env) so JET version constraints do not bleed into
# the main test environment.
test_opt(BVDOutbreakSize._gamma_cdf, (Float64, Float64, Float64);
    target_modules = (BVDOutbreakSize,))
