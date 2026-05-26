using TestItemRunner

if "downgrade" in ARGS
    # AD-gradient items exercise Mooncake against the downgraded dep
    # set; tolerances drift below the package's pinned versions. Skip
    # quality (Aqua/JET/format/doctest) too — those are infra checks
    # that don't change with dep versions.
    @run_package_tests filter = ti -> !(:quality in ti.tags) &&
                                      !(:ad in ti.tags)
elseif "skip_quality" in ARGS
    @run_package_tests filter = ti -> !(:quality in ti.tags)
elseif "quality_only" in ARGS
    @run_package_tests filter = ti -> :quality in ti.tags
else
    @run_package_tests
end
