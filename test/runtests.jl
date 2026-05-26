using TestItemRunner

if "skip_quality" in ARGS
    @run_package_tests filter = ti -> !(:quality in ti.tags)
elseif "quality_only" in ARGS
    @run_package_tests filter = ti -> :quality in ti.tags
else
    @run_package_tests
end
