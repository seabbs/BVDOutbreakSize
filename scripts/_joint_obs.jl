# Shared helper: package loaded observations into the argument bundle
# `bvd_joint` expects, mirroring `joint_obs` in docs/examples/analysis.jl.
function _increments(v)
    d = similar(v, Int)
    prev = 0
    for i in eachindex(v)
        d[i] = v[i] - prev
        prev = v[i]
    end
    return d
end

function joint_obs(o; observe = true)
    _stream(h,
        s) = h === missing ?
             (Union{Missing, Int}[observe ? s : missing], [0]) :
             (observe ? _increments(h.values) :
              fill(missing, length(h.values)), h.offsets)
    rep, rep_off = _stream(o.reported_case_history, o.reported_cases)
    dth, dth_off = _stream(o.death_history, o.total_deaths)
    have_conf = o.confirmed_case_history !== missing ||
                o.confirmed_cases !== missing
    conf,
    conf_off = have_conf ?
               _stream(o.confirmed_case_history, o.confirmed_cases) :
               (Union{Missing, Int}[], Int[])
    edaily = observe ? o.export_deaths_daily :
             fill(missing, length(o.export_deaths_daily))
    return (deaths = dth, reported = rep, export_deaths = edaily,
        kw = (; reported_offsets = rep_off, death_offsets = dth_off,
            confirmed_cases = conf, confirmed_offsets = conf_off,
            tests_analysed = observe ? o.cumulative_tests_analysed :
                             missing, tests_offset = 0))
end
