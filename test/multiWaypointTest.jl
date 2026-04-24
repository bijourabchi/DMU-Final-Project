include("../src/multiWaypointSolver.jl")

using Plots
using Printf

# Usage:
# 1. Pick one or more cases in ACTIVE_CASES.
# 2. Edit TEST_CASES, ROUTE_PRESETS, or OBSTACLE_PRESETS as needed.
# 3. Run this file. Outputs are saved under figures/multiWaypoint/<case>/.

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const FIGURE_ROOT = joinpath(PROJECT_ROOT, "figures", "multiWaypoint")
const SHOW_PLOTS = true
const PLANNER_NAME = "MCTS"
const REWARD_MAP = 7
const DETOUR_BUDGET = 10

rectangle_obstacles(rows, cols) = [(r, c) for r in rows for c in cols]

function combine_obstacles(groups...)
    obstacles = Tuple{Int, Int}[]
    for group in groups
        append!(obstacles, group)
    end
    return unique(obstacles)
end

const ROUTE_PRESETS = Dict(
    :triangle => (
        start = (25, 40),
        waypoints = [(20, 30), (30, 30)],
    ),
    :diamond => (
        start = (25, 40),
        waypoints = [(20, 30), (30, 30), (25, 20), (35, 25)],
    ),
    :corridor => (
        start = (15, 18),
        waypoints = [(18, 30), (22, 42), (30, 56), (38, 70)],
    ),
    :circle => (
        start = (25, 33),
        waypoints = [(19, 38), (19, 48), (25, 53), (31, 48), (31, 38)],
    ),
)

const OBSTACLE_PRESETS = Dict{Symbol, Vector{Tuple{Int, Int}}}(
    :none => Tuple{Int, Int}[],
    :center_block => rectangle_obstacles(25:33, 31:33),
    :detour_wall => combine_obstacles(
        rectangle_obstacles(20:40, 36:38),
        rectangle_obstacles(20:22, 39:55),
    ),
    :split_corridor => combine_obstacles(
        rectangle_obstacles(12:24, 24:26),
        rectangle_obstacles(26:42, 48:50),
    ),
    :circle_bar => rectangle_obstacles(15:35, 42:44),
)

const TEST_CASES = Dict(
    :baseline => (
        description = "Original triangle case on reward map 2 with no obstacles.",
        reward_map = REWARD_MAP,
        route = :triangle,
        obstacles = :none,
        detour_budget = DETOUR_BUDGET,
    ),
    :center_block => (
        description = "Triangle case with the original center obstacle block enabled.",
        reward_map = REWARD_MAP,
        route = :triangle,
        obstacles = :center_block,
        detour_budget = DETOUR_BUDGET,
    ),
    :detour_wall => (
        description = "Larger waypoint set with a wall forcing a detour.",
        reward_map = REWARD_MAP,
        route = :diamond,
        obstacles = :detour_wall,
        detour_budget = DETOUR_BUDGET,
    ),
    :corridor_sweep => (
        description = "Longer corridor-style path across a different reward map.",
        reward_map = REWARD_MAP,
        route = :corridor,
        obstacles = :split_corridor,
        detour_budget = DETOUR_BUDGET,
    ),
    :circle => (
        description = "Circular waypoint layout with a vertical obstacle bar cutting between waypoint groups.",
        reward_map = REWARD_MAP,
        route = :circle,
        obstacles = :circle_bar,
        detour_budget = DETOUR_BUDGET,
    ),
)

const CASE_ORDER = [:baseline, :center_block, :detour_wall, :corridor_sweep, :circle]
const ACTIVE_CASES = CASE_ORDER
#const ACTIVE_CASES = [:circle]

function sanitize_name(value)
    return replace(lowercase(string(value)), r"[^a-z0-9_-]+" => "_")
end

function pretty_name(value)
    return join(uppercasefirst.(split(string(value), "_")), " ")
end

function resolve_reward_map(spec)
    if spec isa Integer
        return WaypointPath.get_reward(spec), "Map $(spec)", "map$(spec)"
    elseif spec isa AbstractMatrix
        return Matrix{Float64}(spec), "Custom Reward Map", "custom_reward_map"
    end

    throw(ArgumentError("Unsupported reward map specification: $(typeof(spec))"))
end

function resolve_route(spec)
    if spec isa Symbol
        route = ROUTE_PRESETS[spec]
        return route.start, copy(route.waypoints)
    elseif spec isa NamedTuple
        return spec.start, collect(spec.waypoints)
    end

    throw(ArgumentError("Unsupported route specification: $(typeof(spec))"))
end

function resolve_obstacles(spec)
    if spec isa Symbol
        return copy(OBSTACLE_PRESETS[spec])
    elseif spec isa AbstractVector{<:Tuple{Int, Int}}
        return unique(collect(spec))
    end

    throw(ArgumentError("Unsupported obstacle specification: $(typeof(spec))"))
end

function validate_start(start::Tuple{Int, Int},
                        reward_map::Matrix{Float64},
                        obstacles::Vector{Tuple{Int, Int}},
                        case_name::Symbol)
    if !WaypointPath.GenerateMDP.inBounds(start, reward_map)
        throw(ArgumentError("Start $(start) is out of bounds for case $(case_name)."))
    end

    if start in Set(obstacles)
        throw(ArgumentError("Start $(start) lies inside an obstacle for case $(case_name)."))
    end
end

function to_plot_coords(points::AbstractVector{<:Tuple{Int, Int}})
    xs = [point[2] for point in points]
    ys = [point[1] for point in points]
    return xs, ys
end

function extract_visited_waypoints(hist::Vector{Tuple{Int, Int}},
                                   waypoints::Vector{Tuple{Int, Int}})
    waypoint_set = Set(waypoints)
    seen = Set{Tuple{Int, Int}}()
    ordered_waypoints = Tuple{Int, Int}[]

    for state in hist
        if state in waypoint_set && !(state in seen)
            push!(ordered_waypoints, state)
            push!(seen, state)
        end
    end

    return ordered_waypoints
end

function annotate_waypoints!(plot_obj, waypoints::Vector{Tuple{Int, Int}})
    for (idx, point) in enumerate(waypoints)
        annotate!(plot_obj, point[2], point[1], text(string(idx), :black, 9))
    end
end

travel_distance(hist::Vector{Tuple{Int, Int}}) = max(length(hist) - 1, 0)

function format_reward(value::Float64)
    return isfinite(value) ? @sprintf("%.2f", value) : string(value)
end

function format_budget_value(value::Real)
    return isfinite(value) ? @sprintf("%.1f", Float64(value)) : "Inf"
end

function format_detour_budgets(budgets::Vector{Float64})
    if isempty(budgets)
        return "n/a"
    end

    rounded = [isfinite(value) ? round(value, digits = 1) : Inf for value in budgets]
    unique_values = unique(rounded)

    if length(unique_values) == 1
        return format_budget_value(unique_values[1])
    end

    if length(unique_values) <= 3
        return "[" * join(format_budget_value.(unique_values), ", ") * "]"
    end

    finite_values = filter(isfinite, unique_values)
    if isempty(finite_values)
        return "[Inf]"
    end

    upper = any(!isfinite(value) for value in unique_values) ? "Inf" : format_budget_value(maximum(finite_values))
    return "$(format_budget_value(minimum(finite_values)))-$(upper)"
end

function plot_solution(reward_map::Matrix{Float64},
                       start::Tuple{Int, Int},
                       configured_waypoints::Vector{Tuple{Int, Int}},
                       visited_waypoints::Vector{Tuple{Int, Int}},
                       hist::Vector{Tuple{Int, Int}},
                       obstacles::Vector{Tuple{Int, Int}},
                       case_name::Symbol,
                       reward_map_title::String,
                       planner_name::String,
                       total_reward::Float64,
                       distance_travelled::Int,
                       detour_label::String)
    p = heatmap(
        reward_map,
        title = "$(planner_name) Multi-Waypoint Solution: $(pretty_name(case_name))\n$(reward_map_title) | Reward / Distance = $(round(total_reward / distance_travelled, digits = 3)) | Detour Budget = $(detour_label)",
        xlabel = "Column",
        ylabel = "Row",
        colorbar = true,
        colorbar_title = "Reward",
        color = :plasma,
        clims = (minimum(reward_map), maximum(reward_map)),
        size = (1100, 850),
        margin = 5Plots.mm,
        right_margin = 12Plots.mm,
        bottom_margin = 12Plots.mm,
        xlim = (0.5, size(reward_map, 2) + 0.5),
        ylim = (0.5, size(reward_map, 1) + 0.5),
        xticks = (1:10:size(reward_map, 2), string.(1:10:size(reward_map, 2))),
        yticks = (1:10:size(reward_map, 1), string.(1:10:size(reward_map, 1))),
        legend = :outerbottom,
        framestyle = :box,
        dpi = 170,
    )

    if !isempty(obstacles)
        obs_x, obs_y = to_plot_coords(obstacles)
        scatter!(
            p,
            obs_x,
            obs_y,
            label = "Obstacles",
            color = :black,
            markersize = 4,
            markerstrokewidth = 0,
            alpha = 0.85,
        )
    end

    if !isempty(hist)
        hist_x, hist_y = to_plot_coords(hist)
        plot!(
            p,
            hist_x,
            hist_y,
            label = "Solution Path",
            color = :red,
            linewidth = 2.3,
            alpha = 0.85,
        )
        scatter!(
            p,
            hist_x,
            hist_y,
            label = "Visited Cells",
            color = :red,
            markersize = 3,
            markerstrokewidth = 0,
            alpha = 0.5,
        )
    end

    if !isempty(configured_waypoints)
        wp_x, wp_y = to_plot_coords(configured_waypoints)
        scatter!(
            p,
            wp_x,
            wp_y,
            label = "Configured Waypoints",
            color = :white,
            markersize = 8,
            markerstrokecolor = :black,
            markerstrokewidth = 1.3,
        )
    end

    if !isempty(visited_waypoints)
        ordered_x, ordered_y = to_plot_coords(visited_waypoints)
        scatter!(
            p,
            ordered_x,
            ordered_y,
            label = "Visited Waypoints",
            color = :lime,
            markersize = 9,
            markerstrokecolor = :black,
            markerstrokewidth = 1.3,
        )
        annotate_waypoints!(p, visited_waypoints)
    end

    scatter!(
        p,
        [start[2]],
        [start[1]],
        label = "Start",
        color = :dodgerblue,
        markersize = 11,
        markerstrokecolor = :black,
        markerstrokewidth = 1.4,
    )

    if !isempty(hist)
        finish = hist[end]
        scatter!(
            p,
            [finish[2]],
            [finish[1]],
            label = "End",
            color = :gold,
            markersize = 11,
            markerstrokecolor = :black,
            markerstrokewidth = 1.4,
        )
    end

    return p
end

function save_plot(plot_obj, output_dir::String, filename::String)
    savefig(plot_obj, joinpath(output_dir, filename))
end

function write_summary(output_dir::String,
                       case_name::Symbol,
                       case_config,
                       reward_map::Matrix{Float64},
                       reward_map_title::String,
                       start::Tuple{Int, Int},
                       configured_waypoints::Vector{Tuple{Int, Int}},
                       filtered_waypoints::Vector{Tuple{Int, Int}},
                       visited_waypoints::Vector{Tuple{Int, Int}},
                       obstacles::Vector{Tuple{Int, Int}},
                       solution,
                       distance_travelled::Int,
                       output_filename::String,
                       summary_filename::String)
    summary_path = joinpath(output_dir, summary_filename)

    function write_summary_file(path::String)
        open(path, "w") do io
            println(io, "Case: $(case_name)")
            println(io, "Description: $(case_config.description)")
            println(io, "Reward Map Spec: $(case_config.reward_map)")
            println(io, "Reward Map Label: $(reward_map_title)")
            println(io, "Grid Size: $(size(reward_map, 1)) x $(size(reward_map, 2))")
            println(io, "Start: $(start)")
            println(io, "Obstacle Count: $(length(obstacles))")
            println(io, "Configured Waypoint Count: $(length(configured_waypoints))")
            println(io, "Filtered Waypoint Count: $(length(filtered_waypoints))")
            println(io, "Ordered Waypoints: $(solution.ordered_wp)")
            println(io, "Visited Waypoints: $(visited_waypoints)")
            println(io, @sprintf("Total Reward: %s", format_reward(solution.rtot)))
            println(io, "Travel Distance (Grid Steps): $(distance_travelled)")
            println(io, "Detour Budgets Used: $(solution.detour_budgets)")
            println(io, "Configured Detour Budget: $(case_config.detour_budget)")
            println(io, "Figure: $(output_filename)")
            println(io)
            println(io, "Configured Waypoints:")
            for (idx, point) in enumerate(configured_waypoints)
                println(io, @sprintf("  W%-2d -> (%d, %d)", idx, point[1], point[2]))
            end
            println(io)
            println(io, "Ordered Waypoints:")
            for (idx, point) in enumerate(solution.ordered_wp)
                println(io, @sprintf("  %d -> (%d, %d)", idx, point[1], point[2]))
            end
            println(io)
            println(io, "Visited Waypoints:")
            for (idx, point) in enumerate(visited_waypoints)
                println(io, @sprintf("  %d -> (%d, %d)", idx, point[1], point[2]))
            end
        end
    end

    write_summary_file(summary_path)
    write_summary_file(joinpath(output_dir, "summary.txt"))
end

function run_case(case_name::Symbol, case_config)
    reward_map, reward_map_title, reward_map_slug = resolve_reward_map(case_config.reward_map)
    start, configured_waypoints = resolve_route(case_config.route)
    obstacles = resolve_obstacles(case_config.obstacles)
    validate_start(start, reward_map, obstacles, case_name)

    filtered_waypoints = WaypointPath.filter_wp(configured_waypoints, reward_map; obs = obstacles)
    output_dir = joinpath(FIGURE_ROOT, sanitize_name(case_name))
    mkpath(output_dir)

    solution = WaypointPath.planner(
        PLANNER_NAME,
        reward_map,
        start,
        configured_waypoints;
        obstacles = obstacles,
        detour_budget = case_config.detour_budget,
    )
    visited_waypoints = extract_visited_waypoints(solution.hist, configured_waypoints)
    distance_travelled = travel_distance(solution.hist)
    detour_label = format_detour_budgets(solution.detour_budgets)

    plot_obj = plot_solution(
        reward_map,
        start,
        configured_waypoints,
        visited_waypoints,
        solution.hist,
        obstacles,
        case_name,
        reward_map_title,
        PLANNER_NAME,
        solution.rtot,
        distance_travelled,
        detour_label,
    )

    output_filename = "01_$(sanitize_name(PLANNER_NAME))_$(reward_map_slug)_detour_$(sanitize_name(detour_label)).png"
    summary_filename = "summary_$(reward_map_slug)_detour_$(sanitize_name(detour_label)).txt"
    save_plot(plot_obj, output_dir, output_filename)

    write_summary(
        output_dir,
        case_name,
        case_config,
        reward_map,
        reward_map_title,
        start,
        configured_waypoints,
        filtered_waypoints,
        visited_waypoints,
        obstacles,
        solution,
        distance_travelled,
        output_filename,
        summary_filename,
    )

    println()
    println("Case $(case_name) complete.")
    println("  Description: $(case_config.description)")
    println("  Output Directory: $(output_dir)")
    println("  Ordered Waypoints: $(solution.ordered_wp)")
    println("  Visited Waypoints: $(visited_waypoints)")
    println("  Travel Distance: $(distance_travelled)")
    println("  Detour Budgets: $(solution.detour_budgets)")
    println(@sprintf("  Reward: %s", format_reward(solution.rtot)))

    if SHOW_PLOTS
        display(plot_obj)
    end
end

function main()
    mkpath(FIGURE_ROOT)

    for case_name in ACTIVE_CASES
        if !haskey(TEST_CASES, case_name)
            throw(ArgumentError("Unknown test case $(case_name). Available cases: $(collect(keys(TEST_CASES)))"))
        end
        println("Running test case $(case_name)")
        run_case(case_name, TEST_CASES[case_name])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
