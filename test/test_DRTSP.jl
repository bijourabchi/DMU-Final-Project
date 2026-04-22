include("../src/discountedRewardTSP.jl")
include("../src/generateMDP.jl")
include("../src/genWavefront.jl")

using Plots
using Printf

# Usage:
# 1. Pick one or more cases in ACTIVE_CASES.
# 2. Edit TEST_CASES, WAYPOINT_PRESETS, or OBSTACLE_PRESETS as needed.
# 3. Run this file. Outputs are saved under figures/discountedRewardTSP/<case>/.

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const FIGURE_ROOT = joinpath(PROJECT_ROOT, "figures", "discountedRewardTSP")
const SHOW_PLOTS = true
const SAVE_WAVEFRONTS_BY_DEFAULT = true
const WAVE_ACTIONS = (
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (-1, 1), (1, -1), (-1, -1),
)

rectangle_obstacles(rows, cols) = [(r, c) for r in rows for c in cols]

function combine_obstacles(groups...)
    obstacles = Tuple{Int, Int}[]
    for group in groups
        append!(obstacles, group)
    end
    return unique(obstacles)
end

const WAYPOINT_PRESETS = Dict{Symbol, Vector{Tuple{Int, Int}}}(
    :triangle => [(25, 40), (20, 30), (30, 30)],
    :diamond => [(25, 40), (20, 30), (30, 30), (25, 20), (35, 25)],
    :corridor => [(15, 18), (18, 30), (22, 42), (30, 56), (38, 70)],
    :circle => [(25, 25), (20, 35), (20, 45), (25, 50), (30, 45), (30, 35)]
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
    :wall => rectangle_obstacles(20:40,37:42)
)

const TEST_CASES = Dict(
    :baseline => (
        description = "Original triangle case on reward map 2 with no obstacles.",
        reward_map = 2,
        waypoints = :triangle,
        obstacles = :none,
        save_wavefronts = true,
    ),
    :center_block => (
        description = "Triangle case with the original center obstacle block enabled.",
        reward_map = 2,
        waypoints = :triangle,
        obstacles = :center_block,
        save_wavefronts = true,
    ),
    :detour_wall => (
        description = "Larger waypoint set with a wall forcing a detour.",
        reward_map = 3,
        waypoints = :diamond,
        obstacles = :detour_wall,
        save_wavefronts = true,
    ),
    :corridor_sweep => (
        description = "Longer corridor-style path across a different reward map.",
        reward_map = 5,
        waypoints = :corridor,
        obstacles = :split_corridor,
        save_wavefronts = true,
    ),
    :circle => (
        description = "Circular Order of Waypoints",
        reward_map = 9,
        waypoints = :circle,
        obstacles = :wall,
        save_wavefronts = true,

    ),
)

const ACTIVE_CASES = [:center_block]
# const ACTIVE_CASES = collect(keys(TEST_CASES))

function sanitize_name(value)
    return replace(lowercase(string(value)), r"[^a-z0-9_-]+" => "_")
end

function resolve_reward_map(spec)
    if spec isa Integer
        return GenerateMDP.get_reward_map(spec)
    elseif spec isa AbstractMatrix
        return Matrix{Float64}(spec)
    end

    throw(ArgumentError("Unsupported reward map specification: $(typeof(spec))"))
end

function resolve_waypoints(spec)
    if spec isa Symbol
        return copy(WAYPOINT_PRESETS[spec])
    elseif spec isa AbstractVector{<:Tuple{Int, Int}}
        return collect(spec)
    end

    throw(ArgumentError("Unsupported waypoint specification: $(typeof(spec))"))
end

function resolve_obstacles(spec)
    if spec isa Symbol
        return copy(OBSTACLE_PRESETS[spec])
    elseif spec isa AbstractVector{<:Tuple{Int, Int}}
        return unique(collect(spec))
    end

    throw(ArgumentError("Unsupported obstacle specification: $(typeof(spec))"))
end

function grid_inbounds(cell::Tuple{Int, Int}, grid)
    r, c = cell
    return 1 <= r <= size(grid, 1) && 1 <= c <= size(grid, 2)
end

function to_plot_coords(points::AbstractVector{<:Tuple{Int, Int}})
    xs = [point[2] for point in points]
    ys = [point[1] for point in points]
    return xs, ys
end

function append_segment!(route::Vector{Tuple{Int, Int}}, segment::Vector{Tuple{Int, Int}})
    if isempty(segment)
        return
    end

    if isempty(route)
        append!(route, segment)
    elseif route[end] == segment[1]
        append!(route, segment[2:end])
    else
        append!(route, segment)
    end
end

function trace_wavefront_path(start::Tuple{Int, Int}, goal::Tuple{Int, Int}, wf::Matrix{Int})
    if start == goal
        return [start]
    end

    if !grid_inbounds(start, wf) || !grid_inbounds(goal, wf)
        return nothing
    end

    start_value = wf[start[1], start[2]]
    if start_value in (0, 1)
        return nothing
    end

    path = [start]
    seen = Set(path)
    max_steps = prod(size(wf))

    while path[end] != goal && length(path) <= max_steps
        current = path[end]
        current_value = wf[current[1], current[2]]

        best_neighbor = nothing
        best_value = current_value

        for (dr, dc) in WAVE_ACTIONS
            neighbor = (current[1] + dr, current[2] + dc)

            if !grid_inbounds(neighbor, wf)
                continue
            end

            neighbor_value = wf[neighbor[1], neighbor[2]]
            if neighbor_value in (0, 1)
                continue
            end

            if neighbor_value < best_value
                best_neighbor = neighbor
                best_value = neighbor_value
            end
        end

        if isnothing(best_neighbor) || best_neighbor in seen
            return nothing
        end

        push!(path, best_neighbor)
        push!(seen, best_neighbor)
    end

    return path[end] == goal ? path : nothing
end

function reconstruct_route(ordered_waypoints::Vector{Tuple{Int, Int}},
                           reward_map::Matrix{Float64},
                           obstacles::Vector{Tuple{Int, Int}})
    legs = NamedTuple[]
    full_route = Tuple{Int, Int}[]

    for leg_idx in 1:max(0, length(ordered_waypoints) - 1)
        start_wp = ordered_waypoints[leg_idx]
        goal_wp = ordered_waypoints[leg_idx + 1]

        wf = genWavefront.gen_wave(goal_wp, zeros(Int, size(reward_map)); obs = obstacles)
        leg_path = trace_wavefront_path(start_wp, goal_wp, wf)
        leg_cost = isnothing(leg_path) ? Inf : (length(leg_path) - 1)

        push!(legs, (
            index = leg_idx,
            start = start_wp,
            goal = goal_wp,
            path = leg_path,
            cost = leg_cost,
            wavefront = wf,
        ))

        if !isnothing(leg_path)
            append_segment!(full_route, leg_path)
        end
    end

    return legs, full_route
end

function base_reward_plot(reward_map::Matrix{Float64}; title::String)
    return heatmap(
        reward_map,
        title = title,
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
end

function add_obstacles!(plot_obj, obstacles::Vector{Tuple{Int, Int}})
    if isempty(obstacles)
        return
    end

    obs_x, obs_y = to_plot_coords(obstacles)
    scatter!(
        plot_obj,
        obs_x,
        obs_y,
        label = "Obstacles",
        color = :black,
        markersize = 4,
        markerstrokewidth = 0,
        alpha = 0.85,
    )
end

function annotate_waypoints!(plot_obj,
                             points::Vector{Tuple{Int, Int}},
                             labels::Vector{String};
                             color = :white,
                             chip_color = :black,
                             chip_size = 11)
    xs, ys = to_plot_coords(points)
    scatter!(
        plot_obj,
        xs,
        ys,
        label = "",
        color = chip_color,
        alpha = 0.72,
        markersize = chip_size,
        markerstrokecolor = :white,
        markerstrokewidth = 1.2,
    )

    for idx in eachindex(points)
        annotate!(plot_obj, xs[idx], ys[idx], text(labels[idx], color, 9, :bold))
    end
end

function plot_environment(reward_map::Matrix{Float64},
                          raw_waypoints::Vector{Tuple{Int, Int}},
                          filtered_waypoints::Vector{Tuple{Int, Int}},
                          obstacles::Vector{Tuple{Int, Int}},
                          case_name::Symbol)
    p = base_reward_plot(reward_map; title = "Environment: $(case_name)")
    add_obstacles!(p, obstacles)

    raw_x, raw_y = to_plot_coords(raw_waypoints)
    scatter!(
        p,
        raw_x,
        raw_y,
        label = "Configured Waypoints",
        color = :white,
        markersize = 8,
        markerstrokecolor = :black,
        markerstrokewidth = 1.5,
    )

    filtered_x, filtered_y = to_plot_coords(filtered_waypoints)
    scatter!(
        p,
        filtered_x,
        filtered_y,
        label = "In-Bounds Waypoints",
        color = :cyan,
        markersize = 6,
        markerstrokecolor = :black,
        markerstrokewidth = 1.0,
    )

    labels = ["W$(idx)" for idx in eachindex(raw_waypoints)]
    annotate_waypoints!(p, raw_waypoints, labels)

    return p
end

function plot_waypoint_solution(reward_map::Matrix{Float64},
                                filtered_waypoints::Vector{Tuple{Int, Int}},
                                ordered_waypoints::Vector{Tuple{Int, Int}},
                                solution_path::Vector{Int},
                                obstacles::Vector{Tuple{Int, Int}},
                                case_name::Symbol)
    p = base_reward_plot(reward_map; title = "Waypoint Ordering: $(case_name)")
    add_obstacles!(p, obstacles)

    all_x, all_y = to_plot_coords(filtered_waypoints)
    scatter!(
        p,
        all_x,
        all_y,
        label = "Candidate Waypoints",
        color = :white,
        markersize = 7,
        markerstrokecolor = :black,
        markerstrokewidth = 1.2,
    )

    if !isempty(ordered_waypoints)
        route_x, route_y = to_plot_coords(ordered_waypoints)
        plot!(
            p,
            route_x,
            route_y,
            label = "Waypoint Order",
            color = :red,
            linewidth = 2.5,
            alpha = 0.85,
        )

        scatter!(
            p,
            route_x,
            route_y,
            label = "Visited Waypoints",
            color = :lime,
            markersize = 9,
            markerstrokecolor = :black,
            markerstrokewidth = 1.2,
        )

        scatter!(
            p,
            [route_x[1]],
            [route_y[1]],
            label = "Start",
            color = :dodgerblue,
            markersize = 11,
            markerstrokecolor = :black,
            markerstrokewidth = 1.4,
        )

        scatter!(
            p,
            [route_x[end]],
            [route_y[end]],
            label = "End",
            color = :gold,
            markersize = 11,
            markerstrokecolor = :black,
            markerstrokewidth = 1.4,
        )

        visit_labels = [
            @sprintf("%d (W%d)", visit_idx, solution_path[visit_idx])
            for visit_idx in eachindex(solution_path)
        ]
        annotate_waypoints!(p, ordered_waypoints, visit_labels)
    end

    return p
end

function plot_full_route(reward_map::Matrix{Float64},
                         filtered_waypoints::Vector{Tuple{Int, Int}},
                         ordered_waypoints::Vector{Tuple{Int, Int}},
                         full_route::Vector{Tuple{Int, Int}},
                         obstacles::Vector{Tuple{Int, Int}},
                         case_name::Symbol)
    p = base_reward_plot(reward_map; title = "Low-Level Route: $(case_name)")
    add_obstacles!(p, obstacles)

    all_x, all_y = to_plot_coords(filtered_waypoints)
    scatter!(
        p,
        all_x,
        all_y,
        label = "Candidate Waypoints",
        color = :white,
        markersize = 6,
        markerstrokecolor = :black,
        markerstrokewidth = 1.0,
    )

    if !isempty(full_route)
        route_x, route_y = to_plot_coords(full_route)
        plot!(
            p,
            route_x,
            route_y,
            label = "Cell-by-Cell Route",
            color = :red,
            linewidth = 2.0,
            alpha = 0.9,
        )
    end

    if !isempty(ordered_waypoints)
        ord_x, ord_y = to_plot_coords(ordered_waypoints)
        scatter!(
            p,
            ord_x,
            ord_y,
            label = "Ordered Waypoints",
            color = :lime,
            markersize = 8,
            markerstrokecolor = :black,
            markerstrokewidth = 1.0,
        )

        annotate_waypoints!(p, ordered_waypoints, ["$(idx)" for idx in eachindex(ordered_waypoints)])
    end

    return p
end

function plot_leg_wavefront(leg, obstacles::Vector{Tuple{Int, Int}}, case_name::Symbol)
    p = heatmap(
        leg.wavefront,
        title = "Wavefront Leg $(leg.index): $(case_name)",
        xlabel = "Column",
        ylabel = "Row",
        colorbar = true,
        colorbar_title = "Wavefront Value",
        color = :viridis,
        size = (1000, 760),
        margin = 5Plots.mm,
        right_margin = 12Plots.mm,
        bottom_margin = 12Plots.mm,
        xlim = (0.5, size(leg.wavefront, 2) + 0.5),
        ylim = (0.5, size(leg.wavefront, 1) + 0.5),
        legend = :outerbottom,
        framestyle = :box,
        dpi = 170,
    )

    add_obstacles!(p, obstacles)

    if !isnothing(leg.path)
        path_x, path_y = to_plot_coords(leg.path)
        plot!(
            p,
            path_x,
            path_y,
            label = "Leg Path",
            color = :red,
            linewidth = 2.0,
        )
    end

    start_x, start_y = to_plot_coords([leg.start])
    goal_x, goal_y = to_plot_coords([leg.goal])

    scatter!(
        p,
        start_x,
        start_y,
        label = "Leg Start",
        color = :dodgerblue,
        markersize = 9,
        markerstrokecolor = :black,
    )

    scatter!(
        p,
        goal_x,
        goal_y,
        label = "Leg Goal",
        color = :gold,
        markersize = 9,
        markerstrokecolor = :black,
    )

    return p
end

function write_summary(output_dir::String,
                       case_name::Symbol,
                       case_config,
                       reward_map::Matrix{Float64},
                       raw_waypoints::Vector{Tuple{Int, Int}},
                       filtered_waypoints::Vector{Tuple{Int, Int}},
                       obstacles::Vector{Tuple{Int, Int}},
                       solution,
                       ordered_waypoints::Vector{Tuple{Int, Int}},
                       legs)
    summary_path = joinpath(output_dir, "summary.txt")

    open(summary_path, "w") do io
        println(io, "Case: $(case_name)")
        println(io, "Description: $(case_config.description)")
        println(io, "Reward Map Spec: $(case_config.reward_map)")
        println(io, "Grid Size: $(size(reward_map, 1)) x $(size(reward_map, 2))")
        println(io, "Obstacle Count: $(length(obstacles))")
        println(io, "Raw Waypoint Count: $(length(raw_waypoints))")
        println(io, "Filtered Waypoint Count: $(length(filtered_waypoints))")
        println(io, "Solution Path Indices: $(solution.path)")
        println(io, "Ordered Waypoints: $(ordered_waypoints)")
        println(io, @sprintf("True Discounted Reward: %.6f", solution.reward))
        println(io, @sprintf("Scaled Prize Collected: %.6f", solution.scaled_prize))
        println(io, @sprintf("Travel Cost: %.6f", solution.travel_cost))
        println(io)
        println(io, "Configured Waypoints:")
        for (idx, point) in enumerate(raw_waypoints)
            println(io, @sprintf("  W%-2d -> (%d, %d)", idx, point[1], point[2]))
        end
        println(io)
        println(io, "Ordered Visit Sequence:")
        for (idx, point) in enumerate(ordered_waypoints)
            println(io, @sprintf("  %d -> (%d, %d)", idx, point[1], point[2]))
        end
        println(io)
        println(io, "Leg Details:")
        for leg in legs
            if isnothing(leg.path)
                println(io, "  Leg $(leg.index): $(leg.start) -> $(leg.goal) : no reconstructed path")
            else
                println(io, @sprintf(
                    "  Leg %d: %s -> %s : %d cells, cost %.1f",
                    leg.index,
                    string(leg.start),
                    string(leg.goal),
                    length(leg.path),
                    leg.cost,
                ))
            end
        end
    end
end

function save_plot(plot_obj, output_dir::String, filename::String)
    savefig(plot_obj, joinpath(output_dir, filename))
end

function cleanup_case_output!(output_dir::String)
    if !isdir(output_dir)
        return
    end

    for entry in readdir(output_dir; join = true)
        if isfile(entry) && (endswith(entry, ".png") || endswith(entry, ".txt"))
            rm(entry)
        end
    end
end

function run_case(case_name::Symbol, case_config)
    reward_map = resolve_reward_map(case_config.reward_map)
    raw_waypoints = resolve_waypoints(case_config.waypoints)
    obstacles = resolve_obstacles(case_config.obstacles)
    filtered_waypoints = waypointOrdering.filter_wp(raw_waypoints, reward_map)

    output_dir = joinpath(FIGURE_ROOT, sanitize_name(case_name))
    mkpath(output_dir)
    cleanup_case_output!(output_dir)

    solver = waypointOrdering.simpleMinExcess()
    waypoint_solver = waypointOrdering.blackboxMinExcess()
    solution = waypointOrdering.order_wp(filtered_waypoints, reward_map; solver = solver, waypointSolver = waypoint_solver, obs = obstacles)

    ordered_waypoints = filtered_waypoints[solution.path]
    legs, full_route = reconstruct_route(ordered_waypoints, reward_map, obstacles)

    environment_plot = plot_environment(reward_map, raw_waypoints, filtered_waypoints, obstacles, case_name)
    waypoint_plot = plot_waypoint_solution(reward_map, filtered_waypoints, ordered_waypoints, solution.path, obstacles, case_name)
    full_route_plot = plot_full_route(reward_map, filtered_waypoints, ordered_waypoints, full_route, obstacles, case_name)

    save_plot(environment_plot, output_dir, "01_environment.png")
    save_plot(waypoint_plot, output_dir, "02_waypoint_solution.png")
    save_plot(full_route_plot, output_dir, "03_full_route.png")

    save_wavefronts = case_config.save_wavefronts
    if save_wavefronts
        for leg in legs
            leg_plot = plot_leg_wavefront(leg, obstacles, case_name)
            save_plot(leg_plot, output_dir, @sprintf("wavefront_leg_%02d.png", leg.index))
        end
    end

    write_summary(
        output_dir,
        case_name,
        case_config,
        reward_map,
        raw_waypoints,
        filtered_waypoints,
        obstacles,
        solution,
        ordered_waypoints,
        legs,
    )

    println()
    println("Case $(case_name) complete.")
    println("  Description: $(case_config.description)")
    println("  Output Directory: $(output_dir)")
    println("  Solution Path: $(solution.path)")
    println("  Ordered Waypoints: $(ordered_waypoints)")
    println(@sprintf("  Reward: %.6f", solution.reward))
    println(@sprintf("  Cost: %.6f", solution.travel_cost))

    if SHOW_PLOTS
        display(environment_plot)
        display(waypoint_plot)
        display(full_route_plot)
    end
end

function main()
    mkpath(FIGURE_ROOT)

    for case_name in ACTIVE_CASES
        if !haskey(TEST_CASES, case_name)
            throw(ArgumentError("Unknown test case $(case_name). Available cases: $(collect(keys(TEST_CASES)))"))
        end

        run_case(case_name, TEST_CASES[case_name])
    end
end

main()
