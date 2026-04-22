include("../src/multiWaypointSolver.jl")

using Plots

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const FIGURE_DIR = joinpath(PROJECT_ROOT, "figures", "multiWaypoint")

rectangle_obstacles(rows, cols) = [(r, c) for r in rows for c in cols]

function combine_obstacles(groups...)
    obstacles = Tuple{Int, Int}[]
    for group in groups
        append!(obstacles, group)
    end
    return unique(obstacles)
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
        annotate!(plot_obj, point[2], point[1], text(string(idx), :black, 9, :bold))
    end
end

function plot_solution(reward_map::Matrix{Float64},
                       start::Tuple{Int, Int},
                       configured_waypoints::Vector{Tuple{Int, Int}},
                       visited_waypoints::Vector{Tuple{Int, Int}},
                       hist::Vector{Tuple{Int, Int}},
                       obstacles::Vector{Tuple{Int, Int}},
                       reward_map_id::Int,
                       planner_name::String,
                       total_reward::Float64)
    p = heatmap(
        reward_map,
        title = "$(planner_name) Multi-Waypoint Solution (Map $(reward_map_id), Reward = $(round(total_reward, digits = 2)))",
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

mkpath(FIGURE_DIR)

planner_name = "MCTS"
start = (25, 40)
wp = [(20, 30), (30, 30), (25, 20), (35, 25)]

obstacles = combine_obstacles(
    rectangle_obstacles(20:40, 36:38),
    rectangle_obstacles(20:22, 39:55),
)

i = 7

sol = WaypointPath.planner(planner_name, i, start, wp; obstacles = obstacles)
reward_map = WaypointPath.get_reward(i)
visited_waypoints = extract_visited_waypoints(sol.hist, wp)

plot_obj = plot_solution(
    reward_map,
    start,
    wp,
    visited_waypoints,
    sol.hist,
    obstacles,
    i,
    planner_name,
    sol.rtot,
)

output_path = joinpath(FIGURE_DIR, "multiWaypointTest_$(planner_name)_map$(i).png")
savefig(plot_obj, output_path)

println("Saved multi-waypoint plot to $(output_path)")
display(plot_obj)
