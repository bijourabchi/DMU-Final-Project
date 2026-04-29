include("valueIterationBaseline.jl")
include("multiWaypointTest.jl")

using Plots
using Printf

function combine_obstacles(groups...)
    obstacles = Tuple{Int, Int}[]
    for group in groups
        append!(obstacles, group)
    end
    return unique(obstacles)
end

const COMPARISON_CASES = ACTIVE_CASES
const LARGE_SCALE_FIGURE_ROOT = joinpath(PROJECT_ROOT, "figures", "largeScaleComparison")

function append_segment!(full_hist::Vector{Tuple{Int,Int}}, seg_hist::Vector{Tuple{Int,Int}})
    if isempty(seg_hist)
        return
    end

    if isempty(full_hist)
        append!(full_hist, seg_hist)
    elseif full_hist[end] == seg_hist[1]
        append!(full_hist, seg_hist[2:end])
    else
        append!(full_hist, seg_hist)
    end
end

function annotate_waypoints!(plot_obj, waypoints::Vector{Tuple{Int, Int}})
    for (idx, point) in enumerate(waypoints)
        annotate!(plot_obj, point[2], point[1], text(string(idx), :black, 9))
    end
end

function add_waypoints!(p, ordered_wp::Vector{Tuple{Int, Int}})
    if isempty(ordered_wp)
        return
    end

    wp_x, wp_y = to_plot_coords(ordered_wp)
    scatter!(
        p,
        wp_x,
        wp_y,
        label = "Ordered Waypoints",
        color = :white,
        markersize = 8,
        markerstrokecolor = :black,
        markerstrokewidth = 1.2,
    )
    annotate_waypoints!(p, ordered_wp)
end

function plot_large_scale_overlay(reward_map,
                                  case_label,
                                  start,
                                  ordered_wp,
                                  vi_hist,
                                  mcts_hist,
                                  shortest_hist,
                                  obstacles,
                                  vi_reward,
                                  mcts_reward,
                                  shortest_reward)
    title_str = string(
        case_label, "\n",
        "VI total reward = ", format_reward(vi_reward),
        " | MCTS total reward = ", format_reward(mcts_reward),
        " | Shortest total reward = ", format_reward(shortest_reward),
    )

    p = make_base_plot(reward_map, title_str)
    add_obstacles!(p, obstacles)
    add_hist!(p, vi_hist, "VI Path", :deepskyblue3)
    add_hist!(p, mcts_hist, "MCTS Path", :red3, linestyle = :dash)
    add_hist!(p, shortest_hist, "Shortest Path", :limegreen, linestyle = :dot)
    add_waypoints!(p, ordered_wp)

    if !isempty(ordered_wp)
        add_start_goal!(p, start, ordered_wp[end])
    else
        add_start_goal!(p, start, start)
    end

    return p
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

function run_vi_route(start::Tuple{Int,Int},
                      ordered_wp::Vector{Tuple{Int,Int}},
                      reward_map::Matrix{Float64},
                      obstacles::Vector{Tuple{Int,Int}},
                      detour_budget::Real)
    full_hist = Tuple{Int,Int}[start]
    current = start

    _, runtime_s = timed_call() do
        for goal in ordered_wp
            leg_problem = build_leg_problem(current, goal, reward_map, obstacles; detour_budget = detour_budget)
            V = value_iteration(leg_problem, GAMMA, ER)
            seg_hist, _, seg_disc = greedy_rollout(leg_problem, V; gamma = GAMMA)
            append_segment!(full_hist, seg_hist)

            if !isfinite(seg_disc) || isempty(seg_hist) || seg_hist[end] != goal
                break
            end

            current = goal
        end
    end

    total_reward = raw_path_reward(full_hist, reward_map)
    return full_hist, total_reward, runtime_s
end

function run_mcts_route(start::Tuple{Int,Int},
                        configured_waypoints::Vector{Tuple{Int,Int}},
                        reward_map::Matrix{Float64},
                        obstacles::Vector{Tuple{Int,Int}},
                        detour_budget::Real)
    solution, runtime_s = timed_call() do
        redirect_stdout(devnull) do
            WaypointPath.planner(
                "MCTS",
                reward_map,
                start,
                configured_waypoints;
                obstacles = obstacles,
                detour_budget = detour_budget,
            )
        end
    end

    ordered_wp = solution.ordered_wp
    visited_wp = extract_visited_waypoints(solution.hist, configured_waypoints)
    total_reward = solution.rtot

    return solution.hist, ordered_wp, total_reward, runtime_s
end

function print_case_results_table(rows)
    case_header = "Case"
    method_header = "Method"
    reward_header = "Total Reward"
    runtime_header = "Runtime (s)"

    case_strings = [row.case_label for row in rows]
    reward_strings = [format_reward(row.discounted_reward) for row in rows]
    runtime_strings = [@sprintf("%.6f", row.runtime_s) for row in rows]

    case_w = max(length(case_header), maximum(length(s) for s in case_strings))
    method_w = max(length(method_header), maximum(length(row.method) for row in rows))
    reward_w = max(length(reward_header), maximum(length(s) for s in reward_strings))
    runtime_w = max(length(runtime_header), maximum(length(s) for s in runtime_strings))

    println(
        rpad(case_header, case_w), " | ",
        rpad(method_header, method_w), " | ",
        lpad(reward_header, reward_w), " | ",
        lpad(runtime_header, runtime_w),
    )
    println(
        repeat("-", case_w), "-+-",
        repeat("-", method_w), "-+-",
        repeat("-", reward_w), "-+-",
        repeat("-", runtime_w),
    )

    for (row, reward_s, runtime_s) in zip(rows, reward_strings, runtime_strings)
        println(
            rpad(row.case_label, case_w), " | ",
            rpad(row.method, method_w), " | ",
            lpad(reward_s, reward_w), " | ",
            lpad(runtime_s, runtime_w),
        )
    end
end

function run_shortest_route(start::Tuple{Int,Int},
                            ordered_wp::Vector{Tuple{Int,Int}},
                            reward_map::Matrix{Float64},
                            obstacles::Vector{Tuple{Int,Int}})
    full_hist = Tuple{Int,Int}[start]
    current = start

    _, runtime_s = timed_call() do
        for goal in ordered_wp
            seg_hist = shortest_path_hist(current, goal, reward_map, obstacles)
            append_segment!(full_hist, seg_hist)

            if isempty(seg_hist) || seg_hist[end] != goal
                break
            end

            current = goal
        end
    end

    total_reward = raw_path_reward(full_hist, reward_map)
    return full_hist, total_reward, runtime_s
end

function save_comparison_plot(plot_obj, case_name::Symbol)
    mkpath(LARGE_SCALE_FIGURE_ROOT)
    filename = "$(sanitize_name(case_name))_comparison.png"
    savefig(plot_obj, joinpath(LARGE_SCALE_FIGURE_ROOT, filename))
end

function large_scale_compare_main()
    rows = NamedTuple[]

    for case_name in COMPARISON_CASES
        case = TEST_CASES[case_name]
        reward_map = WaypointPath.GenerateMDP.get_reward_map(case.reward_map)
        start, configured_waypoints = resolve_route(case.route)
        obstacles = resolve_obstacles(case.obstacles)
        validate_start(start, reward_map, obstacles, case_name)

        mcts_hist, ordered_wp, mcts_reward, mcts_runtime = run_mcts_route(
            start,
            configured_waypoints,
            reward_map,
            obstacles,
            case.detour_budget,
        )

        vi_hist, vi_reward, vi_runtime = run_vi_route(
            start,
            ordered_wp,
            reward_map,
            obstacles,
            case.detour_budget,
        )
        shortest_hist, shortest_reward, shortest_runtime = run_shortest_route(
            start,
            ordered_wp,
            reward_map,
            obstacles,
        )

        case_label = pretty_name(case_name)
        append!(rows, [
            (case_label = case_label, method = "VI", discounted_reward = vi_reward, runtime_s = vi_runtime),
            (case_label = case_label, method = "MCTS", discounted_reward = mcts_reward, runtime_s = mcts_runtime),
            (case_label = case_label, method = "Shortest Path", discounted_reward = shortest_reward, runtime_s = shortest_runtime),
        ])

        comparison = plot_large_scale_overlay(
            reward_map,
            case_label,
            start,
            ordered_wp,
            vi_hist,
            mcts_hist,
            shortest_hist,
            obstacles,
            vi_reward,
            mcts_reward,
            shortest_reward,
        )
        save_comparison_plot(comparison, case_name)
        display(comparison)
    end

    print_case_results_table(rows)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    large_scale_compare_main()
end
