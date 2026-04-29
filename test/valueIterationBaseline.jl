include("../src/multiWaypointSolver.jl")

using Plots
using Printf
using SparseArrays

const GAMMA = 0.95
const ER = 1e-8
const INVALID_REWARD = -1.0e10
const VI_ACTIONS = collect(WaypointPath.GenerateMDP.ACTIONS)

rectangle_obstacles(rows, cols) = [(r, c) for r in rows for c in cols]

# Custom single-leg verification case.
# With only one goal waypoint, ordering is trivial: [start, goal].
const SINGLE_GOAL_CASE = (
    name = "single_goal_center_block",
    reward_map = 7,
    start = (25, 40),
    goal = (20, 30),
    obstacles = rectangle_obstacles(25:33, 31:33),
    detour_budget = 10.0,
)

travel_distance(hist::Vector{Tuple{Int, Int}}) = max(length(hist) - 1, 0)

function format_reward(value::Float64)
    return isfinite(value) ? @sprintf("%.2f", value) : string(value)
end

function to_plot_coords(points::AbstractVector{<:Tuple{Int, Int}})
    xs = [point[2] for point in points]
    ys = [point[1] for point in points]
    return xs, ys
end

struct LegVIProblem
    states::Vector{Tuple{Tuple{Int,Int},Int}}
    state_to_idx::Dict{Tuple{Tuple{Int,Int},Int},Int}
    actions::Vector{Symbol}
    next_idx::Dict{Symbol,Vector{Int}}
    Tmats::Dict{Symbol,SparseMatrixCSC{Float64,Int}}
    Rvecs::Dict{Symbol,Vector{Float64}}
    start_state::Tuple{Tuple{Int,Int},Int}
    goal::Tuple{Int,Int}
    max_travel::Int
end

states(m::LegVIProblem) = m.states
actions(m::LegVIProblem) = m.actions
transition_matrices(m::LegVIProblem; sparse=true) = m.Tmats
reward_vectors(m::LegVIProblem) = m.Rvecs

function feasible_state(pos::Tuple{Int,Int},
                        g::Int,
                        goal::Tuple{Int,Int},
                        wf::Matrix{Int},
                        R::Matrix{Float64},
                        max_travel::Int,
                        obstacle_set::Set{Tuple{Int,Int}})
    if pos in obstacle_set
        return false
    end

    if pos == goal
        return true
    end

    d = WaypointPath.solveMCTS.wavefront_distance(wf, R, pos)
    return isfinite(d) && (g + d <= max_travel)
end

function build_leg_problem(start::Tuple{Int,Int},
                           goal::Tuple{Int,Int},
                           R::Matrix{Float64},
                           obstacles::Vector{Tuple{Int,Int}};
                           detour_budget::Real=10.0)
    obstacle_set = Set{Tuple{Int,Int}}(obstacles)
    wf = WaypointPath.solveMCTS.build_wavefront(goal, R, obstacle_set)
    shortest = WaypointPath.solveMCTS.wavefront_distance(wf, R, start)

    @assert isfinite(shortest) "Goal $goal is unreachable from $start"

    max_travel = Int(round(shortest + detour_budget))

    S = Tuple{Tuple{Int,Int},Int}[]
    for g in 0:max_travel
        for r in 1:size(R, 1), c in 1:size(R, 2)
            pos = (r, c)
            if feasible_state(pos, g, goal, wf, R, max_travel, obstacle_set)
                push!(S, (pos, g))
            end
        end
    end

    idx = Dict{Tuple{Tuple{Int,Int},Int},Int}(s => i for (i, s) in enumerate(S))

    next_idx = Dict{Symbol,Vector{Int}}()
    Tmats = Dict{Symbol,SparseMatrixCSC{Float64,Int}}()
    Rvecs = Dict{Symbol,Vector{Float64}}()

    for a in VI_ACTIONS
        I = Int[]
        J = Int[]
        V = Float64[]
        Ra = fill(INVALID_REWARD, length(S))
        nexta = fill(1, length(S))

        for (i, (pos, g)) in enumerate(S)
            if pos == goal
                push!(I, i)
                push!(J, i)
                push!(V, 1.0)
                Ra[i] = 0.0
                nexta[i] = i
                continue
            end

            if g >= max_travel
                push!(I, i)
                push!(J, i)
                push!(V, 1.0)
                Ra[i] = INVALID_REWARD
                nexta[i] = i
                continue
            end

            sp = WaypointPath.solveMCTS.action_target(pos, a)
            gp = g + 1

            if !WaypointPath.GenerateMDP.inBounds(sp, R) ||
               (sp in obstacle_set) ||
               !haskey(idx, (sp, gp))
                push!(I, i)
                push!(J, i)
                push!(V, 1.0)
                Ra[i] = INVALID_REWARD
                nexta[i] = i
                continue
            end

            j = idx[(sp, gp)]
            push!(I, i)
            push!(J, j)
            push!(V, 1.0)
            Ra[i] = R[sp[1], sp[2]]
            nexta[i] = j
        end

        next_idx[a] = nexta
        Tmats[a] = sparse(I, J, V, length(S), length(S))
        Rvecs[a] = Ra
    end

    return LegVIProblem(
        S,
        idx,
        VI_ACTIONS,
        next_idx,
        Tmats,
        Rvecs,
        (start, 0),
        goal,
        max_travel,
    )
end

function value_iteration(m, gamma=GAMMA, er=ER)
    T = transition_matrices(m, sparse=true)
    R = reward_vectors(m)
    A = actions(m)

    V = ones(length(states(m)))
    Vp = zeros(length(states(m)))

    delta = Inf
    while delta > er
        Vp .= V

        a = A[1]
        V .= R[a] .+ gamma .* (T[a] * Vp)
        for a in A[2:end]
            Q = R[a] .+ gamma .* (T[a] * Vp)
            V .= max.(V, Q)
        end
        delta = maximum(abs.(V .- Vp))
    end
    return V
end

function greedy_rollout(m::LegVIProblem, V; gamma=GAMMA)
    R = reward_vectors(m)
    A = actions(m)

    s_idx = m.state_to_idx[m.start_state]
    hist = Tuple{Int,Int}[m.start_state[1]]
    raw = 0.0
    disc = 0.0
    t = 0

    while true
        pos, _ = m.states[s_idx]
        if pos == m.goal
            break
        end

        best_a = nothing
        best_q = -Inf

        for a in A
            q = R[a][s_idx] + gamma * V[m.next_idx[a][s_idx]]
            if q > best_q
                best_q = q
                best_a = a
            end
        end

        if isnothing(best_a)
            return hist, -Inf, -Inf
        end

        r = R[best_a][s_idx]
        if r <= INVALID_REWARD / 10
            return hist, -Inf, -Inf
        end

        sp_idx = m.next_idx[best_a][s_idx]
        sp, _ = m.states[sp_idx]

        raw += r
        disc += gamma^t * r
        push!(hist, sp)
        s_idx = sp_idx
        t += 1

        if t > m.max_travel + 2
            return hist, -Inf, -Inf
        end
    end

    return hist, raw, disc
end

function discounted_path_reward(hist::Vector{Tuple{Int,Int}},
                                R::Matrix{Float64};
                                gamma=GAMMA)
    total = 0.0
    for (k, pos) in enumerate(hist[2:end])
        total += gamma^(k - 1) * R[pos[1], pos[2]]
    end
    return total
end

function raw_path_reward(hist::Vector{Tuple{Int,Int}}, R::Matrix{Float64})
    total = 0.0
    for pos in hist[2:end]
        total += R[pos[1], pos[2]]
    end
    return total
end

function timed_call(f)
    t0 = time_ns()
    result = f()
    runtime_s = (time_ns() - t0) / 1e9
    return result, runtime_s
end

function print_results_table(rows)
    method_header = "Method"
    reward_header = "Discounted Reward"
    runtime_header = "Runtime (s)"

    reward_strings = [format_reward(row.discounted_reward) for row in rows]
    runtime_strings = [@sprintf("%.6f", row.runtime_s) for row in rows]

    method_w = max(length(method_header), maximum(length(row.method) for row in rows))
    reward_w = max(length(reward_header), maximum(length(s) for s in reward_strings))
    runtime_w = max(length(runtime_header), maximum(length(s) for s in runtime_strings))

    println(
        rpad(method_header, method_w), " | ",
        lpad(reward_header, reward_w), " | ",
        lpad(runtime_header, runtime_w),
    )
    println(
        repeat("-", method_w), "-+-",
        repeat("-", reward_w), "-+-",
        repeat("-", runtime_w),
    )

    for (row, reward_s, runtime_s) in zip(rows, reward_strings, runtime_strings)
        println(
            rpad(row.method, method_w), " | ",
            lpad(reward_s, reward_w), " | ",
            lpad(runtime_s, runtime_w),
        )
    end
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

function shortest_path_hist(start::Tuple{Int,Int},
                            goal::Tuple{Int,Int},
                            R::Matrix{Float64},
                            obstacles::Vector{Tuple{Int,Int}})
    obstacle_set = Set{Tuple{Int,Int}}(obstacles)
    wf = WaypointPath.solveMCTS.build_wavefront(goal, R, obstacle_set)

    if !isfinite(WaypointPath.solveMCTS.wavefront_distance(wf, R, start))
        return Tuple{Int,Int}[]
    end

    path = Tuple{Int,Int}[start]
    max_steps = prod(size(R)) + 1

    while path[end] != goal && length(path) <= max_steps
        curr = path[end]
        curr_val = wf[curr[1], curr[2]]

        best_neighbor = nothing
        best_val = curr_val

        for (_, (dr, dc)) in WaypointPath.solveMCTS.ACTION_STEPS
            neighbor = (curr[1] + dr, curr[2] + dc)

            if !WaypointPath.GenerateMDP.inBounds(neighbor, R)
                continue
            end

            val = wf[neighbor[1], neighbor[2]]
            if val <= 1
                continue
            end

            if val < best_val
                best_val = val
                best_neighbor = neighbor
            end
        end

        if isnothing(best_neighbor)
            return Tuple{Int,Int}[]
        end

        push!(path, best_neighbor)
    end

    return path[end] == goal ? path : Tuple{Int,Int}[]
end

function make_base_plot(reward_map::Matrix{Float64}, title_str::String)
    return heatmap(
        reward_map,
        title = title_str,
        xlabel = "Column",
        ylabel = "Row",
        colorbar = true,
        colorbar_title = "Reward",
        color = :plasma,
        clims = (minimum(reward_map), maximum(reward_map)),
        size = (900, 700),
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

function add_obstacles!(p, obstacles)
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
            alpha = 0.9,
        )
    end
end

function add_start_goal!(p, start, goal)
    scatter!(
        p,
        [start[2]],
        [start[1]],
        label = "Start",
        color = :dodgerblue,
        markersize = 10,
        markerstrokecolor = :black,
        markerstrokewidth = 1.2,
    )
    scatter!(
        p,
        [goal[2]],
        [goal[1]],
        label = "Goal",
        color = :gold,
        markersize = 10,
        markerstrokecolor = :black,
        markerstrokewidth = 1.2,
    )
end

function add_hist!(p, hist, label_name, line_color; linestyle=:solid)
    if isempty(hist)
        return
    end

    hist_x, hist_y = to_plot_coords(hist)
    plot!(
        p,
        hist_x,
        hist_y,
        label = label_name,
        color = line_color,
        linewidth = 2.5,
        alpha = 0.9,
        linestyle = linestyle,
    )
    scatter!(
        p,
        hist_x,
        hist_y,
        label = "$(label_name) Cells",
        color = line_color,
        markersize = 3,
        markerstrokewidth = 0,
        alpha = 0.55,
    )
end

function plot_overlay_solution(reward_map,
                               start,
                               goal,
                               vi_hist,
                               mcts_hist,
                               shortest_hist,
                               obstacles,
                               vi_disc,
                               mcts_disc,
                               shortest_disc)
    title_str = string(
        "VI vs MCTS vs Shortest Path\n",
        "VI disc = ", format_reward(vi_disc),
        " | MCTS disc = ", format_reward(mcts_disc),
        " | Shortest disc = ", format_reward(shortest_disc),
    )

    p = make_base_plot(reward_map, title_str)
    add_obstacles!(p, obstacles)
    add_hist!(p, vi_hist, "VI Path", :deepskyblue3)
    add_hist!(p, mcts_hist, "MCTS Path", :red3, linestyle=:dash)
    add_hist!(p, shortest_hist, "Shortest Path", :limegreen, linestyle=:dot)
    add_start_goal!(p, start, goal)
    return p
end

function vi_compare_main()
    case = SINGLE_GOAL_CASE

    R = WaypointPath.GenerateMDP.get_reward_map(case.reward_map)
    start = case.start
    goal = case.goal
    obstacles = case.obstacles
    detour_budget = case.detour_budget

    @assert WaypointPath.GenerateMDP.inBounds(start, R) "Start out of bounds"
    @assert WaypointPath.GenerateMDP.inBounds(goal, R) "Goal out of bounds"
    @assert !(start in Set(obstacles)) "Start is inside obstacle set"
    @assert !(goal in Set(obstacles)) "Goal is inside obstacle set"

    (vi_hist, _, vi_disc), vi_runtime = timed_call() do
        leg_problem = build_leg_problem(start, goal, R, obstacles; detour_budget=detour_budget)
        V = value_iteration(leg_problem, GAMMA, ER)
        greedy_rollout(leg_problem, V; gamma=GAMMA)
    end

    mcts_solution, mcts_runtime = timed_call() do
        redirect_stdout(devnull) do
            WaypointPath.planner(
                "MCTS",
                R,
                start,
                [goal];
                obstacles = obstacles,
                detour_budget=detour_budget,
            )
        end
    end
    mcts_hist = mcts_solution.hist
    mcts_visited_wp = extract_visited_waypoints(mcts_hist, [goal])
    mcts_disc = discounted_path_reward(mcts_hist, R; gamma=GAMMA)

    shortest_hist, shortest_runtime = timed_call() do
        shortest_path_hist(start, goal, R, obstacles)
    end
    shortest_disc = isempty(shortest_hist) ? -Inf : discounted_path_reward(shortest_hist, R; gamma=GAMMA)

    print_results_table([
        (method = "VI", discounted_reward = vi_disc, runtime_s = vi_runtime),
        (method = "MCTS", discounted_reward = mcts_disc, runtime_s = mcts_runtime),
        (method = "Shortest Path", discounted_reward = shortest_disc, runtime_s = shortest_runtime),
    ])

    comparison = plot_overlay_solution(
        R,
        start,
        goal,
        vi_hist,
        mcts_hist,
        shortest_hist,
        obstacles,
        vi_disc,
        mcts_disc,
        shortest_disc,
    )
    display(comparison)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    vi_compare_main()
end
