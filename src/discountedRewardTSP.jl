module waypointOrdering

include("genWavefront.jl")

"""
High-level waypoint instance for discounted-reward ordering.

Nodes are:
    1 = start node
    2:n = waypoint nodes

Fields:
    dist[i,j]   : metric or approximately metric travel cost from i to j
                  (assumed precomputed by your low-level planner)
    prize[i]    : reward for visiting node i
    start       : index of start node (usually 1)
    gamma       : discount factor in (0,1)
"""
struct DRTSPInstance
    dist::Matrix{Float64}
    prize::Vector{Float64}
    start::Int
    gamma::Float64
    l::Int
    eps::Vector{Float64}
    delta::Float64
end

"""
Returned solution from the high-level planner.

path            : ordered node indices visited, e.g. [1, 4, 2, 5]
reward          : true discounted reward of this path
scaled_prize    : sum of scaled prizes collected
travel_cost     : total path length
"""
struct DRTSPSolution
    path::Vector{Int}
    reward::Float64
    scaled_prize::Float64
    travel_cost::Float64
end

abstract type MinExcessSolver end

struct simpleMinExcess <: MinExcessSolver
end

abstract type WaypointOrdering end

struct blackboxMinExcess <: WaypointOrdering
end

"""
Greedy fallback for the min-excess subproblem.

The routine grows a path from `s` toward `t` by repeatedly adding the
best reward-per-cost intermediate node until the target scaled prize `k_i`
is reached, then closes the path at `t`.
"""
function solve_min_excess(::simpleMinExcess, s::Int, t::Int, dist_hat, r_i_apx, k_i::Float64)
    n = size(dist_hat, 1)

    path = [s]
    visited = falses(n)
    visited[s] = true

    curr = s
    collected = get(r_i_apx, s, 0.0)

    while collected < k_i
        best_v = 0
        best_score = -Inf

        for v in keys(r_i_apx)
            if visited[v] || v == t
                continue
            end

            added_cost = dist_hat[curr, v]
            if !isfinite(added_cost) || added_cost <= 0
                continue
            end

            score = r_i_apx[v] / added_cost
            if score > best_score
                best_score = score
                best_v = v
            end
        end

        if best_v == 0
            break
        end

        push!(path, best_v)
        visited[best_v] = true
        collected += get(r_i_apx, best_v, 0.0)
        curr = best_v
    end

    end_cost = dist_hat[curr, t]
    if !isfinite(end_cost)
        return nothing, Inf
    end

    if path[end] != t
        push!(path, t)
        visited[t] = true
        collected += get(r_i_apx, t, 0.0)
    end

    if collected < k_i
        return nothing, Inf
    end

    excess = path_distance(path, dist_hat) - dist_hat[s, t]
    return path, excess
end

function solve!(::blackboxMinExcess,
                guess_tuple,
                instance::DRTSPInstance,
                dist_hat,
                V,
                s::Int;
                solver::MinExcessSolver = simpleMinExcess())

    P = [s]
    eps_apx = Float64[]

    for i in eachindex(guess_tuple)
        t, tp, _, k = guess_tuple[i]
        tp_prev = (i == 1) ? s : guess_tuple[i - 1][2]
        visited_nodes = Set(P)

        r_i_apx = apx_prize(i, guess_tuple, eps_apx, instance.prize, dist_hat, V, s, visited_nodes)
        Q_i, excess_i = solve_min_excess(solver, tp_prev, t, dist_hat, r_i_apx, k)

        if isnothing(Q_i)
            return nothing
        end

        append_path!(P, Q_i)

        if isempty(Q_i) # || Q_i[end] != tp
            push!(P, tp)
        end

        push!(eps_apx, excess_i)
    end

    return P
end

"""
This implementation follows the high-level guess enumeration structure from the
discounted-reward waypoint ordering algorithm and keeps the best feasible path.
"""
function discounted_waypoint_ordering(instance::DRTSPInstance;
                                      solver::MinExcessSolver = simpleMinExcess(),
                                      waypointSolver::WaypointOrdering = blackboxMinExcess())

    @assert size(instance.dist, 1) == size(instance.dist, 2)
    @assert size(instance.dist, 1) == length(instance.prize)
    @assert 0.0 < instance.gamma < 1.0

    dist_hat = rescale_distances(instance.dist, instance.gamma)
    V = [v for v in 1:length(instance.prize) if v != instance.start && instance.prize[v] > 0] # Remove start and keep only non/zero prizes which should be all of them
    
    n = length(V)
    s = instance.start
    stage_limit = min(instance.l, n)

    d_from_start = shortest_from_start(dist_hat, s)
    scaled_prize = compute_scaled_prizes(instance.prize, d_from_start)

    best_path_ref = Ref([s])
    best_value_ref = Ref(true_discounted_reward(best_path_ref[], instance.prize, instance.dist, instance.gamma))

    guess_tuple = Tuple{Int, Int, Int, Float64}[]

    recurse!(waypointSolver, 1, guess_tuple, instance, dist_hat, V, n, s, stage_limit, solver, best_path_ref, best_value_ref)

    best_path = best_path_ref[]
    best_reward = best_value_ref[]
    total_scaled_prize = collected_scaled_prize(best_path, scaled_prize)
    cost = path_distance(best_path, instance.dist)

    return DRTSPSolution(best_path, best_reward, total_scaled_prize, cost)
end

"""
Compute scaled prizes where `d_v` is the shortest distance from the start to `v`
in the rescaled metric.
"""
function compute_scaled_prizes(prize::Vector{Float64}, d_from_start::Vector{Float64})
    n = length(prize)
    scaled = zeros(Float64, n)
    for v in 1:n
        scaled[v] = prize[v] * 2.0^(-d_from_start[v])
    end
    return scaled
end

"""
Sum of scaled prizes collected along a path, counting each node once.
"""
function collected_scaled_prize(path::Vector{Int}, scaled_prize::Vector{Float64})
    seen = falses(length(scaled_prize))
    total = 0.0
    for v in path
        if !seen[v]
            total += scaled_prize[v]
            seen[v] = true
        end
    end
    return total
end

"""
Construct guess tuples recursively and keep the best feasible path found.
"""
function recurse!(waypointSolver::blackboxMinExcess,
                  i::Int,
                  guess_tuple,
                  instance::DRTSPInstance,
                  dist_hat,
                  V,
                  n::Int,
                  s::Int,
                  stage_limit::Int,
                  solver::MinExcessSolver,
                  best_path_ref,
                  best_value_ref)

    prefix_path = isempty(guess_tuple) ? [s] : solve!(waypointSolver, guess_tuple, instance, dist_hat, V, s; solver = solver)

    if isnothing(prefix_path)
        return
    end

    update_best!(prefix_path, instance, best_path_ref, best_value_ref)

    if i > stage_limit
        return
    end

    visited_nodes = Set(prefix_path)
    remaining_V = [v for v in V if !(v in visited_nodes)]
    remaining_n = length(remaining_V)

    if remaining_n == 0
        return
    end

    r_i, r_i_ub = compute_stage_prizes(i, guess_tuple, instance.prize, dist_hat, remaining_V, s, instance.eps)

    if all(v -> v <= 0.0, values(r_i_ub))
        return
    end

    for t_i in remaining_V
        for tp_i in remaining_V
            for u_i in remaining_V
                K_list = generate_k_candidates(u_i, r_i, r_i_ub, instance.delta, remaining_n)

                for k_i in K_list
                    push!(guess_tuple, (t_i, tp_i, u_i, k_i))
                    recurse!(waypointSolver, i + 1, guess_tuple, instance, dist_hat, V, n, s, stage_limit, solver, best_path_ref, best_value_ref)
                    pop!(guess_tuple)
                end
            end
        end
    end
end

"""
Compute approximate stage prizes for a specific guess tuple `(t_i, tp_i, u_i, k_i)`.
"""
function compute_stage_prizes(i::Int,
                              guess_tuple,
                              prize::Vector{Float64},
                              dist_hat,
                              V,
                              s::Int,
                              eps::Vector{Float64})

    r_i = Dict{Int, Float64}()
    r_i_ub = Dict{Int, Float64}()
    visited_nodes = guessed_visited_nodes(guess_tuple, i - 1)

    if i == 1
        for v in V
            val = prize[v] * 0.5^(dist_hat[s, v])
            r_i[v] = val
            r_i_ub[v] = val
        end
        return r_i, r_i_ub
    end

    base_dist_exp = 0.0
    for j in 1:(i - 1)
        t_j, tp_j, _, _ = guess_tuple[j]
        tp_prev = (j == 1) ? s : guess_tuple[j - 1][2]
        base_dist_exp += dist_hat[tp_prev, t_j] + dist_hat[t_j, tp_j]
    end

    base_eps_exp = sum(eps[1:(i - 1)])
    tp_prev = guess_tuple[i - 1][2]

    for v in V
        if v in visited_nodes
            r_i_ub[v] = 0.0
            r_i[v] = 0.0
            continue
        end

        ub = prize[v] * 0.5^(base_dist_exp + dist_hat[tp_prev, v])
        r_i_ub[v] = ub
        r_i[v] = ub * 0.5^(base_eps_exp)
    end

    return r_i, r_i_ub
end

"""
Generate the candidate list of `k` values.
"""
function generate_k_candidates(u_i::Int,
                               r_i,
                               r_i_ub,
                               delta::Float64,
                               n::Int)

    lower = r_i[u_i]
    upper = n * r_i_ub[u_i]

    if lower <= 0.0
        return Float64[]
    end

    K_list = Float64[]
    k = lower

    while k < upper
        push!(K_list, k)
        k *= (1.0 + delta)
    end

    push!(K_list, k)
    return K_list
end

"""
Rescale distances so that the discount factor becomes `1/2`.
"""
function rescale_distances(dist::Matrix{Float64}, gamma::Float64)
    scale = log2(1 / gamma)
    return dist .* scale
end

"""
Compute the total distance of a given path.
"""
function path_distance(path::Vector{Int}, dist)
    if length(path) <= 1
        return 0.0
    end

    total = 0.0
    for k in 1:(length(path) - 1)
        total += dist[path[k], path[k + 1]]
    end
    return total
end

"""
True discounted reward of a path using the original gamma and original dist matrix.

Reward at a node is received only on first visit.
"""
function true_discounted_reward(path::Vector{Int},
                                prize::Vector{Float64},
                                dist::Matrix{Float64},
                                gamma::Float64)
    seen = falses(length(prize))
    reward = 0.0
    travel = 0.0

    first = path[1]
    seen[first] = true
    reward += prize[first]

    for k in 2:length(path)
        u = path[k - 1]
        v = path[k]
        travel += dist[u, v]

        if !seen[v]
            reward += prize[v] * gamma^(travel)
            seen[v] = true
        end
    end

    return reward
end

"""
Append a new node sequence onto an existing path while avoiding duplicate joins.
"""
function append_path!(P::Vector{Int}, Q::Vector{Int})
    if isempty(Q)
        return
    end

    if isempty(P)
        append!(P, Q)
        return
    end

    if P[end] == Q[1]
        append!(P, Q[2:end])
    else
        append!(P, Q)
    end
end

function update_best!(candidate_path::Vector{Int},
                      instance::DRTSPInstance,
                      best_path_ref,
                      best_value_ref)
    candidate_value = true_discounted_reward(candidate_path, instance.prize, instance.dist, instance.gamma)

    if candidate_value > best_value_ref[]
        best_path_ref[] = copy(candidate_path)
        best_value_ref[] = candidate_value
    end
end

function guessed_visited_nodes(guess_tuple, num_stages::Int)
    visited_nodes = Set{Int}()

    for j in 1:num_stages
        t_j, tp_j, _, _ = guess_tuple[j]
        push!(visited_nodes, t_j)
        push!(visited_nodes, tp_j)
    end

    return visited_nodes
end

function apx_prize(i::Int,
                   guess_tuple,
                   eps_apx::Vector{Float64},
                   prize::Vector{Float64},
                   dist_hat,
                   V,
                   s::Int,
                   visited_nodes)

    r_i_apx = Dict{Int, Float64}()

    if i == 1
        for v in V
            r_i_apx[v] = prize[v] * 0.5^(dist_hat[s, v])
        end
        return r_i_apx
    end

    base_dist = 0.0

    for j in 1:(i - 1)
        tj, tpj, _, _ = guess_tuple[j]
        tp_prev = (j == 1) ? s : guess_tuple[j - 1][2]
        base_dist += dist_hat[tp_prev, tj] + dist_hat[tj, tpj]
    end

    base_eps = sum(eps_apx)
    tp_prev = guess_tuple[i - 1][2]

    for v in V
        if v in visited_nodes
            r_i_apx[v] = 0.0
            continue
        end

        r_i_apx[v] = prize[v] * 0.5^(base_dist + dist_hat[tp_prev, v]) * 0.5^(base_eps)
    end

    return r_i_apx
end

"""
Compute shortest path distances from the start node `s` to every node.
"""
function shortest_from_start(dist::Matrix{Float64}, s::Int)
    return copy(dist[s, :])
end

function inBounds(s, R)
    row_dim = size(R, 1)
    col_dim = size(R, 2)
    r, c = s

    return 1 <= r <= row_dim && 1 <= c <= col_dim
end

function filter_wp(wp::Vector{Tuple{Int, Int}}, R::Matrix{Float64})
    filtered_wp = Tuple{Int, Int}[]
    for w in wp
        if inBounds(w, R)
            push!(filtered_wp, w)
        end
    end
    return filtered_wp
end

function distances(wp::Vector{Tuple{Int, Int}}, R::Matrix{Float64}; obs = [])
    n_wp = size(wp, 1)
    dist = zeros(n_wp, n_wp)

    for i in 1:n_wp
        wf = zeros(Int, size(R))
        wf = genWavefront.gen_wave(wp[i], wf; obs = obs)
        for j in 1:n_wp
            if wp[i] == wp[j]
                dist[i, j] = 0.0
                continue
            end

            dist[i, j] = genWavefront.distance(wp[j], wp[i], wf)

            if dist[i, j] == Inf
                println("No path found between wpi = $(wp[i]) and wpj = $(wp[j])")
            end
        end
    end

    return dist
end

"""
Collect reward at a waypoint and within a 2-cell radius around it.
"""
function prize(wp::Vector{Tuple{Int, Int}}, R::Matrix{Float64})
    wp_prize = zeros(Float64, length(wp))
    row_dim, col_dim = size(R)

    for (i, (r, c)) in enumerate(wp)
        p = 0.0
        for dr in -2:2
            for dc in -2:2
                rr = r + dr
                cc = c + dc
                if 1 <= rr <= row_dim && 1 <= cc <= col_dim
                    p += R[rr, cc]
                end
            end
        end
        wp_prize[i] = p
    end

    return wp_prize
end

"""
Main interface to solve the waypoint ordering problem.
"""
function order_wp(wp::Vector{Tuple{Int, Int}},
                  R::Matrix{Float64};
                  solver::MinExcessSolver = simpleMinExcess(),
                  waypointSolver::WaypointOrdering = blackboxMinExcess(),
                  obs = [])

    wp = filter_wp(wp, R)
    @assert !isempty(wp) "At least one waypoint must lie inside the reward map."

    dist = distances(wp, R; obs = obs)

    prizes = prize(wp, R)
    gamma = 0.9

    l = 3
    xi = [0.84, 0.62, 0.42]
    eps = -log2.(xi)
    delta = 0.1

    instance = DRTSPInstance(dist, prizes, 1, gamma, l, eps, delta)
    sol = discounted_waypoint_ordering(instance; solver = solver, waypointSolver = waypointSolver)

    println("Best path: ", sol.path)
    println("True discounted reward: ", sol.reward)
    println("Scaled prize collected: ", sol.scaled_prize)
    println("Travel cost: ", sol.travel_cost)

    return sol
end

end

if !(@isdefined DiscountedWaypointOrdering)
    const DiscountedWaypointOrdering = waypointOrdering
end
