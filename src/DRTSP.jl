module DiscountedWaypointOrdering

using LinearAlgebra

# ============================================================
# DATA STRUCTURES
# ============================================================

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


# ============================================================
# BASIC UTILITIES
# ============================================================

"""
Rescale distances so that the discount factor becomes 1/2.
"""
function rescale_distances(dist::Matrix{Float64}, gamma::Float64)
    scale = log2(1 / gamma)
    return dist .* scale
end


"""
Compute shortest path distances from start node s to every node.


"""
function shortest_from_start(dist::Matrix{Float64}, s::Int)
    return copy(dist[s, :])
end


"""
Compute scaled prizes

where d_v is shortest distance from start to v in the rescaled metric.
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
Discretize scaled prizes for prize-target enumeration.

Δ controls resolution:
    larger Δ  -> fewer K values, coarser search
    smaller Δ -> more K values, finer search
"""
function discretize_scaled_prizes(scaled_prize::Vector{Float64}, Δ::Float64)
    q = floor.(Int, scaled_prize ./ Δ)
    return q
end


"""
Total travel cost of an ordered path.
"""
function path_cost(path::Vector{Int}, dist::Matrix{Float64})
    if length(path) <= 1
        return 0.0
    end

    total = 0.0
    for k in 1:length(path)-1
        total += dist[path[k], path[k+1]]
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

    # Reward at the start node if desired
    first = path[1]
    seen[first] = true
    reward += prize[first] * gamma^(0.0)

    for k in 2:length(path)
        u = path[k-1]
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
Excess of an s->t path:
    excess(P) = length(P) - d(s,t)

Here:
    path length = total length of the ordered node sequence
    d(s,t)      = shortest path from start to endpoint t
"""
function path_excess(path::Vector{Int}, dist::Matrix{Float64}, s::Int)
    t = path[end]
    return path_cost(path, dist) - dist[s, t]
end


# ============================================================
# MIN-EXCESS SUBROUTINE INTERFACE
# ============================================================

"""
Abstract interface for a min-excess solver.
"""
abstract type AbstractMinExcessSolver end


"""
A placeholder greedy solver.

Idea:
    grow a path from s toward t by inserting intermediate waypoints
    that improve scaled-prize-per-added-cost until target K is reached,
    then append t if needed.
"""
struct GreedyMinExcessSolver <: AbstractMinExcessSolver
end

"""
TO DO:
    Implement apporoximate min-excess solver from paper. 
    Obstacle avoidance in solving processing.
"""
struct DPMinExcessSolver <: AbstractMinExcessSolver
end

"""
Pseudo-polynomial approximation of minimum excess solver
"""
function solve_min_excess(::DPMinExcessSolver,
                            dist::Matrix{Float64},
                            s::Int,
                            t::Int,
                            scaled_prize::Vector{Float64},
                            K::Int,
                            q::Vector{Int})

    n = length(q)

    mask = Vector{Int}()

    child = Dict{Tuple{Vector{Int},Int}, Float64}() # mapping to store minimum cost to start at s, visit nodes whose index stored in mask, and end at V::Int
    parent = Dict{Tuple{Vector{Int},Int}, Union{Nothing,Tuple{Vector{Int},Int}}}() # mapping to previous state for path reconstruction

    start_mask = [s] # Only visited the start

    child[(start_mask,s)] = 0.0
    parent[(start_mask,s)] = nothing

    subset_prize = zeros(Int,2^n)

    for mask in 1:2^n
        subset_prize[mask] = sum(q[mask])
    end

    # Enumerate through all the possible paths, skipping the ones that revist nodes and save the best ones
    for mask in 1:2^n

        if !(s in mask)
            continue
        end

        for u in 1:n
            if !haskey(child, (mask,u))
                continue
            end

            current_cost = child[(mask,u)]

            for v = 1:n
                if v in mask
                    continue
                end

                new_mask = copy(mask)
                push!(new_mask, v)

                new_cost = current_cost + dist[u,v]

                # only keep new state if it's better
                if !haskey(child,(new_mask,v)) || new_cost < child[(new_mask,v)]
                    child[(new_mask,v)] = new_cost
                    parent[(new_mask,v)] = (mask,u)
                end
            end
        end

    end

    best_mask = nothing
    best_cost = Inf

    for mask in 1:2^n

        if !(t in mask)
            continue
        end

        if subset_prize[mask] < K
            continue
        end

        if haskey(child,(mask,t))
            cost = child[(mask,t)]

            if cost < best_cost
                best_cost = cost
                best_mask = mask
            end
        end
    end

    if isnothing(best_mask) # cannot find feasible solution
        return nothing
    end

    path = []
    state = (best_mask,t)

    while !isnothing(state)
        (mask,v) = state
        push!(path,v)
        state = parent[(mask,v)]
    end

    reverse(path)


    return path

end


"""
Returns:
    path::Vector{Int}
or
    nothing if unable to reach the target prize.

"""
function solve_min_excess(::GreedyMinExcessSolver,
                          dist::Matrix{Float64},
                          s::Int,
                          t::Int,
                          scaled_prize::Vector{Float64},
                          K::Int,
                          q::Vector{Int})

    n = length(scaled_prize)

    path = [s]
    visited = falses(n)
    visited[s] = true

    current = s
    collected = q[s]

    # Candidate intermediate nodes exclude start and endpoint initially
    while collected < K
        best_node = 0
        best_score = -Inf

        for v in 1:n
            if visited[v] || v == t
                continue
            end

            added_cost = dist[current, v]
            if added_cost <= 0
                continue
            end

            # Simple benefit/cost heuristic
            score = q[v] / added_cost

            if score > best_score
                best_score = score
                best_node = v
            end
        end

        if best_node == 0
            break
        end

        push!(path, best_node)
        visited[best_node] = true
        collected += q[best_node]
        current = best_node
    end

    # Always try to end at t
    if !visited[t]
        push!(path, t)
        visited[t] = true
        collected += q[t]
    end

    if collected < K
        return nothing
    end

    return path
end


# ============================================================
# OUTER DISCOUNTED-REWARD-TSP DRIVER
# ============================================================

"""
Main high-level planner.

Arguments:
    instance         : DRTSP instance
    Δ                : discretization step for scaled prizes
    solver           : min-excess subroutine

Returns:
    best DRTSPSolution found

"""
function discounted_waypoint_order(instance::DRTSPInstance;
                                   Δ::Float64 = 1e-2,
                                   solver::AbstractMinExcessSolver = GreedyMinExcessSolver())

    n = length(instance.prize)
    s = instance.start

    # Step 1: rescale so the discount becomes 1/2
    dist_rescaled = rescale_distances(instance.dist, instance.gamma)

    # Step 2: shortest distances from start in rescaled metric
    d_from_start = shortest_from_start(dist_rescaled, s)

    # Step 3: scaled prizes
    scaled_prize = compute_scaled_prizes(instance.prize, d_from_start)

    # Step 4: discretize scaled prizes
    q = discretize_scaled_prizes(scaled_prize, Δ)
    maxK = sum(q)

    # Initialize 
    best_path = [s]
    best_reward = instance.prize[s]
    best_scaled = scaled_prize[s]
    best_cost = 0.0

    # below is polynomial time approximation
    # guess endpoint t
    for t in 1:n
        if t == s
            continue
        end

        # guess target scaled prize K
        for K in 1:maxK
            candidate = solve_min_excess(solver, dist_rescaled, s, t, scaled_prize, K, q)

            if candidate === nothing
                continue
            end

            # Score using TRUE discounted reward in the original metric
            reward = true_discounted_reward(candidate, instance.prize, instance.dist, instance.gamma)
            scpr   = collected_scaled_prize(candidate, scaled_prize)
            cost   = path_cost(candidate, instance.dist)

            if reward > best_reward
                best_reward = reward
                best_path = candidate
                best_scaled = scpr
                best_cost = cost
            end
        end
    end

    return DRTSPSolution(best_path, best_reward, best_scaled, best_cost)
end


# ============================================================
# HELPER FOR YOUR USE CASE
# ============================================================

"""
Construct a waypoint-ordering instance.

Inputs:
    waypoint_costs : pairwise travel costs among start + waypoints
    waypoint_prize : node rewards for start + waypoints
    gamma          : discount factor

This assumes you already solved or estimated the low-level travel cost
between each pair of waypoints.
"""
function build_waypoint_instance(waypoint_costs::Matrix{Float64},
                                 waypoint_prize::Vector{Float64},
                                 gamma::Float64;
                                 start::Int = 1)

    @assert size(waypoint_costs, 1) == size(waypoint_costs, 2)
    @assert size(waypoint_costs, 1) == length(waypoint_prize)
    @assert 0.0 < gamma < 1.0

    return DRTSPInstance(waypoint_costs, waypoint_prize, start, gamma)
end

function euclidean_distance(s::Tuple{Int,Int}, w::Tuple{Int,Int})
    return sqrt((s[1] - w[1])^2 + (s[2] - w[2])^2)
end

function inBounds(s,R)
    row_dim = size(R,1)
    col_dim = size(R,2)
    r,c = s

    return r > 0 && r <= row_dim && c > 0 && c <= col_dim

end

function filter_wp(wp::Vector{Tuple{Int,Int}},R::Matrix{Float64})
    filtered_wp = Vector{Tuple{Int,Int}}()
    for w in wp
        if inBounds(w,R)
            push!(filtered_wp, w)
        end

    end
    return filtered_wp
end

function distances(wp::Vector{Tuple{Int,Int}})
    n_wp = size(wp,1)
    
    dist = zeros(n_wp,n_wp)

    for i in 1:n_wp
        for j in 1:n_wp
            dist[i,j] = euclidean_distance(wp[i], wp[j])
        end
    end
    return dist
end

function prize(wp::Vector{Tuple{Int,Int}},R::Matrix{Float64})
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
# ============================================================
# EXAMPLE USAGE
# ============================================================

function order_wp(wp::Vector{Tuple{Int,Int}}, R::Matrix{Float64})
    # wp should be ordered as follows [start_node, wp]
    # R is reward matrix to get waypoint prize
    
    wp = filter_wp(wp,R)
    
    dist = distances(wp)
    prizes = prize(wp,R)
    gamma = 0.9

    instance = build_waypoint_instance(dist, prizes, gamma)

    sol = discounted_waypoint_order(instance; Δ=0.1)

    println("Best path: ", sol.path)
    println("True discounted reward: ", sol.reward)
    println("Scaled prize collected: ", sol.scaled_prize)
    println("Travel cost: ", sol.travel_cost)

    return sol
end

function demo()
    # Example:
    # node 1 = start
    # nodes 2:5 = waypoints
    dist = [
        0.0  2.0  5.0  6.0  4.0;
        2.0  0.0  9.0  5.0  3.0;
        5.0  9.0  0.0  9.0  2.0;
        6.0  5.0  9.0  0.0  8.0;
        4.0  3.0  2.0  8.0  0.0
    ]

    prize = [0.0, 10.0, 8.0, 12.0, 6.0]
    gamma = 0.90

    instance = build_waypoint_instance(dist, prize, gamma)

    sol = discounted_waypoint_order(instance; Δ=0.1)

    println("Best path: ", sol.path)
    println("True discounted reward: ", sol.reward)
    println("Scaled prize collected: ", sol.scaled_prize)
    println("Travel cost: ", sol.travel_cost)

    return sol
end

end # module
