module DiscountedWaypointOrdering

include("genWavefront.jl")
using LinearAlgebra

#=

Notes before push:

    Bring back and revamp solve_min_excess routines defined previously.
    Should have two abstract methods: One as in the previous implementation to solve min excess problem (find good algo for this)
        another abstract method that implements algorithm 3 from the Farbstein paper


=#


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
## TODO: Delete if not needed
function shortest_from_start(dist::Matrix{Float64}, s::Int)
    return copy(dist[s, :])
end


"""
Compute scaled prizes

where d_v is shortest distance from start to v in the rescaled metric.
"""
## TODO: Delete if not needed
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
## TODO: Delete if not needed
function discretize_scaled_prizes(scaled_prize::Vector{Float64}, Δ::Float64)
    q = floor.(Int, scaled_prize ./ Δ)
    return q
end


"""
Total travel cost of an ordered path.
"""
## TODO: Delete if not needed
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
## TODO: Delete if not needed
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
## TODO: Delete if not needed
function path_excess(path::Vector{Int}, dist::Matrix{Float64}, s::Int)
    t = path[end]
    return path_cost(path, dist) - dist[s, t]
end

"""
Append new node onto existing path
"""
function append_path!(P::Vector{Int}, Q::Vector{Int})
    
    if isempty(Q)
        return
    end

    if isempty(P)
        append!(P,Q)
        return
    end

    if P[end] == Q[1]
        append!(P, Q[2:end])
    else
        append!(P, Q)
    end
end

"""
black box Min Excess Path Approximation Subroutine
"""
function solve!(tp_prev,ti,dist,π_i_apx,ki)
end
# ============================================================
# Discounted-TSP SOLVER
# ============================================================

abstract type AbstractMinExcessSolver end

struct MinExcessPath <: AbstractMinExcessSolver 
end

function solve_min_excess(::MinExcessPath, guess_tuple, dist, V, s::Int,l::Int, eps::Vector{Float64}, delta::Float64)
    P = Int[]
    push!(P,s)

    eps_apx = []

    for i = 1:l
        ti,tpi,ui,ki = guess_tuple[i]
        tp_prev = (i == 1) ? s : guess_tuple[i-1][2]

        π_i_apx = ## TODO Implement apx prize calculation

        Pi, excess_i = solve!(tp_prev,ti,dist,π_i_apx,ki) ## TODO Implement

        if Pi == nothing
            return nothing
        end

        append_path!(P,Q_i)

        if isempty(Q_i) || Q_i[end] != tp_i
            push!(P,tp_i)
        end

        push!(eps_apx,excess_i)
    end

    return P
end

struct RewardCollectingCycles <: AbstractMinExcessSolver
end

"""
Main high level planner from Farbstein paper.

Returns best solution found
"""
function discounted_waypoint_order(instance::DRTSPInstance; solver::AbstractMinExcessSolver = MinExcessPath())


    # Step 1: rescale so discount becomes 1/2
    dist_hat = rescale_distances(instance.dist, instance.gamma)

    # Step 2: remove zero-prize nodes except possibly the start
    V = [v for v in 1:length(instance.prize) if v == instance.start || instance.prize[v] > 0]
    n = length(V)
    s = instance.start

    best_path = [s]
    best_value = true_discounted_reward(best_path, instance.prize, instance.dist, instance.gamma)

    # store guesses as tuples: (t_i, tp_i, u_i, k_i)
    guess_tuple = Tuple{Int,Int,Int,Float64}[]
    
    recurse!(1, guess_tuple, instance, dist_hat, V, n, s, solver, best_path, best_value)

    return best_path
end

"""

"""
function recurse!(i::Int,
                  guess_tuple,
                  instance::DRTSPInstance,
                  dist_hat,
                  V,
                  n::Int,
                  s::Int,
                  solver::AbstractMinExcessSolver,
                  best_path,
                  best_value)

    # Base case: full guess tuple has been formed
    if i > l
        ## TODO: Fill in algo 2 and 3 solver
    end

    π_i, π_i_ub = compute_stage_prizes(i, guess_tuple, instance.prize, dist_hat, V, s, instance.eps)

    for t_i in V
        for tp_i in V
            for u_i in V
                K_list = generate_k_candidates(u_i, π_i, π_i_ub, δ, n)

                for k_i in K_list
                    push!(guess_tuple, (t_i, tp_i, u_i, k_i))
                    recurse!(i + 1, guess_tuple, instance, dist_hat, V, n, s, solver, best_path, best_value)
                    pop!(guess_tuple)
                end
            end
        end
    end
end

function compute_stage_prizes(i::Int,
                              guess_tuple,
                              prize::Vector{Float64},
                              dist_hat,
                              V,
                              s::Int,
                              eps::Vector{Float64})

    π_i = Dict{Int,Float64}()
    π_i_ub = Dict{Int,Float64}()

    if i == 1
        for v in V
            val = prize[v] * 0.5^(dist_hat[s, v])
            π_i[v] = val
            π_i_ub[v] = val
        end
        return π_i, π_i_ub
    end

    base_dist_exp = 0.0
    for j in 1:(i-1)
        t_j, tp_j, _, _ = guess_tuple[j]
        tp_prev = (j == 1) ? s : guess_tuple[j-1][2]
        base_dist_exp += dist_hat[tp_prev, t_j] + dist_hat[t_j, tp_j]
    end

    base_eps_exp = sum(eps[1:(i-1)])
    tp_prev = guess_tuple[i-1][2]

    for v in V
        ub = prize[v] * 0.5^(base_dist_exp + dist_hat[tp_prev, v])
        π_i_ub[v] = ub
        π_i[v] = ub * 0.5^(base_eps_exp)
    end

    return π_i, π_i_ub
end

function generate_k_candidates(u_i::Int,
                               π_i,
                               π_i_ub,
                               delta::Float64,
                               n::Int)

    lower = π_i[u_i]
    upper = n * π_i_ub[u_i]

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

# ============================================================
# Helper Functions
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
# TODO: Update this function
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

function distances(wp::Vector{Tuple{Int,Int}},R::Matrix{Float64}; obs = [])
    n_wp = size(wp,1)
    
    dist = zeros(n_wp,n_wp)

    for i in 1:n_wp
        wf = zeros(Int,size(R))
        wf = genWavefront.gen_wave(wp[i],wf; obs = obs)
        for j in 1:n_wp

            if wp[i] == wp[j]
                dist[i,j] = 0
                continue
            end

            dist[i,j] = genWavefront.distance(wp[j], wp[i], wf)

            if dist[i,j] == Inf
                println("No path found between wpi = $wp[i] and wpj = $wp[j]")
            end
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

function order_wp(wp::Vector{Tuple{Int,Int}}, R::Matrix{Float64};
                  solver::AbstractMinExcessSolver = GreedyMinExcessSolver(),
                  visit_all::Bool = true, obs = [])
    
    # wp should be ordered as follows [start_node, wp]
    # R is reward matrix to get waypoint prize
    
    wp = filter_wp(wp,R)
    
    dist = distances(wp,R;obs = obs)

    println(dist)
    prizes = prize(wp,R)
    gamma = 0.9

    instance = build_waypoint_instance(dist, prizes, gamma)

    sol = discounted_waypoint_order(instance; Δ=0.1, solver = solver, visit_all = visit_all)

    println("Best path: ", sol.path)
    println("True discounted reward: ", sol.reward)
    println("Scaled prize collected: ", sol.scaled_prize)
    println("Travel cost: ", sol.travel_cost)

    return sol
end


end # module
