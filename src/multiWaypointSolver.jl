module WaypointPath

include("../src/MCTS.jl") # Load in MCTS module
include("../src/genWavefront.jl") # Wavefront module for path length around obstacles
include("../src/discountedRewardTSP.jl") # Waypoint ordering method
include("../src/generateMDP.jl")

## Initialization

const SOLVERS = ["MCTS"] # Add as more are implemented

function get_reward(i::Int)
    @assert i > 0 && i < 11 "No valid reward map, choose map between 1-10"

    R =  GenerateMDP.get_reward_map(i)
    println("Reward Map For Dataset $i Loaded")
    return R
end

"""
Store all relevant parameters for analyzing planner performance. \\
hist::Vector{Tuple{Int,Int}} -> Full state history
rtot::Float64 -> Total accumulated reward
"""
struct plannerSolution
    hist::Vector{Tuple{Int,Int}} 
    rtot::Float64
end

## Point a to b planning methods
"""
waypointGuidance abstract method will house the point a to b planning functions
"""
abstract type waypointGuidance end

"""
MCTS solver
"""
struct MCTS <: waypointGuidance
end

function guide(::MCTS, start::Tuple{Int,Int}, wp::Tuple{Int,Int}, R::Matrix{Float64}; obstacles::Vector{Tuple{Int,Int}} = [])

    return solveMCTS.evaluate(start,wp,obstacles,R)

end

"""
Not explicetly needed for planners other than MCTS

    larger the i -> more freedom we give the planner. detour_budget will be given by an exponential growth 2^i
"""

"""
generate_path

Inputs:
start -> starting cell
wp -> ordered set of waypoints
obstacles -> list of obstacle tuples
R -> Reward Matrix

Outputs:
(hist, rtot)
hist -> List of all visited states
rtot -> total accumulated reward
"""
function generate_path(solver::waypointGuidance,
                       start::Tuple{Int,Int},
                       wp::Vector{Tuple{Int,Int}},
                       R::Matrix{Float64};
                       obstacles::Vector{Tuple{Int,Int}} = [])

    hist = Tuple{Int,Int}[start]
    rtot = 0.0
    current = start

    for w in wp
        sub_hist, sub_rtot = guide(solver, current, w, R; obstacles = obstacles)

        if sub_hist[end] != w
            println("Waypoint $w not reached, moving to next wp")
            current = w = sub_hist[end]
        else
            println("Waypoint $w reached!")
            current = w
        end
        append!(hist, sub_hist[2:end])
        
        rtot += sub_rtot
        
    end

    return (hist, rtot)
end

## Waypoint ordering/processing methods

function filter_wp(wp::Vector{Tuple{Int, Int}}, R::Matrix{Float64}; obs = [])
    
    filtered_wp = Vector{Tuple{Int,Int}}()
    for w in wp
        if GenerateMDP.inBounds(w, R) && !(w in obs)
            push!(filtered_wp, w)
        end

    end
    return filtered_wp

end

function order_wp(wp::Vector{Tuple{Int, Int}}, R::Matrix{Float64}; obs = [])
    println("Filtering Waypoints $wp")
    filtered_wp = filter_wp(wp,R; obs = obs)
    println("Waypoints Filtered, Ordering...")
    return waypointOrdering.order_wp(filtered_wp,R; obs = obs)
end

## Main Output Driver
"""
planner: Main function to run waypoint path planning mode. 
Inputs:
planner::String -> Choices for point a - b planning Choices = {"MCTS"}
start::Tuple{Int,Int} -> Start grid cell
wp::Vector{Tuple{Int,Int}} -> Waypoint grid cells

Optional Inputs:
obstacles::Vector{Tuple{Int,Int}} = [] -> Obstacle grid cells

Outputs:
sol::plannerSolution

"""
function planner(planner::String,r::Int, start::Tuple{Int,Int}, wp::Vector{Tuple{Int,Int}};obstacles::Vector{Tuple{Int,Int}} = [])
    
    @assert planner in SOLVERS "Not a valid solver"

    if planner == "MCTS"
        solver = MCTS()
    end # Fill in as more methods are implemented

    # Load in reward matrix
    R = get_reward(r)

    # Append start onto wp list for ordering method
    wp_with_start = copy(wp)
    pushfirst!(wp_with_start,start)
    sol = order_wp(wp_with_start,R; obs = obstacles)

    # Unpack start
    ordered_path = sol.path
    wp_idx = ordered_path[2:end]
    ordered_wp = Tuple{Int,Int}[]
    for idx in wp_idx
        push!(ordered_wp, wp_with_start[idx])
    end
    (hist, rtot) = generate_path(solver,start,ordered_wp,R; obstacles = obstacles)

    return plannerSolution(hist,rtot)
end

end
