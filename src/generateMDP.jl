
# Basic idea is for this to be a module where we can generate MDP's from. There are some options on how to do this
# Using QuickMDP, which is a little bit more abstract and doesn't make a whole lot of SCENARIO_FILES
# The better option imo is to have this just return A, T,R, gamma. This is my preferred approach because it more easily allows us to set waypoints and handle the state space in an easier way.
# I had codex write up the QuickMDP version that can apparently track waypoints and which have been visited but I don't understant any of it. 

module GenerateMDP

using QuickPOMDPs: QuickMDP
using POMDPTools: Deterministic, Uniform
using POMDPs
using RINAO

export MDPs, get_mdp, get_mdp_by_name, build_mdp, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10

const BASE_RINAO_PATH = normpath(joinpath(@__DIR__, "..", "..", "RINAO.jl"))
const SCENARIO_JSONS = [
    joinpath(BASE_RINAO_PATH, "Operator_data", "Brainard", "Brainard_S$(i).json")
    for i in 1:10
]
const ACTIONS = [:left, :right, :up, :down, :upright, :upleft, :downright, :downleft]
const INITIAL_STATES = [
    (21, 21), (21, 60), (21, 80),
    (41, 21), (41, 60), (41, 80),
    (61, 21), (61, 60), (61, 80),
]

function sample_init_state()
    return rand(INITIAL_STATES)
end
function next_state(s, a)
    x, y = s
    if a == :left
        y -= 1
    elseif a == :right
        y += 1
    elseif a == :up
        x += 1
    elseif a == :down
        x -= 1
    elseif a == :upright
        y += 1
        x += 1
    elseif a == :upleft
        y -= 1
        x += 1
    elseif a == :downright
        y += 1
        x -= 1
    elseif a == :downleft
        y -= 1
        x -= 1
    end
    return (x, y)
end

function inBounds(s,R)
    row_dim = size(R,1)
    col_dim = size(R,2)
    r,c = s

    return r > 0 && r <= row_dim && c > 0 && c <= col_dim

end

function T(s,a,R) 
    # Transition into next state if able to. If next state is out of bounds, it will just return current state
    sp = next_state(s,a)

    return inBounds(sp,R) ? sp : s
end
function set_visited(mask::NTuple{K,Bool}, idx::Int) where {K}
    if idx < 1 || idx > K
        return mask
    end
    return ntuple(i -> i == idx ? true : mask[i], Val(K))
end

function build_mdp(reward_grid)
    rows, cols = size(reward_grid)
    states = [(r, c) for r in 1:rows for c in 1:cols]

    return QuickMDP(
        states = states,
        actions = ACTIONS,
        initialstate = Uniform(INITIAL_STATES),
        discount = 0.95,
        transition = (s, a) -> Deterministic(next_state(s, a)),
        reward = function (s, a)
            x, y = next_state(s, a)
            return reward_grid[x, y]
        end,
    )
end

function build_mdp(reward_grid::AbstractMatrix, waypoints::NTuple{K,<:Tuple{Int,Int}}) where {K}
    rows, cols = size(reward_grid)
    waypoint_index = Dict{Tuple{Int,Int}, Int}(wp => i for (i, wp) in enumerate(waypoints))
    visited_masks = K == 0 ? [()] : [NTuple{K,Bool}(m) for m in Iterators.product(fill((false, true), K)...)]
    states = [(r, c, v) for r in 1:rows for c in 1:cols for v in visited_masks]
    initial_visited = ntuple(_ -> false, Val(K))

    return QuickMDP(
        states = states,
        actions = ACTIONS,
        initialstate = Uniform([(s[1], s[2], initial_visited) for s in INITIAL_STATES]),
        discount = 0.95,
        transition = function (s, a)
            x, y, visited = s
            xp, yp = next_state((x, y), a)
            idx = get(waypoint_index, (xp, yp), 0)
            visitedp = set_visited(visited, idx)
            return Deterministic((xp, yp, visitedp))
        end,
        reward = function (s, a)
            x, y, _ = s
            xp, yp = next_state((x, y), a)
            return reward_grid[xp, yp]
        end,
    )
end

function build_mdp(reward_grid::AbstractMatrix, waypoints::AbstractVector{<:Tuple{Int,Int}})
    return build_mdp(reward_grid, Tuple(waypoints))
end

function build_mdp(scenario_idx::Integer, waypoints::NTuple{K,<:Tuple{Int,Int}}) where {K}
    if scenario_idx < 1 || scenario_idx > length(SCENARIO_JSONS)
        throw(ArgumentError("No scenario for index $(scenario_idx). Valid indices are 1:$(length(SCENARIO_JSONS))."))
    end
    reward_grid = load_reward_grid(SCENARIO_JSONS[scenario_idx])
    return build_mdp(reward_grid, waypoints)
end

function build_mdp(scenario_idx::Integer, waypoints::AbstractVector{<:Tuple{Int,Int}})
    return build_mdp(scenario_idx, Tuple(waypoints))
end

function load_reward_grid(scenario_json::AbstractString)
    db, _, _ = processJSONInputs(scenario_json, false)
    return db.reward
end

function get_reward_map(scenario::Integer = 1)
    scenario_json = SCENARIO_JSONS[scenario]
    db, _, _ = processJSONInputs(scenario_json, false)
    return db.reward
end

const MDPs = Dict{Symbol, Any}()
for i in eachindex(SCENARIO_JSONS)
    key = Symbol("S$(i)")
    reward_grid = load_reward_grid(SCENARIO_JSONS[i])
    MDPs[key] = build_mdp(reward_grid)
end

for i in 1:10
    key = Symbol("S$(i)")
    @eval const $key = MDPs[$(QuoteNode(key))]
end

function get_mdp(i::Integer)
    key = Symbol("S$(i)")
    if !haskey(MDPs, key)
        throw(ArgumentError("No MDP for index $(i). Valid indices are 1:$(length(MDPs))."))
    end
    return MDPs[key]
end

function get_mdp_by_name(name::Symbol)
    if !haskey(MDPs, name)
        throw(ArgumentError("No MDP with name $(name). Valid names: $(collect(keys(MDPs)))."))
    end
    return MDPs[name]
end

end
