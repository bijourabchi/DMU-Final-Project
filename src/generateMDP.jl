
# Basic idea is for this to be a module where we can generate MDP's from. There are some options on how to do this
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

function load_reward_grid(scenario_json::AbstractString)
    db, _, _ = processJSONInputs(scenario_json, false)
    return db.reward
end

function get_reward_map(scenario::Integer = 1)
    scenario_json = SCENARIO_JSONS[scenario]
    db, _, _ = processJSONInputs(scenario_json, false)
    return db.reward
end

end
