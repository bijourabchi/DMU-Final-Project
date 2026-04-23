## All functions required to running MCTS
using Plots

module solveMCTS
    include("generateMDP.jl")

    const visited_bonus = -100
    const obstacle_bonus = -1e10
    const ACTION_STEPS = (
        (:left, (0, -1)),
        (:right, (0, 1)),
        (:up, (1, 0)),
        (:down, (-1, 0)),
        (:upright, (1, 1)),
        (:upleft, (1, -1)),
        (:downright, (-1, 1)),
        (:downleft, (-1, -1)),
    )

    struct SearchState
        pos::Tuple{Int,Int}
        travel::Float64
    end

    state_position(s::Tuple{Int,Int}) = s
    state_position(s::SearchState) = s.pos

    function step_cost(s::Tuple{Int,Int}, sp::Tuple{Int,Int})
        return s == sp ? 0.0 : 1.0
    end

    mutable struct MCTS
        ## Tracking MCTS properties
        goal
        visited
        obstacle_set
        wavefront
        shortest_distance
        detour_budget
        N # (S,A) Counter
        Q
        t # (S A Sp) Counter
        R # Reward matrix
        A # Action Space
        T # Transition fxn
        discount
    end

    function euclidean_distance(s::Tuple{Int,Int}, w::Tuple{Int,Int})
        return sqrt((s[1] - w[1])^2 + (s[2] - w[2])^2)
    end

    function build_wavefront(goal::Tuple{Int,Int},
                             R::Matrix{Float64},
                             obstacle_set::Set{Tuple{Int,Int}})
        wavefront = zeros(Int, size(R))

        for obstacle in obstacle_set
            if GenerateMDP.inBounds(obstacle, R)
                wavefront[obstacle[1], obstacle[2]] = 1
            end
        end

        if !GenerateMDP.inBounds(goal, R) || goal in obstacle_set
            return wavefront
        end

        wavefront[goal[1], goal[2]] = 2
        queue = Tuple{Int,Int}[goal]
        head = 1

        while head <= length(queue)
            cell = queue[head]
            head += 1
            base_distance = wavefront[cell[1], cell[2]]

            for (_, (dr, dc)) in ACTION_STEPS
                neighbor = (cell[1] + dr, cell[2] + dc)

                if !GenerateMDP.inBounds(neighbor, R)
                    continue
                end

                if wavefront[neighbor[1], neighbor[2]] != 0
                    continue
                end

                wavefront[neighbor[1], neighbor[2]] = base_distance + 1
                push!(queue, neighbor)
            end
        end

        return wavefront
    end

    function wavefront_distance(wavefront::Matrix{Int},
                                R::Matrix{Float64},
                                s::Tuple{Int,Int})
        if !GenerateMDP.inBounds(s, R)
            return Inf
        end

        distance = wavefront[s[1], s[2]]
        if distance <= 1
            return Inf
        end

        return max(distance - 2, 0)
    end

    function wavefront_distance(m::MCTS, s)
        return wavefront_distance(m.wavefront, m.R, state_position(s))
    end

    function advance_state(m::MCTS, s::SearchState, a)
        sp = m.T(s.pos, a, m.R)
        return SearchState(sp, s.travel + step_cost(s.pos, sp))
    end

    leg_budget(m::MCTS) = m.shortest_distance + m.detour_budget

    function is_feasible_state(m::MCTS, s::SearchState)
        h = wavefront_distance(m, s)
        return isfinite(h) && (s.travel + h <= leg_budget(m))
    end

    function action_target(s::Tuple{Int,Int}, action)
        for (candidate, (dr, dc)) in ACTION_STEPS
            if candidate == action
                return (s[1] + dr, s[2] + dc)
            end
        end

        error("Unknown action: $action")
    end

    function feasible_actions(m::MCTS, s::SearchState)
        if state_position(s) == m.goal
            return Symbol[]
        end

        pos = state_position(s)
        actions = Symbol[]

        for (action, _) in ACTION_STEPS
            neighbor = action_target(pos, action)
            if !is_valid_neighbor(m, neighbor)
                continue
            end

            sp = advance_state(m, s, action)
            if is_feasible_state(m, sp)
                push!(actions, action)
            end
        end

        return actions
    end

    function rollout(m::MCTS, s0, policy, max_steps = 100)
        rtot = 0.0
        t = 0
        s = s0
        hist = [state_position(s)]

        while t < max_steps && state_position(s) != m.goal
            actions = feasible_actions(m, s)
            if isempty(actions)
                return (hist, -Inf)
            end

            a = policy(m, s, actions)
            if isnothing(a)
                return (hist, -Inf)
            end

            sp = advance_state(m, s, a)
            sp_pos = state_position(sp)
            r = m.R[sp_pos[1], sp_pos[2]]

            rtot += m.discount^t * r
            t += 1
            s = sp
            push!(hist, sp_pos)
        end

        return (hist, rtot)
    end

    function is_valid_neighbor(m::MCTS, s::Tuple{Int,Int})
        return GenerateMDP.inBounds(s, m.R) && !(s in m.obstacle_set)
    end

    function fallback_goal_action(m::MCTS, s, actions::Vector{Symbol})
        pos = state_position(s)
        best_action = nothing
        best_distance = Inf

        for action in actions
            sp = action_target(pos, action)
            distance = euclidean_distance(sp, m.goal)
            if distance < best_distance
                best_distance = distance
                best_action = action
            end
        end

        return best_action
    end

    function wp_heuristic(m::MCTS, s, actions::Vector{Symbol})
        pos = state_position(s)

        if pos == m.goal
            return nothing
        end

        best_action = nothing
        best_distance = Inf

        for action in actions
            sp = advance_state(m, s, action)
            distance = wavefront_distance(m, sp)
            if distance < best_distance
                best_distance = distance
                best_action = action
            end
        end

        if !isnothing(best_action) && isfinite(best_distance)
            return best_action
        end

        return fallback_goal_action(m, s, actions)
    end

    function heuristic(m::MCTS, s, actions::Vector{Symbol})
        return isempty(actions) ? nothing : rand(actions)
    end

    function state_initialized(m::MCTS, s::SearchState)
        return any(haskey(m.N, (s, a)) for a in m.A)
    end

    function simulate!(m::MCTS, s, d = 100)
        if state_position(s) == m.goal
            return 0.0
        end

        actions = feasible_actions(m, s)
        if isempty(actions)
            return -Inf
        end

        if d <= 0
            return rollout(m, s, wp_heuristic)[2]
        end

        gamma = m.discount

        if !state_initialized(m, s)
            for a in actions
                m.N[(s, a)] = 0
                m.Q[(s, a)] = 0.0
            end

            return rollout(m, s, wp_heuristic)[2]
        end

        a = explore(m, s, actions)
        if isnothing(a)
            return -Inf
        end

        sp = advance_state(m, s, a)
        sp_pos = state_position(sp)

        r = m.R[sp_pos[1], sp_pos[2]]

        if sp_pos in m.visited
            r = visited_bonus
        end
        if sp_pos in m.obstacle_set
            r = obstacle_bonus
        end

        Q = r + gamma * simulate!(m, sp, d - 1)

        m.N[(s, a)] += 1
        m.Q[(s, a)] += (Q - m.Q[(s, a)]) / m.N[(s, a)]
        tval = get(m.t, (s, a, sp), 0)
        m.t[(s, a, sp)] = tval + 1

        return Q
    end

    function explore(m::MCTS, s, actions::Vector{Symbol})
        if isempty(actions)
            return nothing
        end

        N = m.N
        Q = m.Q
        c = 7

        bonus = (Nsa, Ns) -> Nsa == 0 ? Inf : sqrt(log(Ns) / Nsa)
        Ns = sum(N[(s, a)] for a in actions)

        return argmax(a -> Q[(s, a)] + c * bonus(N[(s, a)], Ns), actions)
    end

    function select_action(m, s)
        actions = feasible_actions(m, s)
        if isempty(actions)
            return nothing
        end

        for _ in 1:1000
            simulate!(m, s) # 1000 iterations to choose each action, d = 100 by default
        end

        return argmax(a -> m.Q[(s, a)], actions)
    end

    """
    MCTS planner: will navigate from start s0 to single waypoint wp
    Main thing to play around with: 
    detour_budget::Float64 = 10 -> almost like a freedom knob giving the planner more time to explore the larger it is.
    """
    function evaluate(s0::Tuple{Int,Int},
                      wp::Tuple{Int,Int},
                      obstacles::Vector{Tuple{Int,Int}},
                      R::Matrix{Float64};
                      max_steps::Int = 1000, detour_budget::Float64 = 10.0)

        rtot = 0.0
        t = 0
        s = SearchState(s0, 0.0)
        hist = [state_position(s)]
        Tr = GenerateMDP.T

        S = typeof(s)
        Atype = typeof(:left)

        # These are appropriate containers for the state-action tree statistics.
        n = Dict{Tuple{S, Atype}, Int}()
        q = Dict{Tuple{S, Atype}, Float64}()
        tt = Dict{Tuple{S, Atype, S}, Int}()

        visited = [state_position(s)]
        A = [:left, :right, :up, :down, :upright, :upleft, :downright, :downleft]
        obstacle_set = Set{Tuple{Int,Int}}(obstacles)
        wavefront = build_wavefront(wp, R, obstacle_set)
        shortest_distance = wavefront_distance(wavefront, R, s0)

        if !isfinite(shortest_distance)
            return (hist, -Inf)
        end

        print(typeof(detour_budget))
        m = MCTS(wp,
                 visited,
                 obstacle_set,
                 wavefront,
                 shortest_distance,
                 detour_budget,
                 n,
                 q,
                 tt,
                 R,
                 A,
                 Tr,
                 0.95)

        while state_position(s) != wp && t < max_steps
            a = select_action(m, s)
            if isnothing(a)
                break
            end

            sp = advance_state(m, s, a)
            sp_pos = state_position(sp)
            r = R[sp_pos[1], sp_pos[2]]
            rtot += r
            t += 1
            s = sp
            push!(hist, sp_pos)
            push!(visited, sp_pos)

            if mod(t, 20) == 0 || t == 1
                println("s = $(sp_pos), g = $(s.travel)")
            end
        end

        if state_position(s) != wp # Did not reach waypoint
            rtot = -Inf
        end

        return (hist, rtot)
    end
end
