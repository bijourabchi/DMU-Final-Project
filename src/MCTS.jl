## All functions required to running MCTS

module solveMCTS
    include("generateMDP.jl")

    const waypoint_bonus = 1000
    const visited_bonus = -100
    const obstacle_bonus = -1e10
    const potential_scale = 125.0
    const potential_beta = 0.85
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

    mutable struct MCTS
        ## Tracking MCTS properties
        goal
        visited
        obstacle_set
        wavefront
        N # (S,A) Counter
        Q
        t # (S A Sp) Counter
        R # Reward matrix
        A # Action Space
        T # Transition function
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

    function wavefront_distance(m::MCTS, s::Tuple{Int,Int})
        if !GenerateMDP.inBounds(s, m.R)
            return Inf
        end

        distance = m.wavefront[s[1], s[2]]
        return distance > 1 ? distance - 2 : Inf
    end

    function potential(m::MCTS, s::Tuple{Int,Int})
        distance = wavefront_distance(m, s)
        return isfinite(distance) ? potential_scale * potential_beta^distance : 0.0
    end

    function shaping_bonus(m::MCTS, s::Tuple{Int,Int}, sp::Tuple{Int,Int})
        return m.discount * potential(m, sp) - potential(m, s)
    end

    function transition_reward(m::MCTS, s::Tuple{Int,Int}, sp::Tuple{Int,Int})
        r = m.R[sp[1], sp[2]]

        if sp in m.visited
            r = visited_bonus
        end
        if sp == m.goal
            r = waypoint_bonus
        end
        if sp in m.obstacle_set
            r = obstacle_bonus
        end

        return r + shaping_bonus(m, s, sp)
    end

    function rollout(m::MCTS, s0, policy, max_steps = 100)
        rtot = 0.0
        t = 0
        s = s0
        hist = [s]

        while t < max_steps && s != m.goal
            a = policy(m, s)
            sp = GenerateMDP.T(s, a, m.R)
            r = transition_reward(m, s, sp)

            rtot += m.discount^t * r
            t += 1
            s = sp
            push!(hist, s)
        end

        return (hist, rtot)
    end

    function is_valid_neighbor(m::MCTS, s::Tuple{Int,Int})
        return GenerateMDP.inBounds(s, m.R) && !(s in m.obstacle_set)
    end

    function fallback_goal_action(m::MCTS, s::Tuple{Int,Int})
        best_action = nothing
        best_distance = Inf

        for (action, (dr, dc)) in ACTION_STEPS
            sp = (s[1] + dr, s[2] + dc)
            if !is_valid_neighbor(m, sp)
                continue
            end

            distance = euclidean_distance(sp, m.goal)
            if distance < best_distance
                best_distance = distance
                best_action = action
            end
        end

        return isnothing(best_action) ? rand(m.A) : best_action
    end

    function wp_heuristic(m::MCTS, s::Tuple{Int,Int})
        if s == m.goal
            return rand(m.A)
        end

        best_action = nothing
        best_distance = Inf

        for (action, (dr, dc)) in ACTION_STEPS
            sp = (s[1] + dr, s[2] + dc)
            if !is_valid_neighbor(m, sp)
                continue
            end

            distance = wavefront_distance(m, sp)
            if distance < best_distance
                best_distance = distance
                best_action = action
            end
        end

        if !isnothing(best_action) && isfinite(best_distance)
            return best_action
        end

        return fallback_goal_action(m, s)
    end

    function heuristic(m::MCTS, s)
        return rand(m.A) # Random action
    end

    function simulate!(m::MCTS, s, d = 100)
        if d <= 0
            return rollout(m, s, wp_heuristic)[2]
        end

        A = m.A
        discount = m.discount

        if !haskey(m.N, (s, first(A)))
            for a in A
                m.N[(s, a)] = 0
                m.Q[(s, a)] = 0.0
            end

            return rollout(m, s, wp_heuristic)[2]
        end

        a = explore(m, s)
        sp = m.T(s, a, m.R)
        r = transition_reward(m, s, sp)

        Q = r + discount * simulate!(m, sp, d - 1)

        m.N[(s, a)] += 1
        m.Q[(s, a)] += (Q - m.Q[(s, a)]) / m.N[(s, a)]
        tval = get(m.t, (s, a, sp), 0)
        m.t[(s, a, sp)] = tval + 1

        return Q
    end

    function explore(m::MCTS, s)
        N = m.N
        Q = m.Q
        c = 7
        A = m.A

        bonus = (Nsa, Ns) -> Nsa == 0 ? Inf : sqrt(log(Ns) / Nsa)
        Ns = sum(N[(s, a)] for a in A)

        return argmax(a -> begin
            sp = m.T(s, a, m.R)
            d = wavefront_distance(m, sp)
            Q[(s, a)] + c * bonus(N[(s, a)], Ns) - d
        end, A)
    end

    function select_action(m, s, A)
        for _ in 1:1000
            simulate!(m, s) # 1000 iterations to choose each action, d = 100 by default
        end

        return argmax(a -> m.Q[(s, a)], A)
    end

    """
    MCTS planner: will navigate from start s0 to single waypoint wp
    """
    function evaluate(s0::Tuple{Int,Int},
                      wp::Tuple{Int,Int},
                      obstacles::Vector{Tuple{Int,Int}},
                      R::Matrix{Float64};
                      max_steps::Int = 1000)
        rtot = 0.0
        t = 0
        s = s0
        hist = [s]
        Tr = GenerateMDP.T

        S = typeof(s)
        Atype = typeof(:left)

        n = Dict{Tuple{S, Atype}, Int}()
        q = Dict{Tuple{S, Atype}, Float64}()
        tt = Dict{Tuple{S, Atype, S}, Int}()

        visited = Set([s])
        A = [:left, :right, :up, :down, :upright, :upleft, :downright, :downleft]
        obstacle_set = Set{Tuple{Int,Int}}(obstacles)
        wavefront = build_wavefront(wp, R, obstacle_set)

        m = MCTS(wp, visited, obstacle_set, wavefront, n, q, tt, R, A, Tr, 0.95)

        while s != wp && t < max_steps
            println(t)
            a = select_action(m, s, A)
            sp = Tr(s, a, R)
            r = R[sp[1], sp[2]]
            rtot += r
            t += 1
            s = sp
            push!(hist, s)
            push!(visited, s)
        end

        if s != wp
            rtot = -Inf
        end

        return (hist, rtot)
    end
end
