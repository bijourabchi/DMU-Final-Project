## All functions required to running MCTS
using Plots

include("generateMDP.jl")

const waypoint_bonus = 1000
const visited_bonus = -100
const obstacle_bonus = -1e10

mutable struct MCTS
    ## Tracking MCTS properties
    wp
    visited
    obstacles
    N # (S,A) Counter
    Q
    t # (S A Sp) Counter
    R # Reward matrix
    A # Action Space
    T # Tranisition fxn
    discount
end

function euclidean_distance(s::Tuple{Int,Int}, w::Tuple{Int,Int})
    return sqrt((s[1] - w[1])^2 + (s[2] - w[2])^2)
end

function rollout(m::MCTS,s0,policy,max_steps = 100)
    rtot = 0
    t = 0
    s = s0
    hist = [s]
    while t < max_steps
        a = policy(m,s)
        r = m.R[s[1],s[2]]
        sp = GenerateMDP.T(s,a,m.R)
        d = euclidean_distance(sp, m.wp[1])

        rtot += m.discount^t * r
        t += 1
        s = sp
        push!(hist,s)
    end

    return (hist, rtot)
end

function wp_heuristic(m,s)
    # heuristic rollout policy
    x,y = s

    AS = [
        (:down, m.T(s,:down,m.R)),
        (:up, m.T(s,:up,m.R)),
        (:right, m.T(s,:right,m.R)),
        (:left, m.T(s,:left,m.R)),
        (:upright, m.T(s,:left,m.R)),
        (:upleft, m.T(s,:upleft,m.R)),
        (:downright, m.T(s,:downright,m.R)),
        (:downleft, m.T(s,:downleft,m.R))
    ]

    # Find closest goal from CURRENT state
    closest_goal_cell = m.wp[argmin([(x - g[1])^2 + (y - g[2])^2 for g in m.wp])]
    distances_to_goal = [(next_s[1] - closest_goal_cell[1])^2 + (next_s[2] - closest_goal_cell[2])^2 for (a,next_s) in AS]
    return AS[argmin(distances_to_goal)][1]
end

function heuristic(m::MCTS,s)
    return rand(m.A) # Random Action
end

function simulate!(m::MCTS,s, d = 100)

    if d <= 0
        return rollout(m,s,wp_heuristic)[2]
    end

    A = m.A
    γ = m.discount

    if !haskey(m.N, (s,first(A)))
        for a in A
            m.N[(s,a)] = 0 
            m.Q[(s,a)] = 0.0
        end

        return rollout(m,s,wp_heuristic)[2]
    end

    a = explore(m,s)
    sp = m.T(s,a,m.R)

    r = m.R[sp[1],sp[2]]
    alpha = 0
    if r < 0.5
        alpha = 0.5
    end

    if euclidean_distance(s,m.wp[1]) > euclidean_distance(sp,m.wp[1])
        r += alpha + 1/euclidean_distance(sp,m.wp[1])
    end

    if sp in m.visited
        r = visited_bonus
    end
    if sp in m.wp
        r = waypoint_bonus
    end
    if sp in m.obstacles
        r = obstacle_bonus
    end
    
    Q = r + γ * simulate!(m,s,d-1)

    m.N[(s,a)] += 1
    m.Q[(s,a)] += (Q-m.Q[(s,a)])/m.N[(s,a)]
    tval = get(m.t, (s, a, sp), 0)
    m.t[(s,a,sp)] = tval + 1

    return Q

end

function explore(m::MCTS,s)
    N = m.N
    Q = m.Q
    c = 7

    A = m.A

    bonus = (Nsa, Ns) -> Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)
    Ns = sum(N[(s,a)] for a in A)
    d = euclidean_distance(s,m.wp[1])
    return argmax(a -> Q[(s,a)] + c*bonus(N[(s,a)],Ns) - d,A)
end

function select_action(m,s,A)

    for _ in  1:1000
        simulate!(m,s) # 1000 iterations to choose each action, d = 100 by default
    end

    return argmax(a -> m.Q[(s,a)], A)
end

function evaluate(s0,wp,obstacles,max_steps = 100)

    println("Starting MCTS Planning")
    rtot = 0
    t = 0
    s = s0
    hist = [s]
    Tr = GenerateMDP.T
    R = GenerateMDP.get_reward_map(9)
    
    S = typeof(s)
    Atype = typeof(:left)

    # These would be appropriate containers for your Q, N, and t dictionaries:
    n = Dict{Tuple{S, Atype}, Int}()
    q = Dict{Tuple{S, Atype}, Float64}()
    tt = Dict{Tuple{S, Atype, S}, Int}()

    visited = [s]
    A = [:left, :right, :up, :down, :upright, :upleft, :downright, :downleft]

    
    m = MCTS(wp,visited,obstacles, n,q,tt,R,A,Tr,0.95)

    
    while s != wp[1] && t < max_steps
        A = [:left, :right, :up, :down, :upright, :upleft, :downright, :downleft]
        
        a = select_action(m,s, A)
        sp = Tr(s,a,R)
        r = R[sp[1],sp[2]]
        rtot += r
        t += 1
        s = sp
        push!(hist,s)
        push!(visited,s)

        if mod(t,50) == 0
            println("Still Going, s = $s")
        end
    end

    if s != wp[1] # Did not reach waypoint
        rtot = -Inf
    end

    return (hist, rtot)
end

if abspath("src/MCTS.jl") == @__FILE__
    R = GenerateMDP.get_reward_map(2)
    wp = [(50,50)]

    #s0 = GenerateMDP.sample_init_state()
    s0 = (41, 21)

    ox = 30:1:45
    oy = [31,32]
    obstacles = [(x,y) for x in ox for y in oy]


    hist, rtot = evaluate(s0,wp, obstacles)



    # Create heatmap
    p = heatmap(
        R,
        title="Reward Map with State History",
        xlabel="X Coordinate",
        ylabel="Y Coordinate",
        colorbar=true,
        colorbar_title="Reward Value",
        color=:plasma,
        clims=(0, 1),  # Scale colors to focus on 0-1 range
        size=(900, 700),
        margin=5Plots.mm,
        xlim=(0.5, size(R, 2) + 0.5),
        ylim=(0.5, size(R, 1) + 0.5),
        xticks=(1:10:size(R, 2), string.(1:10:size(R, 2))),
        yticks=(1:10:size(R, 1), string.(1:10:size(R, 1))),
        legend=true,
        framestyle=:box,
        dpi=150
    )

    # Extract x and y coordinates from state history
    hist_x = [s[1] for s in hist]
    hist_y = [s[2] for s in hist]

    # Overlay state history path
    plot!(p, hist_y, hist_x, label="State History", color=:red, linewidth=2, alpha=0.7)

    # Overlay state history points
    scatter!(p, hist_y, hist_x, label="Path Points", color=:red, markersize=4, alpha=0.6)

    # Overlay waypoints
    wp_x = [w[1] for w in wp]
    wp_y = [w[2] for w in wp]
    scatter!(p, wp_y, wp_x, label="Waypoints", color=:lime, markersize=8, markerstrokewidth=2, markerstrokecolor=:black)

    # Display starting point
    scatter!(p, [hist_y[1]], [hist_x[1]], label="Start", color=:cyan, markersize=8, markerstrokewidth=2, markerstrokecolor=:black)

    # Overlay obstacles
    ox = [o[1] for o in obstacles]
    oy = [o[2] for o in obstacles]
    scatter!(p, oy, ox, label="Obstacles", color=:black, markersize=4, alpha=0.6)
    p
end