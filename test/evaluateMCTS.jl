using Plots
using Random

include("../src/MCTS.jl")


# Create solid clumps of obstacles in a 66 x 103 world
function generate_obstacle_clumps(rows::Int, cols::Int, num_clumps::Int=8, clump_radius::Int=4)
    obstacles = Set()
    
    # Generate random centers for clumps
    for _ in 1:num_clumps
        center_row = rand(clump_radius+1:rows-clump_radius)
        center_col = rand(clump_radius+1:cols-clump_radius)
        
        # Add solid obstacles within circular radius of center
        for r in 1:rows
            for c in 1:cols
                distance = sqrt((r - center_row)^2 + (c - center_col)^2)
                if distance <= clump_radius
                    push!(obstacles, (r, c))
                end
            end
        end
    end
    
    return collect(obstacles)
end

function sample_init_state(n_rows, n_cols, wp, Obstacles)

    s_x = rand(1:n_rows)
    s_y = rand(1:n_cols)
    s = (s_x,s_y)

    while s in Obstacles || s == wp 
        s_x = rand(1:n_rows)
        s_y = rand(1:n_cols)
        s = (s_x,s_y)
    end

    return s
end

# Create figures directory if it doesn't exist
mkpath("figures")

for i = 1:10

    Obstacles = generate_obstacle_clumps(66, 103)

    R = GenerateMDP.get_reward_map(i)

    n_rows = size(R,1)
    n_cols = size(R,2)

    wp_x = rand(1:n_rows)
    wp_y = rand(1:n_cols)
    wp = [(wp_x,wp_y)]

    s0 = sample_init_state(n_rows, n_cols, wp, Obstacles)

    hist, rtot = evaluate(s0,wp, Obstacles)

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
    ox = [o[1] for o in Obstacles]
    oy = [o[2] for o in Obstacles]
    scatter!(p, oy, ox, label="Obstacles", color=:black, markersize=4, alpha=0.6)
    
    # Save plot
    savefig(p, "figures/MCTS_$i.png")

end