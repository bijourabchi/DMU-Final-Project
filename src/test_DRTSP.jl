include("DRTSP.jl")
include("generateMDP.jl")
include("genWavefront.jl")

R = GenerateMDP.get_reward_map(2)

wp = [(25,40),(20,30),(30,30)]

obs = []
for ox in 25:1:33
    for oy in 31:1:33
        push!(obs,(ox,oy))
    end
end

#solver = DiscountedWaypointOrdering.GreedyMinExcessSolver()
solver = DiscountedWaypointOrdering.DPMinExcessSolver()
sol = DiscountedWaypointOrdering.order_wp(wp,R;solver = solver, obs = obs)

path = sol.path
wp = DiscountedWaypointOrdering.filter_wp(wp,R)
ordered_wp = wp[path]

wavefront = zeros(Int,size(R))

#wavefront = genWavefront.gen_wave((54,20),wavefront;obs = [(50,50)])
#path = genWavefront.get_path((20,30),(54,20),wavefront)
#genWavefront.visualize_wf(wavefront;path = path)


using Plots

p = heatmap(
    R,
    title="Reward Map with Ordered Waypoints",
    xlabel="X Coordinate",
    ylabel="Y Coordinate",
    colorbar=true,
    colorbar_title="Reward Value",
    color=:plasma,
    clims=(0, 1),
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

# Extract coordinates
wp_x = [w[1] for w in ordered_wp]
wp_y = [w[2] for w in ordered_wp]

# Plot path line
plot!(p, wp_y, wp_x,
    label="Path",
    color=:red,
    linewidth=2,
    alpha=0.7
)

# Plot waypoints
scatter!(p, wp_y, wp_x,
    label="Waypoints",
    color=:lime,
    markersize=8,
    markerstrokewidth=2,
    markerstrokecolor=:black
)

o_x = [w[1] for w in obs]
o_y = [w[2] for w in obs]

scatter!(p, o_y, o_x,
    label="obstacles",
    color=:black,
    markersize=8,
    markerstrokewidth=2,
    markerstrokecolor=:black
)

for i in 1:length(wp_x)
    annotate!(
        p,
        wp_y[i], wp_x[i],
        text(string(i), :black, 10, :bold)
    )
end


scatter!(p, [wp_y[1]], [wp_x[1]],
    color=:blue, markersize=10,
    label="Start"
)

scatter!(p, [wp_y[end]], [wp_x[end]],
    color=:yellow, markersize=10,
    label="End"
)
