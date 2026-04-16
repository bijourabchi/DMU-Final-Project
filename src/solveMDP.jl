include("generateMDP.jl")

reward = GenerateMDP.get_reward_map(1)



heatmap(
    reward,
    title="Reward Map",
    xlabel="X Coordinate",
    ylabel="Y Coordinate",    colorbar=true,    colorbar_title="Reward Value",
    color=:plasma,
    size=(900, 700),
    margin=5Plots.mm,
    xticks=(1:10:size(reward, 2), string.(1:10:size(reward, 2))),
    yticks=(1:10:size(reward, 1), string.(1:10:size(reward, 1))),
    legend=false,
    framestyle=:box,
    dpi=150
)
