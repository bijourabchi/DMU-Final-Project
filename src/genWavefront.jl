module genWavefront

using Plots

function gen_wave(goal::Tuple{Int,Int},waveFront::Matrix{Int}; obs = [])
    gx,gy = goal
    waveFront[gx,gy] = 2
    [waveFront[obx,oby] = 1 for (obx,oby) in obs]
    obs_set = Set(obs)
    visited = Set([goal])
    q = [goal]
    actions = [(0,1), (0,-1), (1,0), (-1,0),(1,1), (-1,1), (1,-1), (-1,-1)]

    while !isempty(q)
        
        cell = popfirst!(q)
        cx,cy = cell


        for (dx,dy) in actions
            neighbor = (cx+dx,cy+dy)

            if !inBounds(neighbor, waveFront) || (neighbor in visited) || neighbor in obs_set
                continue
            end

            push!(visited,neighbor)
            push!(q,neighbor)

            if !(neighbor in obs)            
                waveFront[neighbor[1],neighbor[2]] = waveFront[cx,cy] + 1
            end
            
        end
    end
    waveFront[gx,gy] = 2
    return waveFront
end

function inBounds(cell::Tuple{Int,Int},wavefront::Matrix{Int})
    n_row = size(wavefront,1)
    n_col = size(wavefront,2)
    r,c = cell
    return r > 0 && r <= n_row && c > 0 && c <= n_col
end

"""
Return length of path from start to goal. Will navigate around obstacles. If no valid path can be made, Inf will be returned

dist::Matrix{Int}
"""
function distance(start::Tuple{Int,Int},goal::Tuple{Int,Int},wf::Matrix{Int})
    
    path = [start]
    count = 0

    while path[end] != goal && count < 500
        curr = path[end]
        action = get_action(curr,wf)
        next = curr .+ action
        push!(path,next)

        count += 1
    end

    if path[end] != goal
        return Inf
    end
    return length(path)
end

function get_path(start::Tuple{Int,Int},goal::Tuple{Int,Int},wf::Matrix{Int})
        
    path = [start]
    count = 0

    while path[end] != goal && count < 500
        curr = path[end]
        action = get_action(curr,wf)
        next = curr .+ action
        push!(path,next)
        count += 1
        
    end

    if path[end] != goal
        return Inf
    end
    return path
end
"""
Return action from cell. Action is lowest wv value that isn't an obstacle
"""
function get_action(cell::Tuple{Int,Int}, wf::Matrix{Int})
    actions = [(0,1), (0,-1), (1,0), (-1,0),(1,1), (-1,1), (1,-1), (-1,-1)]
    cx,cy = cell
    neighbors = [(cx+dx,cy+dy) for (dx,dy) in actions]

    wave_vals = []

    for neighbor in neighbors
        nx,ny = neighbor
        wave_val = wf[nx,ny]
        if wave_val != 1 # Obstacle
            push!(wave_vals,wave_val)
        else
            push!(wave_vals,Inf)
        end
    end

    min_idx = argmin(wave_vals)
    action = (neighbors[min_idx][1] - cell[1], neighbors[min_idx][2] - cell[2])
    
    return action
    
end
function visualize_wf(waveFront::Matrix{Int}; path = nothing)
    p = heatmap(
    waveFront,
    title="Wavefront",
    xlabel="X Coordinate",
    ylabel="Y Coordinate",
    colorbar=true,
    colorbar_title="WF value",
    color=:plasma,
    size=(900, 700),
    margin=5Plots.mm,
    xlim=(0.5, size(waveFront, 2) + 0.5),
    ylim=(0.5, size(waveFront, 1) + 0.5),
    xticks=(1:10:size(waveFront, 2), string.(1:10:size(waveFront, 2))),
    yticks=(1:10:size(waveFront, 1), string.(1:10:size(waveFront, 1))),
    legend=true,
    framestyle=:box,
    dpi=150
    )
    if !isnothing(path)
        
        # Extract x and y coordinates from state history
        hist_x = [s[1] for s in path]
        hist_y = [s[2] for s in path]

        # Overlay state history path
        plot!(p, hist_y, hist_x, label="State History", color=:red, linewidth=2, alpha=0.7)
    end

    savefig(p,"wavefront.png")
end
end

