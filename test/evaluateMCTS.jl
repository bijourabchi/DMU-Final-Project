include("multiWaypointTest.jl")

using Random
using Statistics
using Printf

# Usage:
# 1. Run this file directly to evaluate all multi-waypoint cases.
# 2. CSV summaries are written under results/multiWaypointEvaluation/<case>/ by default.
# 3. Edit the DEFAULT_* constants below for the usual workflow.
# 4. Optional environment overrides:
#    MCTS_EVAL_CASES=baseline,circle,center_block,detour_wall,corridor_sweep
#    MCTS_EVAL_REWARD_MAPS=1:10
#    MCTS_EVAL_DETOUR_BUDGETS=0:5:20
#    MCTS_EVAL_TRIALS=25
#    MCTS_EVAL_BASE_SEED=400
#    MCTS_EVAL_OUTPUT_ROOT=results\multiWaypointEvaluation

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_OUTPUT_ROOT = joinpath(PROJECT_ROOT, "results", "multiWaypointEvaluation")
const DEFAULT_EVAL_CASES = collect(CASE_ORDER)
const DEFAULT_REWARD_MAP_IDS = collect(1:10)
const DEFAULT_DETOUR_BUDGETS = collect(0:5:20)
const DEFAULT_TRIALS_PER_CONFIGURATION = 1
const DEFAULT_BASE_SEED = 400

function resolve_output_root()
    raw = strip(get(ENV, "MCTS_EVAL_OUTPUT_ROOT", ""))
    if isempty(raw)
        return DEFAULT_OUTPUT_ROOT
    end

    return isabspath(raw) ? normpath(raw) : normpath(joinpath(PROJECT_ROOT, raw))
end

const OUTPUT_ROOT = resolve_output_root()

function parse_int_list(raw::AbstractString)
    values = Int[]

    for token in split(raw, ",")
        token = strip(token)
        isempty(token) && continue

        parts = split(token, ":")
        if length(parts) == 1
            push!(values, parse(Int, token))
        elseif length(parts) == 2
            start = parse(Int, parts[1])
            stop = parse(Int, parts[2])
            step = start <= stop ? 1 : -1
            append!(values, collect(start:step:stop))
        elseif length(parts) == 3
            start = parse(Int, parts[1])
            step = parse(Int, parts[2])
            stop = parse(Int, parts[3])
            append!(values, collect(start:step:stop))
        else
            throw(ArgumentError("Unsupported integer list token: $(token)"))
        end
    end

    return unique(values)
end

function parse_symbol_list_env(name::String, default_values::Vector{Symbol})
    raw = strip(get(ENV, name, ""))
    if isempty(raw)
        return default_values
    end

    return [Symbol(strip(token)) for token in split(raw, ",") if !isempty(strip(token))]
end

function parse_int_list_env(name::String, default_values::Vector{Int})
    raw = strip(get(ENV, name, ""))
    return isempty(raw) ? default_values : parse_int_list(raw)
end

const EVAL_CASES = parse_symbol_list_env("MCTS_EVAL_CASES", DEFAULT_EVAL_CASES)
const REWARD_MAP_IDS = parse_int_list_env("MCTS_EVAL_REWARD_MAPS", DEFAULT_REWARD_MAP_IDS)
const DETOUR_BUDGETS = parse_int_list_env("MCTS_EVAL_DETOUR_BUDGETS", DEFAULT_DETOUR_BUDGETS)
const TRIALS_PER_CONFIGURATION = parse(Int, get(ENV, "MCTS_EVAL_TRIALS", string(DEFAULT_TRIALS_PER_CONFIGURATION)))
const BASE_SEED = parse(Int, get(ENV, "MCTS_EVAL_BASE_SEED", string(DEFAULT_BASE_SEED)))

function mean_or_nan(values)
    return isempty(values) ? NaN : mean(values)
end

function std_or_nan(values)
    count = length(values)
    if count == 0
        return NaN
    elseif count == 1
        return 0.0
    end

    return std(values)
end

function format_csv_value(value)
    if value isa AbstractFloat
        if isnan(value)
            return "NaN"
        elseif isinf(value)
            return value > 0 ? "Inf" : "-Inf"
        end

        return @sprintf("%.6f", value)
    end

    return string(value)
end

function escape_csv(value)
    text = format_csv_value(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end

    return text
end

function write_csv(rows::Vector{<:NamedTuple}, output_path::String)
    if isempty(rows)
        return
    end

    headers = collect(propertynames(first(rows)))
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(io, join(string.(headers), ","))

        for row in rows
            values = [escape_csv(getproperty(row, header)) for header in headers]
            println(io, join(values, ","))
        end
    end
end

function trial_seed(case_index::Int, reward_map_id::Int, detour_budget::Int, trial_idx::Int)
    return BASE_SEED + 1_000_000 * case_index + 10_000 * reward_map_id + 100 * detour_budget + trial_idx
end

function reward_per_distance(total_reward::Float64, total_distance::Int)
    if total_distance <= 0 || !isfinite(total_reward)
        return NaN
    end

    return total_reward / total_distance
end

function prepare_case(case_name::Symbol, case_config, reward_map_id::Int)
    reward_map = WaypointPath.GenerateMDP.get_reward_map(reward_map_id)
    start, configured_waypoints = resolve_route(case_config.route)
    obstacles = resolve_obstacles(case_config.obstacles)
    validate_start(start, reward_map, obstacles, case_name)

    filtered_waypoints = WaypointPath.filter_wp(configured_waypoints, reward_map; obs = obstacles)
    if isempty(filtered_waypoints)
        throw(ArgumentError("Case $(case_name) has no valid waypoints on reward map $(reward_map_id)."))
    end

    return (
        reward_map = reward_map,
        start = start,
        configured_waypoints = configured_waypoints,
        filtered_waypoints = filtered_waypoints,
        obstacles = obstacles,
    )
end

function run_trial(case_data, detour_budget::Int, seed::Int)
    Random.seed!(seed)

    start_ns = time_ns()
    solution = redirect_stdout(devnull) do
        WaypointPath.planner(
            PLANNER_NAME,
            case_data.reward_map,
            case_data.start,
            case_data.configured_waypoints;
            obstacles = case_data.obstacles,
            detour_budget = detour_budget,
        )
    end
    runtime_seconds = (time_ns() - start_ns) / 1e9

    visited_waypoints = extract_visited_waypoints(solution.hist, case_data.filtered_waypoints)
    total_distance = travel_distance(solution.hist)
    successful = isfinite(solution.rtot) && length(visited_waypoints) == length(case_data.filtered_waypoints)
    completion_rate = length(visited_waypoints) / length(case_data.filtered_waypoints)

    return (
        success = successful,
        total_reward = solution.rtot,
        total_distance = total_distance,
        reward_per_distance = reward_per_distance(solution.rtot, total_distance),
        visited_waypoints = length(visited_waypoints),
        completion_rate = completion_rate,
        runtime_seconds = runtime_seconds,
    )
end

function summarize_trials(case_name::Symbol,
                          reward_map_id::Int,
                          detour_budget::Int,
                          waypoint_count::Int,
                          trials::Vector{<:NamedTuple})
    successful_trials = [trial for trial in trials if trial.success]

    reward_ratios_successful = [trial.reward_per_distance for trial in successful_trials if isfinite(trial.reward_per_distance)]
    rewards_successful = [trial.total_reward for trial in successful_trials if isfinite(trial.total_reward)]
    distance_all = [trial.total_distance for trial in trials]
    distance_successful = [trial.total_distance for trial in successful_trials]
    visited_all = [trial.visited_waypoints for trial in trials]
    completion_all = [trial.completion_rate for trial in trials]
    runtimes = [trial.runtime_seconds for trial in trials]

    return (
        case_name = string(case_name),
        reward_map = reward_map_id,
        requested_detour_budget = detour_budget,
        trials = length(trials),
        waypoint_count = waypoint_count,
        successful_trials = length(successful_trials),
        failure_trials = length(trials) - length(successful_trials),
        success_rate = length(successful_trials) / length(trials),
        reward_per_distance_mean_successful = mean_or_nan(reward_ratios_successful),
        reward_per_distance_std_successful = std_or_nan(reward_ratios_successful),
        total_reward_mean_successful = mean_or_nan(rewards_successful),
        total_reward_std_successful = std_or_nan(rewards_successful),
        total_distance_mean_all = mean_or_nan(distance_all),
        total_distance_std_all = std_or_nan(distance_all),
        total_distance_mean_successful = mean_or_nan(distance_successful),
        total_distance_std_successful = std_or_nan(distance_successful),
        visited_waypoints_mean_all = mean_or_nan(visited_all),
        visited_waypoints_std_all = std_or_nan(visited_all),
        completion_rate_mean_all = mean_or_nan(completion_all),
        completion_rate_std_all = std_or_nan(completion_all),
        runtime_seconds_mean = mean_or_nan(runtimes),
        runtime_seconds_std = std_or_nan(runtimes),
    )
end

function evaluate_case(case_name::Symbol, case_config, case_index::Int)
    output_dir = joinpath(OUTPUT_ROOT, sanitize_name(case_name))
    mkpath(output_dir)

    prepared_cases = Dict{Int, NamedTuple}()
    for reward_map_id in REWARD_MAP_IDS
        prepared_cases[reward_map_id] = prepare_case(case_name, case_config, reward_map_id)
    end

    combined_rows = NamedTuple[]

    println()
    println("Evaluating case $(case_name)")
    println("  Reward maps: $(REWARD_MAP_IDS)")
    println("  Detour budgets: $(DETOUR_BUDGETS)")
    println("  Trials per configuration: $(TRIALS_PER_CONFIGURATION)")

    for detour_budget in DETOUR_BUDGETS
        budget_rows = NamedTuple[]
        budget_path = joinpath(output_dir, @sprintf("detour_budget_%02d.csv", detour_budget))

        println("  Detour budget $(detour_budget)")

        for reward_map_id in REWARD_MAP_IDS
            case_data = prepared_cases[reward_map_id]
            trial_rows = NamedTuple[]

            for trial_idx in 1:TRIALS_PER_CONFIGURATION
                seed = trial_seed(case_index, reward_map_id, detour_budget, trial_idx)
                push!(trial_rows, run_trial(case_data, detour_budget, seed))
            end

            summary_row = summarize_trials(
                case_name,
                reward_map_id,
                detour_budget,
                length(case_data.filtered_waypoints),
                trial_rows,
            )

            push!(budget_rows, summary_row)
            push!(combined_rows, summary_row)

            println(
                @sprintf(
                    "    Map %2d -> success %2d/%2d, reward/distance %.4f, reward %.4f, distance %.2f",
                    reward_map_id,
                    summary_row.successful_trials,
                    summary_row.trials,
                    summary_row.reward_per_distance_mean_successful,
                    summary_row.total_reward_mean_successful,
                    summary_row.total_distance_mean_all,
                )
            )
        end

        write_csv(budget_rows, budget_path)
    end

    write_csv(combined_rows, joinpath(output_dir, "summary_all_detour_budgets.csv"))
end

function evaluate_all_cases()
    mkpath(OUTPUT_ROOT)

    for (case_index, case_name) in enumerate(EVAL_CASES)
        if !haskey(TEST_CASES, case_name)
            throw(ArgumentError("Unknown test case $(case_name). Available cases: $(collect(keys(TEST_CASES)))"))
        end

        evaluate_case(case_name, TEST_CASES[case_name], case_index)
    end
end

function main()
    evaluate_all_cases()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
elseif isinteractive()
    println("Loaded test/evaluateMCTS.jl.")
    println("Run evaluate_all_cases() to execute the evaluation sweep with the current DEFAULT_* settings.")
end

main()