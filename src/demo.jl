
filepath = "../../RINAO.jl/"

grid_image = filepath * "Operator_data/Brainard/BrainardTestArea.png"
trajReferenceFile = filepath *  "Operator_data/Brainard/Brainard_ref_trajs.jl"

S1 = filepath * "Operator_data/Brainard/Brainard_S1.jl"
S2 = filepath * "Operator_data/Brainard/Brainard_S2.jl"
S3 = filepath * "Operator_data/Brainard/Brainard_S3.jl"
S4 = filepath * "Operator_data/Brainard/Brainard_S4.jl"
S5 = filepath * "Operator_data/Brainard/Brainard_S5.jl"
S6 = filepath * "Operator_data/Brainard/Brainard_S6.jl"
S7 = filepath * "Operator_data/Brainard/Brainard_S7.jl"
S8 = filepath * "Operator_data/Brainard/Brainard_S8.jl"
S9 = filepath * "Operator_data/Brainard/Brainard_S9.jl"
S10 = filepath * "Operator_data/Brainard/Brainard_S10.jl"
op_files = [S1,S2,S3,S4,S5,S6,S7,S8,S9,S10]
# op_files = [S6]
toPlot = true
γ = 0.95
include(S5)
inputJSON = "../RINAO.jl/" * inputJSON
db, vars, inputs = processJSONInputs(inputJSON, toPlot)