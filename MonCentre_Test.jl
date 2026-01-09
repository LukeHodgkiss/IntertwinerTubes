#module TestPackages

# Importing Paxkages
using SparseArrayKit: SparseArray, nonzero_values, nonzero_keys, nonzero_pairs
using MAT
using LinearAlgebra
using TensorOperations
using CSV, DataFrames
using StaticArrays: SVector
using Profile
using ProfileView


# Importing Locally
include("SparseAlgebraObjects.jl")
using .SparseAlgebraObjects: TubeAlgebra, random_left_linear_combination_ijk, random_right_linear_combination_ijk, create_left_ijk_basis, create_right_ijk_basis
include("F_symbolTools.jl")
using .FSymbolTools
include("MonoidalCentreTools.jl")
using .MonoidalCentreTools
include("Idempotents.jl")
using .Idempotents: find_idempotents, dim_calc
include("Saving_Stuff.jl")
using .Saving_Stuff

# Read in F-symbol
println("Reading in F symbol data")

###################################
# -  Rep A4 over Rep A4 Rep A4  - #
###################################
#=
vars = matread("Luke_F.mat")
F = SparseArray{ComplexF64}(vars["F"])
quantum_dims = vec(vars["dim"])
size_dict = Dict(:module_label => size(F, 1),
                 :module_label_N => size(F, 1),
                 :module_label_M => size(F, 1),
                 :fusion_label => size(F, 2),
                 :multiplicity_label => size(F)[end],
                 :multiplicity_label_M => size(F)[end],
                 :multiplicity_label_N => size(F)[end])

N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
=#

############################
# -  Doubled Fibonnacci  - #
############################
#=
# F Symbol Fibonnacci
index_file = "/home/lukehodgkiss/Documents/FindingTubesJulia/FibData/Luke_Fib_ind.csv"
value_file = "/home/lukehodgkiss/Documents/FindingTubesJulia/FibData/Luke_Fib_var.csv"

df_index = CSV.read(index_file, DataFrame; header=false)
df_var = CSV.read(value_file, DataFrame; header=false)

F3_DOK = Dict{CartesianIndex{10}, ComplexF64}()

for i in 1:nrow(df_index)
    row_idx = df_index[i, :]
    row_val = df_var[i, :]
    
    key = ntuple(j -> Int(row_idx[j]), 10)
    var = complex(row_val[1], row_val[2])

    F3_DOK[CartesianIndex(key)] = var
end

# Fib shape
F3_shape = (2,4,4,2,2,4,2,2,1,2)
size_dict = Dict(
    :module_label      => F3_shape[1],
    :module_label_N    => F3_shape[1],
    :module_label_M    => F3_shape[1],
    :fusion_label      => F3_shape[2],
    :multiplicity_label=> F3_shape[end],
    :multiplicity_label_M => F3_shape[end],
    :multiplicity_label_N => F3_shape[end]
)

ϕ = (1+5^0.5)/2.0
quantum_dims = [1,ϕ,ϕ, ϕ^2]
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
F = SparseArray{ComplexF64, 10}(F3_DOK, F3_shape)
=#

##########################
# -  Doubled Haagerup  - #
##########################

# Haagerup example
index_file = "/home/lukehodgkiss/Documents/Part3Essay/Algebra data/Luke_Haagerup/Luke_Haagerup_ind.csv"
value_file = "/home/lukehodgkiss/Documents/Part3Essay/Algebra data/Luke_Haagerup/Luke_Haagerup_var.csv"

df_index = CSV.read(index_file, DataFrame; header=false)
df_var = CSV.read(value_file, DataFrame; header=false)

F3_DOK = Dict{CartesianIndex{10}, ComplexF64}()

for i in 1:nrow(df_index)
    row_idx = df_index[i, :]
    row_val = df_var[i, :]
    
    key = ntuple(j -> Int(row_idx[j]), 10)
    var = complex(row_val[1], row_val[2])

    F3_DOK[CartesianIndex(key)] = var
    #F3_DOK[CartesianIndex(key...)] = var
end

# Haagerup shape
F3_shape = (6, 36, 36, 6, 6, 36, 4, 4, 1, 4)
size_dict = Dict(
    :module_label      => F3_shape[1],
    :module_label_N    => F3_shape[1],
    :module_label_M    => F3_shape[1],
    :fusion_label      => F3_shape[2],
    :multiplicity_label=> F3_shape[end],
    :multiplicity_label_M => F3_shape[end],
    :multiplicity_label_N => F3_shape[end]
)
# Haag dims
Haagerup_dim_mat_file = matread("/home/lukehodgkiss/Documents/FindingTubesJulia/Haagerup data/Luke_H3xH3_dims.mat")
quantum_dims = vec(Haagerup_dim_mat_file["dimD"])
#quantum_dims = Haagerup_dim_mat_file["dimM"]
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
F = SparseArray{ComplexF64, 10}(F3_DOK, F3_shape)
#=

=#

d_algebra = 10000
N_M, N_M_sparsetensor = create_fusion_rules(F)
@time tube_map, tube_map_shape, tube_map_inv = create_tube_map(N_M, size_dict, F)
f_ijk_sparse = create_f_ijk_sparse(F, quantum_dims, size_dict, tube_map, tube_map_shape)
@time dimension_dict = create_dim_dict(size_dict, tube_map_shape)
tubealgebra = TubeAlgebra(N_diag_blocks, d_algebra, dimension_dict, f_ijk_sparse)
@show dimension_dict(1,1,1)
for i in 1:size_dict[:module_label_M]
    for j in 1:size_dict[:module_label_M]
        for k in 1:size_dict[:module_label_M]
            @show dimension_dict(i,j,k)
        end
    end
end

#@time f_ijk_sparse(1,1,1)

@show keys(tube_map_shape)
@show values(tube_map_shape)
