#module TestPackages

# Importing Paxkages
using SparseArrayKit: SparseArray, nonzero_values, nonzero_keys, nonzero_pairs, reindexdims
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
include("IntertwinerTools.jl")
using .IntertwinerTools
include("Idempotents.jl")
using .Idempotents: find_idempotents, dim_calc
include("Saving_Stuff.jl")
using .Saving_Stuff


function module_associator(F_M, F_N, d_Y, d_N)
    d_algebra = 27863 
    size_dict = Dict(:module_label_N => size(F_N, 1),
                     :module_label_M => size(F_M, 1),
                     :fusion_label => size(F_N, 2),
                     :multiplicity_label_M => size(F_M)[end],
                     :multiplicity_label_N => size(F_N)[end])

    N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]

    N_M, N_M_sparsetensor = create_fusion_rules(F_M)
    N_N, N_N_sparsetensor = create_fusion_rules(F_N)
    tubes_map, tube_map_shape, tube_map_inv = create_tube_map(N_M, N_N, size_dict)
    f_ijk_sparse = create_f_ijk_sparse(F_M, F_N, d_Y, size_dict, tubes_map, tube_map_shape, N_M, N_N)
   
    dimension_dict = create_dim_dict(size_dict, tubes_map, tube_map_shape, N_M, N_N)
    tubealgebra = TubeAlgebra(N_diag_blocks, d_algebra, dimension_dict, f_ijk_sparse)
    idempotents_dict = find_idempotents(tubealgebra)
    @show length(idempotents_dict)
    expected_size_ω = length(F_M.data)
    ω_MN = construct_irreps(tubealgebra, idempotents_dict, size_dict, tube_map_inv, d_Y, d_N, create_left_ijk_basis, expected_size_ω)

    return ω_MN
end

# Read in F-symbol
println("Reading in F symbol data")

##########################
# -  Doubled Haagerup  - #
##########################
#=
println("M=N=Doubled Haagerup")
base_dir = @__DIR__
index_file = joinpath(base_dir, "Haagerup data/Luke_Haagerup_ind.csv")
value_file = joinpath(base_dir, "Haagerup data/Luke_Haagerup_var.csv")

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
    :multiplicity_label_N => F3_shape[end])

# Haag dims
Haagerup_dim_mat_file = matread("Haagerup data/Luke_H3xH3_dims.mat")
quantum_dims = vec(Haagerup_dim_mat_file["dimD"])
#quantum_dims = Haagerup_dim_mat_file["dimM"]
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
F = SparseArray{ComplexF64, 10}(F3_DOK, F3_shape)
=#

########################
# -  Vec over Vec G  - #
########################
println("N = Vec over Vec_S3")

cayley_table_S3 = [ 1 2 3 4 5 6;
                    2 1 4 3 6 5;
                    3 5 1 6 2 4;
                    4 6 2 5 1 3;
                    5 3 6 1 4 2;
                    6 4 5 2 3 1 ]

F_M = F_mod_cat_Vec_Vec_G(cayley_table_S3)
quantum_dims = vec(ones(size(cayley_table_S3, 1)))

########################## 
# -  Vec_G over Vec_G  - #
########################## 
println("N = Vec_S2 over Vec_S3")
cayley_table_S3 = [ 1 2 3 4 5 6;
                    2 1 4 3 6 5;
                    3 5 1 6 2 4;
                    4 6 2 5 1 3;
                    5 3 6 1 4 2;
                    6 4 5 2 3 1 ]

F_N = F_mod_cat_Vec_G_Vec_G(cayley_table_S3)
quantum_dims = vec(ones(size(cayley_table_S3, 1)))

size_dict = Dict(
                 :module_label_N => size(F_N, 1),
                 :module_label_M => size(F_M, 1),
                 :fusion_label => size(F_N, 2),
                 :multiplicity_label_M => size(F_M)[end],
                 :multiplicity_label_N => size(F_N)[end])

N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]

quantum_dims_N = quantum_dims
quantum_dims_Y = quantum_dims

@time ω_MN = module_associator(F_M, F_N, quantum_dims_Y, quantum_dims_N)

test = pentagon_eqn(ω_MN, F_N, F_M, ω_MN, ω_MN) 
if test > 1e-9
    println("Beep Boop: $(test)")
else
    println("Boop Beep: $(test)")
end

