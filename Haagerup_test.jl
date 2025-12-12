module TestPackages

# Importing Paxkages
using SparseArrayKit: SparseArray, nonzero_values, nonzero_keys, nonzero_pairs
using MAT
using LinearAlgebra
using TensorOperations
using CSV, DataFrames
using StaticArrays: SVector


# Importing Locally
include("SparseAlgebraObjects.jl")
using .SparseAlgebraObjects: TubeAlgebra, random_left_linear_combination_ijk, random_right_linear_combination_ijk, create_left_ijk_basis, create_right_ijk_basis
include("F_symbolTools.jl")
using .FSymbolTools
include("IntertwinerTools.jl")
using .IntertwinerTools     
include("Idempotents.jl")
using .Idempotents: find_idempotents
include("Saving_Stuff.jl")
using .Saving_Stuff

# Read in F-symbols
"""
vars = matread("Luke_F.mat")
F = SparseArray{ComplexF64}(vars["F"])
quantum_dims = vec(vars["dim"])
size_dict = Dict(:module_label => size(F, 1),
                 :module_label_N => size(F, 1),
                 :module_label_M => size(F, 1),
                 :fusion_label => size(F, 2),
                 :multiplicity_label => size(F)[end])
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
"""

# Haagerup example
println("Reading in F symbol data")

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
    :multiplicity_label=> F3_shape[end]
)
quantum_dims = Float64[ 1.,  1.,  1.,  3.30277564,  3.30277564, 3.30277564,  1.,  1.,1.,  3.30277564, 3.30277564,  3.30277564,  1.,  1.,  1., 3.30277564,  3.30277564,  3.30277564,  3.30277564,  3.30277564, 3.30277564, 10.90832691, 10.90832691, 10.90832691,  3.30277564, 3.30277564,  3.30277564, 10.90832691, 10.90832691, 10.90832691, 3.30277564,  3.30277564,  3.30277564, 10.90832691, 10.90832691, 10.90832691 ]
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
F = SparseArray{ComplexF64, 10}(F3_DOK, F3_shape)

println("Finished reading in F symbol data")


# Create f_ijk sparse algebra
fusion_rules_M = make_fusion_rules(F, size_dict)
fusion_rules_N = make_fusion_rules(F, size_dict)
tubes_ij = make_tubes_ij(fusion_rules_M, fusion_rules_N)
f_ijk_sparse = make_f_ijk_sparse(F, F, quantum_dims, size_dict, tubes_ij)


#dimension_dict = compute_dim_dict(size_dict, tubes_ij)
dimension_dict = make_dim_dict(size_dict, tubes_ij)
N_diag_blocks = size_dict[:module_label_N]*size_dict[:module_label_M]
d_algebra_squared = N_diag_blocks*N_diag_blocks
tubealgebra = TubeAlgebra(N_diag_blocks, d_algebra_squared, dimension_dict, f_ijk_sparse)

#@show f_ijk_sparse(1,1,1)

# Calculate idempotents
@time idempotents_dict = find_idempotents(tubealgebra)
println(idempotents_dict)

#ω = construct_irreps(tubealgebra, idempotents_dict, size_dict, tubes_ij, quantum_dims)
#save_ω(ω)

#U = sparse_clebsch_gordon_coefficients(ω, quantum_dims)
#save_ω(U)

end # Module
