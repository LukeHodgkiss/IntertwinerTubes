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
include("IntertwinerTools.jl")
using .IntertwinerTools
include("Idempotents.jl")
using .Idempotents: find_idempotents
include("Saving_Stuff.jl")
using .Saving_Stuff

# Read in F-symbol

global dim_alg_glob = 0
println("Reading in F symbol data")

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
    :multiplicity_label=> F3_shape[end]
)

ϕ = (1+5^0.5)/2.0
quantum_dims = [1,ϕ,ϕ, ϕ^2]
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
F = SparseArray{ComplexF64, 10}(F3_DOK, F3_shape)
=#

#=
# Rep A4 over Rep A4 Rep A4
vars = matread("Luke_F.mat")
F = SparseArray{ComplexF64}(vars["F"])
quantum_dims = vec(vars["dim"])
size_dict = Dict(:module_label => size(F, 1),
                 :module_label_N => size(F, 1),
                 :module_label_M => size(F, 1),
                 :fusion_label => size(F, 2),
                 :multiplicity_label => size(F)[end])
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
=#


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
    :multiplicity_label=> F3_shape[end]
)
# Haag dims
Haagerup_dim_mat_file = matread("/home/lukehodgkiss/Documents/FindingTubesJulia/Haagerup data/Luke_H3xH3_dims.mat")
quantum_dims = vec(Haagerup_dim_mat_file["dimD"])
#quantum_dims = Haagerup_dim_mat_file["dimM"]
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
F = SparseArray{ComplexF64, 10}(F3_DOK, F3_shape)


println("Finished reading in F symbol data")

# Create f_ijk sparse algebra
fusion_rules_M = make_fusion_rules(F, size_dict)
fusion_rules_N = make_fusion_rules(F, size_dict)
#=
N_abc = reindexdims(F[:,:,:,:,:,1,:,:,:,:], (2,4,5,7)) # Y, M1, M2, μ
T_shape = ones(size(N_abc, 4))
@tensor N[Y, M1, M2] = N_abc[Y, M1, M2, μ]*T_shape[μ]
M_abc = reindexdims(F[:,:,:,:,:,1,:,:,:,:], (2,4,5,7))# Y, N1, N2, ν
@tensor M[Y, N1, N2] = N_abc[Y, N1, N2, μ]*T_shape[μ]
#@tensor tubes[]
=#

tubes_ij = make_tubes_ij(fusion_rules_M, fusion_rules_N)
f_ijk_sparse = make_f_ijk_sparse(F, conj!(F), quantum_dims, size_dict, tubes_ij)

#dimension_dict = compute_dim_dict(size_dict, tubes_ij)
dimension_dict = make_dim_dict(size_dict, tubes_ij)

#=
for i in 1:4
    #i = 10
    @show is_associative(f_ijk_sparse(i,i,i), 1e-14)
end
=#

N_diag_blocks = size_dict[:module_label_N]*size_dict[:module_label_M]
#d_algebra_squared = 27863#N_diag_blocks*N_diag_blocks # calculate by contracting fusion tensor then summing up all the entries
d_algebra = 27863  #N_diag_blocks*N_diag_blocks # calculate by contracting fusion tensor then summing up all the entries
tubealgebra = TubeAlgebra(N_diag_blocks, d_algebra, dimension_dict, f_ijk_sparse)

#@show f_ijk_sparse(4,1,1)

# Calculate idempotents



Profile.clear()
#Profile.init(n = 10^7, delay = 0.001)
#@profile find_idempotents(tubealgebra)
@profview find_idempotents(tubealgebra)
Profile.print(format = :flat, sortedby = :count)
ProfileView.view()

#idempotents_dict = find_idempotents(tubealgebra)
#=
for irrep in idempotents_dict
    println("Irrep has size: $(length(irrep))")
    
    for (ij, proj) in irrep
        println("Projector $(ij) has shape $(size(proj)) ")
        dim_alg_glob=dim_alg_glob+(size(proj)[2]^2)
    end
    
end
=#
@show dim_alg_glob
#@show idempotents_dict[2]
#println(size(irrep) for irrep in idempotents_dict)

#ω = construct_irreps(tubealgebra, idempotents_dict, size_dict, tubes_ij, quantum_dims)
#save_ω(ω)

#U = sparse_clebsch_gordon_coefficients(ω, quantum_dims)
#save_ω(U)
#=

full_dimension_dict = compute_dim_dict(size_dict, tubes_ij)
println(full_dimension_dict)

println(dimension_dict(1,1))
=#

#end # Module
