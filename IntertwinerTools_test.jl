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
using .Idempotents: find_idempotents, dim_calc
include("Saving_Stuff.jl")
using .Saving_Stuff

# Read in F-symbol
println("Reading in F symbol data")


####################################
# -- Input Data for intertwiner -- #
####################################
#= Input is the 3F symbol, ω = 2F, U = 1F, then we output 0F  =#

############################
# -  Doubled Fibonnacci  - #
############################
#=
=#

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
    :module_label_N    => F3_shape[1],
    :module_label_M    => F3_shape[1],
    :fusion_label      => F3_shape[2],
    :multiplicity_label_M => F3_shape[end],
    :multiplicity_label_N => F3_shape[end]
)

ϕ = (1+5^0.5)/2.0
quantum_dims = [1,ϕ,ϕ, ϕ^2]
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
F = SparseArray{ComplexF64, 10}(F3_DOK, F3_shape)
F_N = deepcopy(F)
F_M = deepcopy(F)
#@show nonzero_values(F)


###################################
# -  Rep A4 over Rep A4 Rep A4  - #
###################################
#=

vars = matread("/home/lukehodgkiss/Documents/FindingTubesJulia/RepA4Data/Luke_F.mat")
F = SparseArray{ComplexF64}(vars["F"])
quantum_dims = vec([1.0 1.0 1.0 3.0])
size_dict = Dict(:module_label => size(F, 1),
                 :module_label_N => size(F, 1),
                 :module_label_M => size(F, 1),
                 :fusion_label => size(F, 2),
                 :multiplicity_label => size(F)[end],
                 :multiplicity_label_M => size(F)[end],
                 :multiplicity_label_N => size(F)[end])

N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]

F_M = deepcopy(F)
F_N = deepcopy(F)

=#

##########################
# -  Doubled Haagerup  - #
##########################
#=

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
    :multiplicity_label_N => F3_shape[end])

# Haag dims
Haagerup_dim_mat_file = matread("/home/lukehodgkiss/Documents/FindingTubesJulia/Haagerup data/Luke_H3xH3_dims.mat")
quantum_dims = vec(Haagerup_dim_mat_file["dimD"])
#quantum_dims = Haagerup_dim_mat_file["dimM"]
N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
F = SparseArray{ComplexF64, 10}(F3_DOK, F3_shape)

=#

########################
# -  Vec over Vec G  - #
########################
#=

cayley_table_S3 = [ 1 2 3 4 5 6;
                    2 1 4 3 6 5;
                    3 5 1 6 2 4;
                    4 6 2 5 1 3;
                    5 3 6 1 4 2;
                    6 4 5 2 3 1 ]

@show cayley_table_S3[1,1]
@show cayley_table_S3[1,2]

F_M = F_mod_cat_Vec_Vec_G(cayley_table_S3)
quantum_dims = vec(ones(size(cayley_table_S3, 1)))
F = F_M
#F_N = F_M

size_dict = Dict(:module_label => size(F, 1),
                 :module_label_N => size(F, 1),
                 :module_label_M => size(F, 1),
                 :fusion_label => size(F, 2),
                 :multiplicity_label => size(F)[end],
                 :multiplicity_label_M => size(F)[end],
                 :multiplicity_label_N => size(F)[end])

N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
#=
=#
########################## 
# -  Vec_G over Vec_G  - #
########################## 
cayley_table_S3 = [ 1 2 3 4 5 6;
                    2 1 4 3 6 5;
                    3 5 1 6 2 4;
                    4 6 2 5 1 3;
                    5 3 6 1 4 2;
                    6 4 5 2 3 1 ]

#g1,g2 != g1g2, g2g3 for g1 = 4, g2 = 1

F_N = F_mod_cat_Vec_G_Vec_G(cayley_table_S3)
quantum_dims = vec(ones(size(cayley_table_S3, 1)))
#F = deepcopy(F_N)
#F_M = deepcopy(F_N)
size_dict = Dict(
                 :module_label_N => size(F_N, 1),
                 :module_label_M => size(F_M, 1),
                 :fusion_label => size(F_N, 2),
                 :multiplicity_label_M => size(F_M)[end],
                 :multiplicity_label_N => size(F_N)[end])

N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
=#


println("Finished reading in F symbol data")

println("Sparsity of F_N: $(length(nonzero_keys(F_N))/ prod(size(F_N)))")
println("Sparsity of F_M: $(length(nonzero_keys(F_M))/ prod(size(F_M)))")


@show size(F_M)
d_algebra = 27863 
N_M, N_M_sparsetensor = create_fusion_rules(F_M)
N_N, N_N_sparsetensor = create_fusion_rules(F_N)
tubes_map, tube_map_shape, tube_map_inv = create_tube_map(N_M, N_N, size_dict)
f_ijk_sparse = create_f_ijk_sparse(F_M, F_N, quantum_dims, size_dict, tubes_map, tube_map_shape, N_M, N_N)
dimension_dict = create_dim_dict(size_dict, tubes_map, tube_map_shape, N_M, N_N)
tubealgebra = TubeAlgebra(N_diag_blocks, d_algebra, dimension_dict, f_ijk_sparse)

#=
@show dimension_dict(16,16,16)
@show tube_map_inv[(4,4,4,4,7)]
@show f_ijk_sparse(16,16,16)[3,3,3,1,1,1,1,1,1]

for key in keys(tubes_map)
    if key[1]==4 && key[2]==4 && key[3]==4 && key[4]==4
        println(key, tubes_map[key], tube_map_inv[key[1:4]..., tubes_map[key]])
    end
end

f_abc = sort([Tuple(key) for key in nonzero_keys(f_ijk_sparse(16,16,16))])
for f in f_abc
    println(f) 
end
=#

println(" Fusion Rules ")
#@show N_M

println(" Non-zero Hom spaces ")
#@show tube_map_shape
println("This is the way")
#@show keys(tube_map_shape)
@show length(tube_map_shape) 

idempotents_dict = find_idempotents(tubealgebra) # Compilation
dim_calc(idempotents_dict)

#####################
# -  Idempotents  - #
#####################

#=
Profile.clear()
#Profile.init(n = 10^7, delay = 0.001)
#@profile find_idempotents(tubealgebra)
@profview begin
    idempotents_dict = find_idempotents(tubealgebra)
end
Profile.print(format = :flat, sortedby = :count)
ProfileView.view()
=#

############################## 
# -  Module Associator: ω  - #
##############################
println("Computing ω")
@time ω = construct_irreps(tubealgebra, idempotents_dict, size_dict, tube_map_inv, quantum_dims, create_left_ijk_basis, F)
println("Sparsity of ω: $(length(nonzero_keys(ω))/ prod(size(ω)))")
save_ω(ω)

####################################
# -  Testing Pentagon Condition  - #
####################################

@show size(F_N)
@show size(F_M)
@show size(ω)

function make_mpo(F)
    mpo = reindexdims(F,(5,1,2,7, 4,1,6,10, 2,3,6,9, 5,3,4,8))
    mpo = reshape(mpo,(prod(size(F)[[5,1,2,7]]),prod(size(F)[[4,1,6,10]]),prod(size(F)[[2,3,6,9]]),prod(size(F)[[5,3,4,8]])))
    return mpo
end

function make_peps(F)
    peps = reindexdims(F,(1,2,5,7, 5,3,4,8, 1,6,4,10, 2,3,6,9))
    peps = reshape(peps,(prod(size(F)[[1,2,5,7]]),prod(size(F)[[5,3,4,8]]),prod(size(F)[[1,6,4,10]]),prod(size(F)[[2,3,6,9]])))
    return peps
end

function pentagon_eqn(F1, F2, F3, F4, F5)
    
    @tensor lhs[-1 -2 -3 -4 -5 -6] := make_mpo(F1)[-1 -2 -3 1] * make_peps(F2)[-4 -5 1 -6]
    @tensor rhs[-1 -2 -3 -4 -5 -6] := make_peps(F3)[1 2 -3 -6] * make_mpo(F4)[-1 3 1 -4] * make_mpo(F5)[3 -2 2 -5]

    test = norm(lhs-rhs)
    @show test
    return test
end

mpo = make_mpo(ω)
peps_M = make_peps(F_M)
peps_N = make_peps(F_N)

@show size(mpo)
@show size(peps_M)
@show size(peps_N)

pentagon_eqn(ω, F_N, F_M, ω, ω)

#pentagon_eqn(F_N, F_N, F_N, F_N, F_N)
#pentagon_eqn(ω, ω, ω, ω, ω)
#pentagon_eqn(F, ω, ω, F, F)


#@show dropnearzeros!(ω - F)

#=
F_doubled = reindexdims(F,(1,1,2,3,4,4,5,6,6,7,8,9,10))
@tensor lhs[a, b, c, d, e, f, g, h, i, k, l, m, n, o, p] := F_doubled[a, a_, b, c, d, d_, e, f, f_, g, h, i, j] * ω[k, l, f_, d_, a_, m, n, j, o, p]

F_doubled1 = reindexdims(F, (1,1, 2,2, 3,3, 4,4, 5, 6,6, 7,8,9,10))
F_doubled2 = reindexdims(F, (1,2,3,3,4,5,6,6,7,8,9,10))
@tensor rhs[a, b, c, d, e, f, g, h, i, k, l, m, n, o, p] := F_doubled1[k, k_, l, l_, b, b_, e, e_, a, q, q_, n, g, r, s] * F_doubled2[k_, q, c, c_, d, e_, m, m_, s, h, t, p] * ω[l_, b_, c_, m_, q_, f, r, t, i, o]
@show norm(lhs-rhs)


=#
#=
Profile.clear()
#Profile.init(n = 10^7, delay = 0.001)
#@profile find_idempotents(tubealgebra)
@profview begin
    ω = construct_irreps(tubealgebra, idempotents_dict, size_dict, tube_map_inv, quantum_dims, create_left_ijk_basis, F)
end
=#

#########################################
# -  Clebsch Gorndon Coefficients: U  - #
#########################################

#=
@time U = sparse_clebsch_gordon_coefficients(ω, quantum_dims)
println("Sparsity of U: $(length(nonzero_keys(U))/ prod(size(U)))")
=#

#save_ω(U)

####################### 
# -  6j Symbols: F  - #
####################### 



################### 
# -  Testing ω  - #
################### 

#=
#ω = reindexdims(ω, (1,2,3,4,5,6,10,9,8,7))
size_dict = Dict(:module_label => size(ω, 1),
                 :module_label_N => size(ω, 4),
                 :module_label_M => size(ω, 2),
                 :fusion_label => size(ω, 3),
                 :multiplicity_label => size(ω)[end],
                 :multiplicity_label_M => size(ω)[end],
                 :multiplicity_label_N => size(ω)[end])
@show keys(N_M)
@time N_M, N_M_sparsetensor = create_fusion_rules(ω)
@show keys(N_M)
@time tubes_map, tube_map_shape, tube_map_inv = create_tube_map(N_M, N_M, size_dict)
f_ijk_sparse = create_f_ijk_sparse(ω, conj!(ω), quantum_dims, size_dict, tubes_map, tube_map_shape, N_M, N_M)
dimension_dict = create_dim_dict(size_dict, tubes_map, tube_map_shape, N_M, N_M)
tubealgebra = TubeAlgebra(N_diag_blocks, d_algebra, dimension_dict, f_ijk_sparse)

@time idempotents_dict = find_idempotents(tubealgebra) # Compilation
dim_calc(idempotents_dict)
=#

#end # Module
