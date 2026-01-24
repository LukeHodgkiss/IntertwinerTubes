
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

# Read in F-symbol
println("Reading in F symbol data")

using CSV, DataFrames, SparseArrays

base_dir = "/home/lukehodgkiss/Documents/FindingTubesJulia/A4data/ModCats"


modcat_dirs = sort( filter(d -> startswith(d, "modcat") && isdir(joinpath(base_dir, d)), readdir(base_dir)) )

n_modcats = length(modcat_dirs)


F = Vector{SparseArray{ComplexF64,10}}(undef, n_modcats)
q_dims = Vector{Vector{Float64}}()

for k in 1:n_modcats
    folder = joinpath(base_dir, "modcat$k")

    ind_file  = joinpath(folder, "ind.csv")
    var_file  = joinpath(folder, "var.csv")
    size_file = joinpath(folder, "size.csv")
    qdim_file = joinpath(folder, "qdim.csv")
    

    df_index = CSV.read(ind_file, DataFrame; header=false)
    df_var   = CSV.read(var_file, DataFrame; header=false)
    df_size  = CSV.read(size_file, DataFrame; header=false)
    df_qdim  = CSV.read(qdim_file, DataFrame; header=false)

    shape = Tuple(Int.(Vector(df_size[1, :])))
    q_dim_k = (Vector(df_qdim[1, :]))
    n_mod_objects = shape[1]
    @assert n_mod_objects == length(q_dim_k)
    println("Mod cat $k has $n_mod_objects module objects with quantum dimensions: $(q_dim_k)")
    
    DOK = Dict{CartesianIndex{10}, ComplexF64}()

    for i in 1:nrow(df_index)
        idx = ntuple(j -> Int(df_index[i, j]), 10)
        val = complex(df_var[i, 1], df_var[i, 2])
        DOK[CartesianIndex(idx)] = val
    end

    F[k] = SparseArray{ComplexF64,10}(DOK, shape)
    push!(q_dims, q_dim_k)
end

# Fusion category

println("Finished reading in F symbol data")

function module_associator(F_M, F_N, d_Y, d_N)
    d_algebra = 100000 
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


#=
test = pentagon_eqn(ω_MN, F[N], F[M],ω_MN, ω_MN) # Maybe thery ar epermed so need nm and mn?
println("Beep Boop: $(test)")
if test > 1e-9
    println("Beep Boop: $(test)")
end
=#

#module_associator(F[M], F[N], q_dims[1], q_dims[N])'
M,N = 1, 5
println("Modcats: $((M,N))")       
ω_MN = module_associator(F[M], F[N], q_dims[1], q_dims[N] )
test = pentagon_eqn(ω_MN, F[N], F[M], ω_MN, ω_MN) 
if test > 1e-10
    println("Beep Boop: $(test)")
end

#=

ω = Matrix{SparseArray{ComplexF64, 10}}(undef, n_modcats, n_modcats)
for M in 1:n_modcats
    for N in 1:n_modcats
        println("Modcats: $((M,N))")
       
        ω[M,N] = module_associator(F[M], F[N], q_dims[1], q_dims[N] )
        test = pentagon_eqn(ω[M,N], F[N], F[M], ω[M,N], ω[M,N]) 
        if test > 1e-10
            println("Beep Boop: $(test)")
        end
    end 
end

=#

#=
  
U = Array{SparseArray{ComplexF64, 10}}(undef, n_modcats, n_modcats, n_modcats)
M,N,O = 1,1,1
println("Modcats: $((M,N,O))")
@time U[M,N,O] = sparse_clebsch_gordon_coefficients(ω[M,N], ω[M,O], ω[O, N], q_dims[1])
@show size(U[M,N,O])
@show size(ω[M,N])
pentagon_eqn(ω[M,N], U[M,N,O], ω[M,N], U[M,N,O], ω[O,N])

test = 0#pentagon_eqn(ω[M,N], ω[N,M], ω[M,N], U[M,N,O], ω[O,N])
if test > 1e-9
    println("Beep Boop: $(test)")
else
    println("Boop Beep: $(test)")
end

U = Array{SparseArray{ComplexF64, 10}}(undef, n_modcats, n_modcats, n_modcats)
for M in 1:n_modcats
    for N in 1:n_modcats
        for O in 1:n_modcats
            println("Modcats: $((M,N,O))")
            @time U[M,N,O] = sparse_clebsch_gordon_coefficients(ω[M,N], ω[M,O], ω[O, N], q_dims[1])
            
            test = pentagon_eqn(ω[M,N], ω[N,M], U[M,N,O], ω[M,N], ω[O,N])
            if test > 1e-9
                println("Beep Boop: $(test)")
            else
                println("Boop Beep: $(test)")
            end
            #==#
        end
    end
end
=#
#=
=#
