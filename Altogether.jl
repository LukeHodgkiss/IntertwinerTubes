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
    println(" Finding Idempotents ")
    @time idempotents_dict = find_idempotents(tubealgebra)
    @show length(idempotents_dict)

    expected_size_ω = length(F_M.data)
    println(" Constructing ω_MN ")
    @time ω_MN = construct_irreps(tubealgebra, idempotents_dict, size_dict, tube_map_inv, d_Y, d_N, create_left_ijk_basis, expected_size_ω)
    return ω_MN
end

# Read in F-symbol
println("Reading in F symbol data")

##########################
# -  Doubled Haagerup  - #
##########################

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
F_M = F
F_N = F
#=

=#

##########################
# -  SU_2_8  - #
##########################
#=

println("M=N=SU_2_8")
base_dir = @__DIR__
index_file = joinpath(base_dir, "SU_2_k/su2_8_ind.csv")
value_file = joinpath(base_dir, "SU_2_k/su2_8_var.csv")
size_file = joinpath(base_dir, "SU_2_k/su2_8_size.csv")
qdim_file = joinpath(base_dir, "SU_2_k/su2_8_dim.csv")
    

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

df_size  = CSV.read(size_file, DataFrame; header=false)
df_qdim  = CSV.read(qdim_file, DataFrame; header=false)

shape = Tuple(Int.(Vector(df_size[1, :])))
quantum_dims = (Vector(df_qdim[1, :]))

size_dict = Dict(
    :module_label      => shape[1],
    :module_label_N    => shape[1],
    :module_label_M    => shape[1],
    :fusion_label      => shape[2],
    :multiplicity_label=> shape[end],
    :multiplicity_label_M => shape[end],
    :multiplicity_label_N => shape[end])


N_diag_blocks = size_dict[:module_label_M] * size_dict[:module_label_N]
F = SparseArray{ComplexF64, 10}(F3_DOK, shape)

F_M = F
F_N = F

=#

#=
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
=#



#@time ω_MN = module_associator(F_M, F_N, quantum_dims_Y, quantum_dims_N)
#@time ω_MN = module_associator(F_M, F_N, quantum_dims_Y, quantum_dims_N)
#==#
@time begin
F_M_doubled = reindexdims(F_M, (1,2,2,3,3,4,5,6,6,7,8,9,10))
@tensor f[M1, M2, M3, N1, N2, N3, Y1, m1, n1, Y2, m2, n2, Y3, m3, n3] := conj(F_M_doubled[M1, Y1, Y1_, Y2, Y2_, M3, M2, Y3, Y3_, m1, m2, mn, m3]) * F_N[N1, Y1_, Y2_, N3, N2, Y3_, n1, n2, mn, n3,]
end

@time begin
    
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
f_ijk_sparse = create_f_ijk_sparse(F_M, F_N, quantum_dims, size_dict, tubes_map, tube_map_shape, N_M, N_N)
dimension_dict = create_dim_dict(size_dict, tubes_map, tube_map_shape, N_M, N_N)

#dim_calc(idempotents_dict)
end

##################
@time begin
    
F_M_doubled = reindexdims(F_M, (1,2,2,3,3,4,5,6,6,7,8,9,10))
@tensor f[M1, M2, M3, N1, N2, N3, Y1, m1, n1, Y2, m2, n2, Y3, m3, n3] := conj(F_M_doubled[M1, Y1, Y1_, Y2, Y2_, M3, M2, Y3, Y3_, m1, m2, mn, m3]) * F_N[N1, Y1_, Y2_, N3, N2, Y3_, n1, n2, mn, n3]

##################

tube_map_inv_DOK = Dict{NTuple{8,Int}, Int}()
for (key_inv, val_Ymn) in tube_map_inv
    M2, M1, N1, N2, lin_ind = key_inv 
    Y, m, n = val_Ymn
    tube_map_inv_DOK[M2, M1, N1, N2, Y, m, n, lin_ind] = 1
end
tube_map_inv_DOK = Dict( CartesianIndex(Tuple(k)) => v for (k,v) in tube_map_inv_DOK )
maxvals = zeros(Int, 8)
@inbounds for k in keys(tube_map_inv_DOK); for d in 1:8; maxvals[d] = max(maxvals[d], k[d]); end; end
tube_map_shape = Tuple(maxvals)
tube_map_inv_tens = SparseArray{Int,8}(tube_map_inv_DOK, tube_map_shape)
end

##################

tube_map_DOK = Dict{NTuple{8,Int}, Int}()
for (key, a) in tubes_map
    M2, M1, N1, N2, Y, m, n = key
    tube_map_DOK[M2, M1, N1, N2, Y, m, n, a] = 1
end
tube_map_DOK = Dict( CartesianIndex(Tuple(k)) => v for (k,v) in tube_map_DOK )
maxvals = zeros(Int, 8)
@inbounds for k in keys(tube_map_DOK); for d in 1:8; maxvals[d] = max(maxvals[d], k[d]); end; end
tube_map_shape = Tuple(maxvals)
tube_map_tens = SparseArray{Int,8}(tube_map_DOK, tube_map_shape) # Actually thesmaa sinvverse lol

##############

f_doubled = reindexdims(f, (1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,8,9,10,11,12,13,14,15))
println("CalculATING f_ijk_abc")
@time begin

@tensor f_ijk_abc[M1, N1, M2, N2, M3, N3 ,a,b,c] := (
    f_doubled[M1, M1_, M1__, M2, M2_, M2__, M3, M3_, M3__, N1, N1_, N1__, N2, N2_, N2__, N3, N3_, N3__, Y1, m1, n1, Y2, m2, n2, Y3, m3, n3] *
    tube_map_tens[M2_, M1_, N1_, N2_, Y1, m1, n1, a] *
    tube_map_tens[M3_, M2__, N2__, N3_, Y2, m2, n2, b] *
    tube_map_tens[M3__, M1__, N1__, N3__, Y3, m3, n3, c] 
)
end

MN_to_a_map = CartesianIndices((size_dict[:module_label_M], size_dict[:module_label_N]))

function f_ijk_sparse_altogether(i, j, k)
    
    M1, N1 = Tuple(MN_to_a_map[i])
    M2, N2 = Tuple(MN_to_a_map[j])
    M3, N3 = Tuple(MN_to_a_map[k])
    
    return f_ijk_abc[M1, N1, M2, N2, M3, N3, :,:,:]
end 

##############
println("CalculATING Idempotents")
@time  begin
tubealgebra = TubeAlgebra(N_diag_blocks, d_algebra, dimension_dict, f_ijk_sparse_altogether)
idempotents_dict = find_idempotents(tubealgebra)
end

##############
println("CalculATING Q_map")
@time begin
MN_to_a_map = CartesianIndices((size_dict[:module_label_M], size_dict[:module_label_N]))
#Q = Dict{NTuple{5,Int}, ComplexF64}()
Q = Dict{NTuple{7,Int}, ComplexF64}()
for X in 1:length(idempotents_dict); irrep_X = idempotents_dict[X];
    for (ik, Q_ik) in irrep_X;i, k = ik
        M1, N1 = Tuple(MN_to_a_map[i])
        M3, N3 = Tuple(MN_to_a_map[k])

        for col in axes(Q_ik,2), row in axes(Q_ik,1)
        #Q[X, i, k, row, col] = Q_ik[row, col]
        Q[X, M1, N1, M3, N3, row, col] = Q_ik[row, col]
        end
    end
end

Q_shape = (length(idempotents_dict), size_dict[:module_label_M], size_dict[:module_label_N], size_dict[:module_label_M], size_dict[:module_label_N], tube_map_shape[end], tube_map_shape[end])
Q_DOK = Dict(CartesianIndex(Tuple(k)) => v for (k,v) in Q)
Q_map = SparseArray{ComplexF64, 7}(Q_DOK, Q_shape)

tube_map_inv_tens_doubled = reindexdims(tube_map_inv_tens, (1,1,2,2,3,3,4,4,5,6,7,8))
@tensor Q_col[X, M1, N1, M2, N2, Y_r, m_r, n_r, row] := (
            Q_map[X, M1_, N1_, M2_, N2_, row, lin_ind_c] *
            tube_map_inv_tens_doubled[M2, M2_, M1, M1_, N1, N1_, N2, N2_, Y_r, m_r, n_r, lin_ind_c] )

end

##################
@time begin 
println("CalculATING ω")
f_doubled = reindexdims(f, (1,1,2,2,3,3,4,4,5,5,6,6,7,8,9,10,11,12,13,14,15))
Q_col_doubled = reindexdims(Q_col, (1,1, 2,3,4,5, 6,7,8,9))
@tensor ρ[X, M1, Y1, N2, N1, M2, row, n1, m1, col] := (
            conj(Q_col_doubled[X, X_, M1_, N1_, M3_, N3_, Y_r, m_r, n_r, row]) *
            f_doubled[M1, M1_, M2, M2_, M3, M3_, N1, N1_, N2, N2_, N3, N3_, Y1, m1, n1, Y_c, m_c, n_c, Y_r, m_r, n_r] *
            Q_col[X_, M2_, N2_, M3, N3, Y_c, m_c, n_c, col] )
end

quantum_dims_dok = Dict{NTuple{1,Int}, Float64}(); for i in 1:size(quantum_dims)[1]; quantum_dims_dok[Tuple(i)] = quantum_dims[i]; end
quantum_dims_dok = Dict(CartesianIndex(Tuple(k)) => v for (k,v) in quantum_dims_dok)
quantum_dims_tens = SparseArray{Float64, 1}(quantum_dims_dok, size(quantum_dims))

d_Y, d_N = SparseArray(quantum_dims_tens), SparseArray(quantum_dims_tens)
d_N_doubled, d_Y_doubled = reindexdims(d_N, (1,1)), reindexdims(d_Y, (1,1))

println("CalculATING pentagon equation")
@time test = pentagon_eqn(ρ, F_N, F_M, ρ, ρ) 
if test > 1e-9
    println("Beep Boop: $(test)")
else
    println("Boop Beep: $(test)")
end



#@tensor ρ_unitary[X, M1, Y1, N2, N1, M2, row, n1, m1, col] := ρ[X, M1, Y1_, N2_, N1_, M2, row, n1, m1, col] * sqrt(d_N_doubled[N1, N1_])/(sqrt(d_N_doubled[N2, N2_] * d_Y_doubled[Y1, Y1_]))

#@time ω_MN = module_associator(F_M, F_N, quantum_dims_Y, quantum_dims_N)

#@show norm(abs(ρ-ω_MN))
