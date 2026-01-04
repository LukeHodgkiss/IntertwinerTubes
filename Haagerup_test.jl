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
#=
=#

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
@show length(f_ijk_sparse(12,12,1))

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
#@show f_ijk_sparse(12,12,12)

#@show f_ijk_sparse(1,1,1)

# Calculate idempotents
#=
Profile.clear()
#Profile.init(n = 10^7, delay = 0.001)
#@profile find_idempotents(tubealgebra)
@profview find_idempotents(tubealgebra)
Profile.print(format = :flat, sortedby = :count)
ProfileView.view()

idempotents_dict = find_idempotents(tubealgebra)
=#

function dim_calc(idempotents_dict)
    dim_alg_glob = 0
    for irrep in idempotents_dict
        println("Irrep has size: $(length(irrep))")
        
        for (ij, proj) in irrep
            println("Projector $(ij) has shape $(size(proj)) ")
            dim_alg_glob=dim_alg_glob+((size(proj)[2])^2)
        end 
    end
    #@show dim_alg_glob
end
#dim_calc(idempotents_dict)

function create_fusion_rules(F)
    #F[M2, Y1, Y, M1, M2, Y, 1, :, 1, :]
    #F = reindexdims(F, (1,5,3,8))
    #F = reindexdims(F, (4,5,3,10))

    hom_space = slice_sparse_tensor(F, Dict(2=>1, 7=>1, 9=>1)) #F[M2, Y1, Y, M1, M2, Y, 1, :, 1, :]
    # 1=>M2, 3=>Y, 4=>M1, 5=>M2, 6=>Y, 
    F = reindexdims(hom_space, (3,1,2,6)) #M1, M2, Y: k


    N_M1 = Dict{CartesianIndex{3}, Int}()

    for indx in nonzero_keys(F)
        @inbounds begin
            key = CartesianIndex(indx[1], indx[2], indx[3])
            N_M1[key] = get(N_M1, key, 0) + 1
        end
    end
    
    return N_M1
end 
@time N_M = create_fusion_rules(F)


function create_tube_map(N_M, N_N, size_dict)
    tube_map = Dict{NTuple{7,Int}, Int}()
    tube_map_shape = Dict{NTuple{4,Int}, Int}() # Add expected shape as #N**2 * #M**2 and set initial value to 0 
    sizehint!(tube_map_shape, size_dict[:module_label_M]^2 * size_dict[:module_label_N]^2 )

    # M1, M2, N1, N2, Y, m, n, 
    #function tubes_ij(N2::Int, N1::Int, M1::Int, M2::Int)

    for (indx_M, val_M) in N_M
        M1, M2, Y = indx_M.I  
        #println(indx_M)

        for (indx_N, val_N) in N_N
            N1, N2, YN = indx_N.I
            #println(indx_N)

            linear_index = 1 # need to check if for this given Y, YN pair is the M1, M2, N1, N2 seen before
            # Can we just do:
            #common_Y = sort(collect(intersect(keys(N_N), keys(N_M))))
            
            if Y == YN
                #@show val_M, val_N

                @inbounds for m in 1:val_M
                    @inbounds for n in 1:val_N
                        #key = (M1, M2, N1, N2, Y, m, n)
                        #println(m, n)
                        #key = (N2, N1, M1, M2, Y, m, n)
                        #key = (M2, M1, N1, N2, Y, m, n)

                        #tube_map_shape[(M2, M1, N1, N2)] = get(tube_map_shape, (M2, M1, N1, N2), 0) + 1 # linear_index
                        #tube_map[key] = tube_map_shape[(M2, M1, N1, N2)] 

                        key = (M1, M2, N2, N1, Y, m, n)

                        tube_map_shape[(M1, M2, N2, N1)] = get(tube_map_shape, (M1, M2, N2, N1), 0) + 1 # linear_index
                        tube_map[key] = tube_map_shape[(M1, M2, N2, N1)] 
                        
                    end
                end

                #tube_map_shape[(M1, M2, N1, N2)] = val_M*val_N
            end
        end
    end
    
    return tube_map, tube_map_shape
end

@time tubes_ij, tube_map_shape = create_tube_map(N_M, N_M, size_dict)
@show get(N_M, (1,2,3), 0)
@show get(N_M, (2,1,3), 0)

#@show tubes_ij[(2, 1, 1, 2, 3, 1, 1)]
#@show tubes_ij[(1, 2, 2, 1, 3, 1, 1)]


f_ijk_sparse = create_f_ijk_sparse(F, conj!(F), quantum_dims, size_dict, tubes_ij, tube_map_shape, N_M, N_M)
dimension_dict = create_dim_dict(size_dict, tubes_ij, tube_map_shape, N_M, N_M)
tubealgebra = TubeAlgebra(N_diag_blocks, d_algebra, dimension_dict, f_ijk_sparse)

@time f = f_ijk_sparse(12,12,1)
@show length(f)
@time idempotents_dict = find_idempotents(tubealgebra)
#@show idempotents_dict = find_idempotents(tubealgebra)
dim_calc(idempotents_dict)

#=
N_M1 = reindexdims(F, (1,5,3,10))
@show length(nonzero_pairs(N_M1) )

N_M1.= 1
@show length(nonzero_pairs(N_M1) )
temp_vec = SparseArray(ones(size(N_M1)[4]))
print(size(N_M1))
#N_M1 = dropdims(N_M1, dims=4)
@tensor N_M[M1,M2,Y] := N_M1[M1,M2,Y,k]*temp_vec[k]
#@show N_M1
#@tensor N_M1[M1,M2,Y] := N_M1[M1,M2,Y, k]
=#


#==#
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
