module MonoidalCentreTools

export create_fusion_rules, create_tube_map, create_f_ijk_sparse, create_dim_dict

using Base.Threads
using SparseArrayKit
using LinearAlgebra
using TensorOperations
using StaticArrays: SVector, @SVector
using Arpack 
include("F_symbolTools.jl")
using .FSymbolTools    

function sparsity(U)
    return length(nonzero_keys(U))/ prod(size(U))
end

function expected_nnz(sparsityA::Float64, sparsityB::Float64, contracted_sizes, free_sizes)

    K = prod(contracted_sizes)
    p_entry = 1 - (1 - sparsityA * sparsityB)^K #Assumining wholly unstructured
    total_entries = prod(free_sizes)

    return round(Int, total_entries * p_entry)
end

function create_fusion_rules(F)
    hom_space = slice_sparse_tensor(F, Dict(2=>1, 7=>1, 9=>1)) #F[M2, Y1, Y, M1, M2, Y, 1, :, 1, :] # 1=>M2, 3=>Y, 4=>M1, 5=>M2, 6=>Y, 
    F = reindexdims(hom_space, (3,1,2,6)) #M1, M2, Y: k

    N_M1 = Dict{CartesianIndex{3}, Int}()

    for indx in nonzero_keys(F)
        @inbounds begin
            key = CartesianIndex(indx[1], indx[2], indx[3])
            N_M1[key] = get(N_M1, key, 0) + 1
        end
    end

    shape = size(F)[1:3]
    N_M_sparsetensor = SparseArray{Int,3}(N_M1, shape)
    return N_M1, N_M_sparsetensor
end 

function create_tube_map(N_M, size_dict, F)
    
    #@tensor N_M_by_N_N[M1,N1] := N_M[M1, MN, Y]*N_M[N1, MN, Y]

    tube_map = Dict{NTuple{6,Int}, Int}()
    expected_tube_map_nnz = expected_nnz(sparsity(F), sparsity(F), (size_dict[:fusion_label],), (size_dict[:module_label_M], size_dict[:module_label_N], size_dict[:fusion_label], size_dict[:multiplicity_label_M], size_dict[:multiplicity_label_N]) )
    sizehint!(tube_map, expected_tube_map_nnz)

    tube_map_inv = Dict{NTuple{3,Int}, NTuple{4,Int}}()
    sizehint!(tube_map_inv, expected_tube_map_nnz)

    tube_map_shape = Dict{NTuple{2,Int}, Int}()
    expected_tube_map_shape_nnz = expected_nnz(sparsity(F), sparsity(F), (size_dict[:module_label_M], size_dict[:module_label_N]),  (size_dict[:fusion_label], size_dict[:multiplicity_label_M], size_dict[:multiplicity_label_N]) )
    sizehint!(tube_map_shape, expected_tube_map_shape_nnz )

    for (indx_M, val_M) in N_M
        M1, MN, Y = indx_M.I  

        for (indx_N, val_N) in N_M
            #N1, MN_, Y_ = indx_N.I
            MN_, N1, Y_ = indx_N.I

            linear_index = 1 
            if Y == Y_ && MN == MN_
                @inbounds for m in 1:val_M
                    @inbounds for n in 1:val_N
                        
                        key = (M1, N1, Y, MN, m, n)
                        tube_map_shape[(M1, N1)] = get(tube_map_shape, (M1, N1), 0) + 1 # linear_index
                        
                        key_inv = (M1, N1, tube_map_shape[(M1, N1)])
                        tube_map_inv[key_inv] = (Y, MN, m, n)

                        tube_map[key] = tube_map_shape[(M1, N1)] 
                        
                    end
                end
            end
        end
    end
    
    return tube_map, tube_map_shape, tube_map_inv
end  


function create_f_ijk_sparse(F::SparseArray{ComplexF64, 10}, F_quantum_dims::Vector{Float64}, size_dict::Dict{Symbol, Int}, 
                           tubes_ij, tube_map_shape)
    cache = Dict{Tuple{Int,Int,Int}, SparseArray}()
    sizehint!(cache, size_dict[:module_label_M]^3 * size_dict[:module_label_N]^3 )

    #function f_ijk_sparse(i::Int, j::Int, k::Int)
    function f_ijk_sparse(A::Int, Ap::Int, App::Int)

        key = (A,Ap,App)
        if haskey(cache, key)
            return cache[key]
        end

        # --- Contractions ---
        #=
        F_App[App, X2, X1, X3p, X2p, X3, k2, j2, j1, k3] # double
        F_Ap[X2, Ap, A, X3p, X2p, X1p, k2p, j2, k1, j3]
        F_A[X2, X1, A, X3p, X3, X1p, j1, k3p, k1p, j3] # double
        # Double everything except js

        F_App[X2, X2_, X1, X1_, X3p, X3p_, X2p, X2p_, X3, X3_, k2, j2, j1, k3] * 
        F_ApA[X2_, X2__, X3p_, X3p__, X2p_, X1p, X1p_, k2p, j2, k1, j3] * 
        F_A[X2__, X1_, X3p__, X3_, X1p_, j1, k3p, k1p, j3] 
        f_ijk_abc[X1, X1p, k1, k1p, X2, X2p, k2, k2p, X3, X3p, k3, k3p] 
        =#

        F_App = F[App,:,:,:,:,:,:,:,:,:]
        conj!(F_App)

        F_ApA = F[:,Ap,A,:,:,:,:,:,:,:]

        F_A = F[:,:,A,:,:,:,:,:,:,:]
        conj!(F_A)

        F_App = reindexdims(F_App, (1,1,2,2,3,3,4,4,5,5,6,7,8,9))
        F_ApA = reindexdims(F_ApA, (1,1,2,2,3,4,4,5,6,7,8))

        @tensor f_ijk_abc[X1, X1p, k1, k1p, X2, X2p, k2, k2p, X3, X3p, k3, k3p] := F_App[X2, X2_, X1, X1_, X3p, X3p_, X2p, X2p_, X3, X3_, k2, j2, j1, k3] * F_ApA[X2_, X2__, X3p_, X3p__, X2p_, X1p, X1p_, k2p, j2, k1, j3] * F_A[X2__, X1_, X3p__, X3_, X1p_, j1, k3p, k1p, j3] 
        dropnearzeros!(f_ijk_abc; tol = 1e-10)
        @show nonzero_keys(f_ijk_abc)

        f_abc_DOK = sizehint!(Dict{CartesianIndex{3}, ComplexF64}(), length(nonzero_keys(f_ijk_abc)))

        for (CI, val) in nonzero_pairs(f_ijk_abc)
            X1, X1p, k1, k1p,   X2, X2p, k2, k2p,   X3, X3p, k3, k3p = Tuple(CI)
            
            idx_a = tubes_ij[(A, Ap, X1p, X1, k1, k1p)]
            idx_b = tubes_ij[(App, Ap, X2p, X2, k2p, k2)]
            idx_c = tubes_ij[(A, App, X3p, X3, k3, k3p)]
            
            f_abc_DOK[CartesianIndex(idx_a, idx_b, idx_c)] = val
        end

        shape = (tube_map_shape[(A, Ap)], tube_map_shape[(Ap, App)], tube_map_shape[(A, App)])
        reindexed_f_symbol = SparseArray{ComplexF64,3}(f_abc_DOK, shape)

        cache[key] = reindexed_f_symbol
        return reindexed_f_symbol
    end

    return f_ijk_sparse
end


function is_associative(f, tol = 1e-9)
    @tensor test[i,j,l,m] := f[i,j,k] * f[k,l,m] - f[j,l,k] * f[i,k,m]

    max_violation = maximum(abs.(test))

    return (max_violation <= tol), max_violation
end

function create_dim_dict(size_dict, tube_map_shape)
    cache = Dict{Tuple{Int,Int,Int}, Tuple{Int,Int,Int}}()
    sizehint!(cache, (size_dict[:module_label_N]*size_dict[:module_label_M])^3)

    function dim_ijk(i,j,k)
        key = (i,j,k)
        if haskey(cache, key)
            return cache[key]
        end

        M_1, N_1 = index_to_tuple(i, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_2, N_2 = index_to_tuple(j, (size_dict[:module_label_N], size_dict[:module_label_M]))
        
        size_a = get(tube_map_shape, (M_2, M_1, N_1, N_2), 0)
        if size_a == 0
            return 
        end
        M_3, N_3 = index_to_tuple(k, (size_dict[:module_label_N], size_dict[:module_label_M]))
        size_b = get(tube_map_shape, (M_3, M_2, N_2, N_3), 0)
        if size_b == 0
            return
        end
        size_c = get(tube_map_shape, (M_3, M_1, N_1, N_3), 0)
        if size_c == 0
            return
        end
        
        cache[key] = (size_a, size_b, size_c)
        return (size_a, size_b, size_c)
    end
end

end # Module