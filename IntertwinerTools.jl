module IntertwinerTools

export is_associative, create_fusion_rules, create_tube_map, create_f_ijk_sparse, create_dim_dict, construct_irreps, sparse_clebsch_gordon_coefficients

using Base.Threads
using SparseArrayKit
using LinearAlgebra
using TensorOperations
using StaticArrays: SVector, @SVector
using Arpack # For eigenvalue solver in constructing CG from P_abc

include("F_symbolTools.jl")
using .FSymbolTools    
 
# --- Tuple <-> Index ---
tuple_to_index(tup::NTuple, shape::NTuple) = LinearIndices(shape)[tup...]
index_to_tuple(idx::Int, shape::NTuple) = Tuple(CartesianIndices(shape)[idx])


function create_fusion_rules(F)
    #hom_space =F[M2, Y1, Y, M1, M2, Y, 1, :, 1, :]

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

function create_tube_map(N_M, N_N, size_dict)
    tube_map = Dict{NTuple{7,Int}, Int}()
    sizehint!(tube_map, size_dict[:module_label_M]^2 * size_dict[:module_label_N]^2 * size_dict[:fusion_label])

    tube_map_inv = Dict{NTuple{5,Int}, NTuple{3,Int}}()
    sizehint!(tube_map_inv, size_dict[:module_label_M]^2 * size_dict[:module_label_N]^2 * size_dict[:fusion_label])

    tube_map_shape = Dict{NTuple{4,Int}, Int}()
    sizehint!(tube_map_shape, size_dict[:module_label_M]^2 * size_dict[:module_label_N]^2 )

    # M1, M2, N1, N2, Y, m, n, 
    
    for (indx_M, val_M) in N_M
        M1, M2, Y = indx_M.I  
        
        for (indx_N, val_N) in N_N
            N1, N2, YN = indx_N.I
            
            
            if Y == YN
                
                @inbounds for m in 1:val_M
                    @inbounds for n in 1:val_N
                        key = (M1, M2, N2, N1, Y, m, n)
                        tube_map_shape[(M1, M2, N2, N1)] = get(tube_map_shape, (M1, M2, N2, N1), 0) + 1 # linear_index
                        
                        key_inv = (M1, M2, N2, N1, tube_map_shape[(M1, M2, N2, N1)])
                        tube_map_inv[key_inv] = (Y, m, n)

                        tube_map[key] = tube_map_shape[(M1, M2, N2, N1)] 
                        #=
                        key = (M2, M1, N1, N2, Y, n, m)
                        tube_map_shape[(M2, M1, N1, N2)] = get(tube_map_shape, (M2, M1, N1, N2), 0) + 1 # linear_index
                        
                        key_inv = (M2, M1, N1, N2, tube_map_shape[(M2, M1, N1, N2)])
                        tube_map_inv[key_inv] = (Y, n, m)

                        tube_map[key] = tube_map_shape[(M2, M1, N1, N2)] 
                        =#
                        #=
                        key = (M1, M2, N1, N2, Y, m, n)
                        tube_map_shape[(M1, M2, N1, N2)] = get(tube_map_shape, (M1, M2, N1, N21), 0) + 1 # linear_index
                        
                        key_inv = (M1, M2, N1, N2, tube_map_shape[(M1, M2, N1, N2)])
                        tube_map_inv[key_inv] = (Y, m, n)

                        tube_map[key] = tube_map_shape[(M1, M2, N1, N2)] 
                        =#
                    end
                end
            end
        end
    end
    
    return tube_map, tube_map_shape, tube_map_inv
end


function create_f_ijk_sparse(F_M::SparseArray{ComplexF64, 10}, F_N::SparseArray{ComplexF64, 10}, 
                           F_quantum_dims::Vector{Float64}, size_dict::Dict{Symbol, Int}, 
                           tubes_ij, tube_map_shape, N_M, N_N)
    cache = Dict{Tuple{Int,Int,Int}, SparseArray}()
    sizehint!(cache, size_dict[:module_label_M]^3 * size_dict[:module_label_N]^3 )

    MN_to_a_map = CartesianIndices((size_dict[:module_label_M], size_dict[:module_label_N]))

    function f_ijk_sparse(i::Int, j::Int, k::Int)
        key = (i,j,k)
        #println("i,j,k: $((key))")
        if haskey(cache, key)
            return cache[key]
        end

        # --- Decode flattened indices ---
        M_1, N_1 = Tuple(MN_to_a_map[i])
        M_2, N_2 = Tuple(MN_to_a_map[j])
        M_3, N_3 = Tuple(MN_to_a_map[k])

        # --- Slice tensors ---
        F_N_slice = slice_sparse_tensor(F_N, Dict(1=>N_1, 4=>N_3, 5=>N_2))
        F_M_slice = slice_sparse_tensor(F_M, Dict(1=>M_1, 4=>M_3, 5=>M_2))

        # --- Quantum dimension prefactors ---
        sqrtd = sqrt.(F_quantum_dims)
        #sqrtd = F_quantum_dims.^(1/4)
        dY1 = SparseArray(sqrtd)
        dY2 = SparseArray(sqrtd)
        dY3 = SparseArray(1.0 ./ sqrtd)

        # --- Contractions ---
        F_N_sliced_doubled = reindexdims(F_N_slice, (1,1,2,2,3,3,4,5,6,7))
        F_M_sliced_doubled = conj!(reindexdims(F_M_slice, (1,1,2,2,3,3,4,5,6,7)))
        
        @tensor dxdxpdy_F_M_dot_F_N[y,p,x,r,s,n,a,m,b] := F_N_sliced_doubled[y_, y__, p_, p__,x_, x__, r,s,l,n ] * F_M_sliced_doubled[y, y_,p, p_, x, x_,a,m,l,b ] * dY1[y__] * dY2[p__] * dY3[x__]
        
        dropnearzeros!(dxdxpdy_F_M_dot_F_N; tol = 1e-10)

        keys_iterator = nonzero_keys(dxdxpdy_F_M_dot_F_N)
        vals = collect(nonzero_values(dxdxpdy_F_M_dot_F_N))

        f_abc_DOK = sizehint!(Dict{CartesianIndex{3}, ComplexF64}(), length(keys_iterator))

        for (CI, val) in zip(keys_iterator, vals)
            Y1, Y2, Y3, n1, n2, n4, m1, m2, m3 = Tuple(CI)
            
            idx_a = tubes_ij[(M_2, M_1, N_1, N_2, Y1, m1, n1)]
            idx_b = tubes_ij[(M_3, M_2, N_2, N_3, Y2, m2, n2)]
            idx_c = tubes_ij[(M_3, M_1, N_1, N_3, Y3, m3, n4)]
            
            f_abc_DOK[CartesianIndex(idx_a, idx_b, idx_c)] = val
        end

        shape = (tube_map_shape[(M_2, M_1, N_1, N_2)], tube_map_shape[(M_3, M_2, N_2, N_3)], tube_map_shape[(M_3, M_1, N_1, N_3)])
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

function create_dim_dict(size_dict::Dict{Symbol, Int}, tubes_ij, tube_map_shape, N_M, N_N)
    cache = Dict{Tuple{Int,Int,Int}, Tuple{Int,Int,Int}}()

    MN_to_a_map = CartesianIndices((size_dict[:module_label_M], size_dict[:module_label_N]))

    function dim_ijk(i,j,k)
        key = (i,j,k)
        if haskey(cache, key)
            return cache[key]
        end

        M_1, N_1 = Tuple(MN_to_a_map[i])
        M_2, N_2 = Tuple(MN_to_a_map[j])
        
        size_a = get(tube_map_shape, (M_2, M_1, N_1, N_2), 0)
        if size_a == 0
            return 
        end
        M_3, N_3 = Tuple(MN_to_a_map[j])
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

# ----------------------------------------
# - Post Processing
# ----------------------------------------

using LinearAlgebra
using StaticArrays

function construct_irreps( algebra, irrep_projectors, size_dict, tube_map_inv::Dict{NTuple{5,Int},NTuple{3,Int}}, q_dim_a::AbstractVector,create_left_ijk_basis, F,)
  
    N_irrep  = length(irrep_projectors)
    N_blocks = algebra.N_diag_blocks

    mod_N = size_dict[:module_label_N]
    mod_M = size_dict[:module_label_M]

    MN_to_a_map = CartesianIndices((size_dict[:module_label_M], size_dict[:module_label_N]))
    

    keys_vec = Vector{SVector{10,Int}}()
    vals_vec = Vector{ComplexF64}()
    sizehint!(keys_vec, length(F.data))
    sizehint!(vals_vec, length(F.data))

    for irrep in 1:N_irrep
        println("# -- Irrep $(irrep) -- #")
        iproj = irrep_projectors[irrep]
        k = first(keys(iproj))[2]

        for i in 1:N_blocks
            haskey(iproj, (i,k)) || continue
            Q_ik = iproj[(i,k)]
            adj_Q_ik = adjoint(Q_ik)

            M1, N1 = Tuple(MN_to_a_map[i])

            for j in 1:N_blocks
                algebra.dim_ijk(i,j,k) === nothing && continue
                haskey(iproj, (j,k)) || continue
                Q_jk = iproj[(j,k)]

                M2, N2 = Tuple(MN_to_a_map[j])

                T_a = create_left_ijk_basis(algebra, i, j, k).basis

                for a in eachindex(T_a)
                    Y, m, n = tube_map_inv[(M1, M2, N2, N1, a)]
                    #@show  size(Q_jk), size(T_a[a]), size(adjoint(Q_ik))
                    #=
                    @show (i,k), (i,j,k), (j,k)
                    @show  size(adj_Q_ik), size(T_a[a]), size(Q_jk)
                    =#
                    #ρ = adj_Q_ik * T_a[a] * Q_jk * sqrt(q_dim_a[N2])/(sqrt(q_dim_a[N1])*sqrt(q_dim_a[Y])) #/ sqrt(q_dim_a[Y]) #
                    ρ = adj_Q_ik * T_a[a] * Q_jk * sqrt(q_dim_a[N1])/(sqrt(q_dim_a[N2])*sqrt(q_dim_a[Y])) #/ sqrt(q_dim_a[Y]) #

                    #ρ = adjoint(adj_Q_ik * T_a[a] * Q_jk) * sqrt(q_dim_a[N2])/(sqrt(q_dim_a[N1])*sqrt(q_dim_a[Y]))
                    #ρ = Q_jk * T_a[a] * adjoint(Q_ik) #* sqrt(q_dim_a[N2])/(sqrt(q_dim_a[N1])*sqrt(q_dim_a[Y]))
                    
                    #/ sqrt(q_dim_a[Y]) # * sqrt(q_dim_a[N1])/sqrt(q_dim_a[N2])
                    
                    @inbounds for col in axes(ρ,2), row in axes(ρ,1)
                        val = ρ[row,col]
                        iszero(val) && continue
                        #F[M2, Y1, Y, M1, M2, Y, 1, :, 1, :]
                        #key = SVector{10,Int}( irrep, M1, Y, N2, N1, M2, row, n, m, col)
                        #key = SVector{10,Int}( irrep, M1, Y, N2, N1, M2, row, n, m, col)
                        #key = SVector{10,Int}( irrep, N1, Y, M2, M1, N2, row, n, m, col)
                        #key = SVector{10,Int}( irrep, M2, Y, N1, N2, M1, row, n, m, col) # this is wokring for vecG over vecG
                        #key = SVector{10,Int}( irrep, M1, Y, N2, N1, M2, row, m, n, col)
                        key = SVector{10,Int}( irrep, M1, Y, N2, N1, M2, m, row, col, n)

                        
                        #key = SVector{10,Int}( Y, M1, irrep, N2, N1, M2, col, n, m, row)

                        #key = SVector{10,Int}( Y, M1, irrep, N2, N1, M2, m, col, row, n)
                        
                        # Y, M1, irrep, N2, N1, M2, row, m, n, col
                        # 'M4', 'Y1', 'Y2', 'M5', 'M6', 'Y3', 's', 't', 'l', 'j'
                        # irrep, M1, Y, N2, N1, M2, row, m, n, col
                        #=
                        key = SVector{10,Int}(
                             M2, irrep, Y, N2, N1, M1, row, n, m, col
                        )=#

                        #=
                        new_coords = np.vstack([np.full(N_nz, irrep),
                                            np.full(N_nz, M_1),
                                            np.full(N_nz, Y),
                                            np.full(N_nz, N_2),
                                            np.full(N_nz, N_1),
                                            np.full(N_nz, M_2),
                                            rows,
                                            np.full(N_nz, n),
                                            np.full(N_nz, m),
                                            cols])
                        =#
                        push!(keys_vec, key)
                        push!(vals_vec, val)
                    end
                end
            end
        end
    end

    maxvals = zeros(Int, 10)
    @inbounds for k in keys_vec
        for d in 1:10
            maxvals[d] = max(maxvals[d], k[d])
        end
    end
    shape = Tuple(maxvals)
    tol = 1e-10
    dok = Dict(
        CartesianIndex(Tuple(k)) => v
        for (k,v) in zip(keys_vec, vals_vec) if abs(v)> tol
    )
    
    return SparseArray{ComplexF64,10}(dok, shape)
end


FP_dimension(d_quantum) = sum(d_quantum .^ 2)

function sparse_clebsch_gordon_coefficients(ω, fusion_cat_quantum_dims)
    X_dim = size(ω, 1)
    FP_dim = FP_dimension(fusion_cat_quantum_dims)
    
    U_keys_vec = Vector{SVector{10,Int}}()
    U_vals_vec = Vector{ComplexF64}()

    cart_ind_map = CartesianIndices(old_shape[1:half]) # reshape row index into N, i_a, M, i_b, O, i_c,

    for a in 1:X_dim
        ω_a_doubled = reindexdims(ω[a, ntuple(_ -> :, ndims(ω)-1)...], (1,1,2,2,3,4,5,5,6,7,8,9)) # [M1, Y, N2, N1, M2, i_a, k_a, l_a, j_a] 

        # Double ω: M1, M2, Y (1,2,5)
        for b in 1:X_dim
            ω_b = ω[b, ntuple(_ -> :, ndims(ω)-1)...] #[O1, Y, M2, M1, O2, i_b, k_b, l_b, j_b] 
            @tensor P_ab[M1, O1, Y, N2, N1, M2, O2, i_a, i_b, k_a, l_b, j_a, j_b] := ω_a_doubled[M1, M1_, Y, Y_, N2, N1, M2, M2_, i_a, k_a, l_a, j_a] * ω_b[O1, Y_, M2_, M1_, O2, i_b, l_a, l_b, j_b] #[M1, O1, Y, N2, N1, M2, O2, i_a, i_b, k_a, l_b, j_a, j_b] 
            P_ab_doubled = reindexdims(P_ab, (1,2,2,3,4,4,5,5,6,7,7,8,9,10,11,12,13)) #[M1, O1,O1_, Y,Y_, N2,N2_, N1,N1_, M2, O2.O2_, i_a, i_b, k_a, l_b, j_a, j_b]

            # Double P_ab: O1,O2,N1,N2
            for c in 1:X_dim
                ω_c = ω[c, ntuple(_ -> :, ndims(ω)-1)...] #[N1, Y, O2, O1, N2, i_c, l_c, k_c, j_c] 
                conj!(ω_c)
                @tensor P_abc[N1, i_a, M1, i_b, O1, i_c, N2, j_a, M2, j_b, O2, j_c] := P_ab_doubled[M1, O1,O1_, Y, N2,N2_, N1,N1_, M2, O2,O2_, i_a, i_b, l_c, k_c, j_a, j_b] * ω_c[N2_, Y, O1_, O2_, N1_, i_c, l_c, k_c, j_c]
                P_abc ./= FP_dim
                @tensor trace_P_abc[] := P_abc[N, i_a, M, i_b, O, i_c, N, i_a, M, i_b, O, i_c]
                
                N_eigs = round(Int, abs(trace_P_abc[]))
                if abs(N_eigs) > 1e-10
                    #println("abc = ($a,$b,$c) has N_eigs = $N_eigs")
                    old_shape = size(P_abc)
                    half = length(old_shape)/2
                    dL = prod(old_shape[1:half])
                    dR = prod(old_shape[half+1:end])
                    P_mat = reshape(P_abc, dL, dR)
                    eigvals, eigvecs = eigs(P_mat; nev=N_eigs)

                    
                    @inbounds for col_index in 1:size(eigvecs,2)
                        for row_index in 1:size(eigvecs,1)
                            key = SVector{10,Int}(a,b,c,Tuple(cart_ind_map[row_index])..., col_index)
                            val = eigvecs[row_index,col_index]
                            push!(U_keys_vec, key)
                            push!(U_vals_vec, val)
                        end
                    end
                    #U_abc[(a,b,c)] = reshape(eigvecs, (old_shape[1:half]..., size(eigvecs,2))) # True sparsity pattern of F_symbol - sparse in a,b,c, eig_index but not the middle 6 indices
                end
            end
        end
    end

    U_DOK = Dict( CartesianIndex(Tuple(k)) => v for (k,v) in zip(U_keys_vec, U_vals_vec) )

    maxvals = zeros(Int, 10)
    @inbounds for k in U_keys_vec
        for d in 1:10
            maxvals[d] = max(maxvals[d], k[d])
        end
    end
    
    U = SparseArray{ComplexF64,10}(U_DOK, Tuple(maxvals))
    return U
end


end # Module