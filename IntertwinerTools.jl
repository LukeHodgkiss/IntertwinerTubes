module IntertwinerTools

export make_fusion_rules, make_tubes_ij, make_f_ijk_sparse, compute_dim_dict, is_associative, make_dim_dict, sparse_clebsch_gordon_coefficients, create_f_ijk_sparse, create_dim_dict #, construct_irreps

using Base.Threads
using SparseArrayKit
using LinearAlgebra
using TensorOperations
using StaticArrays: SVector, @SVector
#using StaticArrays

# Importing Locally
include("F_symbolTools.jl")
using .FSymbolTools    
 
# --- Tuple <-> Index ---
tuple_to_index(tup::NTuple, shape::NTuple) = LinearIndices(shape)[tup...]
index_to_tuple(idx::Int, shape::NTuple) = Tuple(CartesianIndices(shape)[idx])

function remove_zeros(sparse_tensor::SparseArray; tol=1e-10)
    mask = abs.(sparse_tensor.nzval) .>= tol
    rows, cols = findnz(sparse_tensor)[1:2]
    return sparse(rows[mask], cols[mask], sparse_tensor.nzval[mask], size(sparse_tensor,1), size(sparse_tensor,2))
end


export make_fusion_rules

# --- Fusion Rules Factory ---

function make_fusion_rules(F::SparseArray, size_dict::Dict)
    N_fusion_elements = size_dict[:fusion_label]
    cache = Dict{Tuple{Int,Int}, Dict{Int, Vector{Int}}}()
    function fusion_rules(M2::Int, M1::Int; tol=1e-10)
        
        key = (M2, M1)
        if haskey(cache, key)
            return cache[key]
        end
            
        nonzero_M2M1 = Dict{Int, Vector{Int}}()
        Y1 = 1

        for Y in 1:N_fusion_elements
            hom_space = slice_sparse_tensor(F, Dict(1=>M2, 2=>Y1, 3=>Y, 4=>M1, 5=>M2, 6=>Y, 7=>1, 9=>1)) #F[M2, Y1, Y, M1, M2, Y, 1, :, 1, :]
                        
            # iterate nonzeros 
            for (idx, val) in nonzero_pairs(hom_space)
                if abs(val) > tol
                    alpha = idx[1]   
                    #alpha = idx[2]   
                    push!(get!(nonzero_M2M1, Y, Int[]), alpha)
                end
            end
        end
        cache[key] = nonzero_M2M1
        return nonzero_M2M1
    end
    return fusion_rules
end



# --- Tubes Factory ---
function make_tubes_ij(fusion_rules_M, fusion_rules_N)
    cache = Dict{Tuple{Int,Int,Int,Int}, Dict{Tuple{Int, Int, Int}, Int}}()

    function tubes_ij(N2::Int, N1::Int, M1::Int, M2::Int)
        key = (N2, N1, M1, M2)
        if haskey(cache, key)
            return cache[key]
        end

        nonzero_N = fusion_rules_N(N1, N2)
        nonzero_M = fusion_rules_M(M1, M2)

        joined = Dict{Tuple{Int, Int, Int}, Int}() # Y, α, β:  linear_index
        linear_index = 1

        common_Y = sort(collect(intersect(keys(nonzero_N), keys(nonzero_M))))

        for Y in common_Y
            vals_N = sort(nonzero_N[Y]) # Multiplicity inxed associated to N-module fusion
            vals_M = sort(nonzero_M[Y]) # Multiplicity inxed associated to M-module fusion

            # Iterate over all pairs in lexicographic order (ensured by sorting)
            for (a, b) in Iterators.product(vals_N, vals_M)
                joined[(Y, a, b)] = linear_index
                linear_index += 1
            end
        end

        cache[key] = joined
        return joined
    end

    return tubes_ij
end

function dropnearzeros!(A::SparseArray; tol = 1e-12)
    
    for (I, v) in collect(A.data)   
        if abs(v) ≤ tol
            delete!(A.data, I)
        end
    end
    return A
end

# --- Factory for f_ijk_sparse ---
function make_f_ijk_sparse(F_N::SparseArray{ComplexF64, 10}, F_M::SparseArray{ComplexF64, 10}, 
                           F_quantum_dims::Vector{Float64}, size_dict::Dict{Symbol, Int}, 
                           tubes_ij)
    cache = Dict{Tuple{Int,Int,Int}, SparseArray}()

    function f_ijk_sparse(i::Int, j::Int, k::Int)
        key = (i,j,k)
        #println("i,j,k: $((key))")
        if haskey(cache, key)
            return cache[key]
        end

        # --- Decode flattened indices ---
        M_1, N_1 = index_to_tuple(i, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_2, N_2 = index_to_tuple(j, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_3, N_3 = index_to_tuple(k, (size_dict[:module_label_N], size_dict[:module_label_M]))
        #println("M_1, N_1 = $M_1, $N_1")
        #println("M_2, N_2 = $M_2, $N_2")
        #println("M_3, N_3 = $M_3, $N_3")

        # --- Slice tensors ---
        F_N_slice = slice_sparse_tensor(F_N, Dict(1=>N_1, 4=>N_3, 5=>N_2))
        F_M_slice = slice_sparse_tensor(F_M, Dict(1=>M_1, 4=>M_3, 5=>M_2))

        # --- Quantum dimension prefactors ---
        sqrtd = sqrt.(F_quantum_dims)
        dY1 = SparseArray(sqrtd)
        dY2 = SparseArray(sqrtd)
        dY3 = SparseArray(1.0 ./ sqrtd)

        # --- Contractions ---
        F_N_sliced_doubled = reindexdims(F_N_slice, (1,1,2,2,3,3,4,5,6,7))
        F_M_sliced_doubled = conj!(reindexdims(F_M_slice, (1,1,2,2,3,3,4,5,6,7)))

        """
        println("F_quantum_dims:  $(F_quantum_dims)")

        println("F_N $( (size(F_N)) )")
        println("F_N_sliced $( (size(F_N_slice)) )")
        println("F_N_sliced_doubled $( (size(F_N_sliced_doubled)) )")


        println("F_M $( (size(F_M)) )")
        println("F_M_sliced $( (size(F_M_slice)) )")
        println("F_M_sliced_doubled $( (size(F_M_sliced_doubled)) )")
        """
        #@tensor F_N_sliced_doubled_scaled[y_, p_, x_,r,s,l,n ] := F_N_sliced_doubled[y_, y__, p_, p__,x_, x__, r,s,l,n ] * dY1[y__] * dY2[p__] * dY3[x__]
        #@tensor dxdxpdy_F_M_dot_F_N[y,p,x,r,s,n,a,m,b] := F_N_sliced_doubled_scaled[y_, p_, x_, r,s,l,n ] * F_M_sliced_doubled[y, y_,p, p_, x, x_,a,m,b,l ]
        @tensor dxdxpdy_F_M_dot_F_N[y,p,x,r,s,n,a,m,b] := F_N_sliced_doubled[y_, y__, p_, p__,x_, x__, r,s,l,n ] * F_M_sliced_doubled[y, y_,p, p_, x, x_,a,m,l,b ] * dY1[y__] * dY2[p__] * dY3[x__]
        dropnearzeros!(dxdxpdy_F_M_dot_F_N; tol = 1e-10)

        # --- Remove small values ---
        #dxdxpdy_F_M_dot_F_N = remove_zeros!(dxdxpdy_F_M_dot_F_N)
        #println("dxdxpdy_F_M_dot_F_N has hsape: $(size(dxdxpdy_F_M_dot_F_N))")
        # Convert linear indices to Cartesian indices
        keys_array = collect(nonzero_keys(dxdxpdy_F_M_dot_F_N))                     
        nnz = length(keys_array)
        N_axes = ndims(dxdxpdy_F_M_dot_F_N)
        #println("NNZ=$(nnz)")
        #println("N_axes=$(N_axes)")
        
        coords_matrix = Array{Int,2}(undef, N_axes, nnz)
        for (col, CI) in enumerate(keys_array)
            @inbounds coords_matrix[:, col] .= Tuple(CI)
        end
        
        Y1, Y2, Y3, n1, n2, n4, m1, m2, m3  = ntuple(d -> coords_matrix[d, :], N_axes)
        #Y1, Y2, Y3, n1, m3, n4, m1, m2, n2  = ntuple(d -> coords_matrix[d, :], N_axes)
        #Y1, Y2, Y3, m1, m2, m3, n1, n2, n4  = ntuple(d -> coords_matrix[d, :], N_axes)
        """
        coords_matrix = reduce(hcat, keys_array)'
        coords_matrix = permutedims(reduce(hcat, keys_array))
        Y1, Y2, Y3, n1, n2, n4, m1, m2, m3 = (coords_matrix[ax, :] for ax in 1:10)
        """
        #print("Subalgebra: $((i,j,k))")
        #@show Y1 Y2 Y3 n1 n2 n4 m1 m2 m3

        vals = collect(nonzero_values(dxdxpdy_F_M_dot_F_N))
        
        # Get linear indices from tube maps
        idx_a_map = tubes_ij(M_2, M_1, N_1, N_2)
        idx_b_map = tubes_ij(M_3, M_2, N_2, N_3)
        idx_c_map = tubes_ij(M_3, M_1, N_1, N_3)
        
        """
        index_a = []
        for tup_a in zip(Y1, m1, n1)
            #println("tup_a: ((tup_a))")
            push!(index_a, idx_a_map[tup_a])
        end
        """

        index_a = [idx_a_map[tup_a] for tup_a in zip(Y1, m1, n1)]
        #print("idx_a_map= $(idx_a_map)")
        index_b = [idx_b_map[tup_b] for tup_b in zip(Y2, m2, n2)]
        #print("idx_b_map= $(idx_b_map)")
        index_c = [idx_c_map[tup_c] for tup_c in zip(Y3, m3, n4)]
        #print("idx_c_map= $(idx_c_map)")


        # --- Build new SparseArrayKit array with reindexed coordinates ---
        f_abc_DOK = Dict{CartesianIndex{3}, ComplexF64}()
        for idx in 1:nnz
            f_abc_DOK[CartesianIndex(index_a[idx], index_b[idx], index_c[idx])] = vals[idx]
        end

        shape = (length(idx_a_map), length(idx_b_map), length(idx_c_map))
        reindexed_f_symbol = SparseArray{ComplexF64,3}(f_abc_DOK, shape)

        dropnearzeros!(reindexed_f_symbol; tol = 1e-10)
        # Cache and return
        cache[key] = reindexed_f_symbol
        return reindexed_f_symbol
    end

    return f_ijk_sparse
end

function create_f_ijk_sparse(F_N::SparseArray{ComplexF64, 10}, F_M::SparseArray{ComplexF64, 10}, 
                           F_quantum_dims::Vector{Float64}, size_dict::Dict{Symbol, Int}, 
                           tubes_ij, tube_map_shape, N_M, N_N)
    cache = Dict{Tuple{Int,Int,Int}, SparseArray}()
    sizehint!(cache, size_dict[:module_label_M]^3 * size_dict[:module_label_N]^3 )

    function f_ijk_sparse(i::Int, j::Int, k::Int)
        key = (i,j,k)
        #println("i,j,k: $((key))")
        if haskey(cache, key)
            return cache[key]
        end

        # --- Decode flattened indices ---
        M_1, N_1 = index_to_tuple(i, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_2, N_2 = index_to_tuple(j, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_3, N_3 = index_to_tuple(k, (size_dict[:module_label_N], size_dict[:module_label_M]))
        #println("M_1, N_1 = $M_1, $N_1")
        #println("M_2, N_2 = $M_2, $N_2")
        #println("M_3, N_3 = $M_3, $N_3")

        # --- Slice tensors ---
        F_N_slice = slice_sparse_tensor(F_N, Dict(1=>N_1, 4=>N_3, 5=>N_2))
        F_M_slice = slice_sparse_tensor(F_M, Dict(1=>M_1, 4=>M_3, 5=>M_2))

        # --- Quantum dimension prefactors ---
        sqrtd = sqrt.(F_quantum_dims)
        dY1 = SparseArray(sqrtd)
        dY2 = SparseArray(sqrtd)
        dY3 = SparseArray(1.0 ./ sqrtd)

        # --- Contractions ---
        F_N_sliced_doubled = reindexdims(F_N_slice, (1,1,2,2,3,3,4,5,6,7))
        F_M_sliced_doubled = conj!(reindexdims(F_M_slice, (1,1,2,2,3,3,4,5,6,7)))

        """
        println("F_quantum_dims:  $(F_quantum_dims)")

        println("F_N $( (size(F_N)) )")
        println("F_N_sliced $( (size(F_N_slice)) )")
        println("F_N_sliced_doubled $( (size(F_N_sliced_doubled)) )")


        println("F_M $( (size(F_M)) )")
        println("F_M_sliced $( (size(F_M_slice)) )")
        println("F_M_sliced_doubled $( (size(F_M_sliced_doubled)) )")
        """
        #@tensor F_N_sliced_doubled_scaled[y_, p_, x_,r,s,l,n ] := F_N_sliced_doubled[y_, y__, p_, p__,x_, x__, r,s,l,n ] * dY1[y__] * dY2[p__] * dY3[x__]
        #@tensor dxdxpdy_F_M_dot_F_N[y,p,x,r,s,n,a,m,b] := F_N_sliced_doubled_scaled[y_, p_, x_, r,s,l,n ] * F_M_sliced_doubled[y, y_,p, p_, x, x_,a,m,b,l ]
        @tensor dxdxpdy_F_M_dot_F_N[y,p,x,r,s,n,a,m,b] := F_N_sliced_doubled[y_, y__, p_, p__,x_, x__, r,s,l,n ] * F_M_sliced_doubled[y, y_,p, p_, x, x_,a,m,l,b ] * dY1[y__] * dY2[p__] * dY3[x__]
        dropnearzeros!(dxdxpdy_F_M_dot_F_N; tol = 1e-10)

        # Convert linear indices to Cartesian indices
        #=
        keys_array = collect(nonzero_keys(dxdxpdy_F_M_dot_F_N))                     
        nnz = length(keys_array)
        N_axes = ndims(dxdxpdy_F_M_dot_F_N)
        
        coords_matrix = Array{Int,2}(undef, N_axes, nnz)
        for (col, CI) in enumerate(keys_array)
            @inbounds coords_matrix[:, col] .= Tuple(CI)
        end
        Y1, Y2, Y3, n1, n2, n4, m1, m2, m3  = ntuple(d -> coords_matrix[d, :], N_axes)
        =#
        #=
        N_axes = ndims(dxdxpdy_F_M_dot_F_N)
        Y1, Y2, Y3, n1, n2, n4, m1, m2, m3 = ntuple(d -> [Tuple(k)[d] for k in nonzero_keys(dxdxpdy_F_M_dot_F_N)], N_axes)
        nnz = length(Y1)
        =#
        #=
        N_axes = ndims(dxdxpdy_F_M_dot_F_N)
        keys_tuples = [Tuple(k) for k in nonzero_keys(dxdxpdy_F_M_dot_F_N)]
        Y1, Y2, Y3, n1, n2, n4, m1, m2, m3 = ntuple(d -> [k[d] for k in keys_tuples], N_axes)
        nnz = length(Y1)


        index_a = [tubes_ij[(M_2, M_1, N_1, N_2, Y1_, m1_, n1_)] for (Y1_, m1_, n1_) in zip(Y1, m1, n1)]
        index_b = [tubes_ij[(M_3, M_2, N_2, N_3, Y2_, m2_, n2_)] for (Y2_, m2_, n2_) in zip(Y2, m2, n2)]
        index_c = [tubes_ij[(M_3, M_1, N_1, N_3, Y3_, m3_, n4_)] for (Y3_, m3_, n4_) in zip(Y3, m3, n4)]

        f_abc_DOK = Dict{CartesianIndex{3}, ComplexF64}()
        sizehint!(f_abc_DOK, nnz )

        for idx in 1:nnz
            f_abc_DOK[CartesianIndex(index_a[idx], index_b[idx], index_c[idx])] = vals[idx]
        end

        vals = collect(nonzero_values(dxdxpdy_F_M_dot_F_N))
        =#

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

""" Tests assocaitivity of algebra given strcuture constants """
function is_associative(f, tol = 1e-9)
    @tensor test[i,j,l,m] := f[i,j,k] * f[k,l,m] - f[j,l,k] * f[i,k,m]

    max_violation = maximum(abs.(test))

    return (max_violation <= tol), max_violation
end

# --- Function that computes dimension of subalgebra ijk
function make_dim_dict(size_dict::Dict{Symbol, Int}, tubes_ij)
    cache = Dict{Tuple{Int,Int,Int}, Tuple{Int,Int,Int}}()

    function dim_ijk(i,j,k)
        key = (i,j,k)
        if haskey(cache, key)
            return cache[key]
        end

        N_Mcat = size_dict[:module_label_N]
        N_Ncat = size_dict[:module_label_M]
        M_1, N_1 = index_to_tuple(i, (N_Mcat, N_Ncat))
        M_2, N_2 = index_to_tuple(j, (size_dict[:module_label_N], size_dict[:module_label_M]))
        
        map_a = tubes_ij(M_2, M_1, N_1, N_2)
        size_a = length(map_a)
        if size_a == 0
            return 
        end
        M_3, N_3 = index_to_tuple(k, (size_dict[:module_label_N], size_dict[:module_label_M]))
        map_b = tubes_ij(M_3, M_2, N_2, N_3)
        size_b = length(map_b)
        if size_b == 0
            return
        end
        map_c = tubes_ij(M_3, M_1, N_1, N_3)
        size_c = length(map_c)
        if size_c == 0
            return
        end

        cache[key] = (size_a, size_b, size_c)
        return (size_a, size_b, size_c)
    end
end

function create_dim_dict(size_dict::Dict{Symbol, Int}, tubes_ij, tube_map_shape, N_M, N_N)
    cache = Dict{Tuple{Int,Int,Int}, Tuple{Int,Int,Int}}()

    function dim_ijk(i,j,k)
        key = (i,j,k)
        if haskey(cache, key)
            return cache[key]
        end

        N_Mcat = size_dict[:module_label_N]
        N_Ncat = size_dict[:module_label_M]
        M_1, N_1 = index_to_tuple(i, (N_Mcat, N_Ncat))
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


# --- Construct Dictionary Of Nonzero Blocks ---
function compute_dim_dict(size_dict::Dict{Symbol, Int}, tubes_ij)
    N_Mcat = @view size_dict[:module_label_N]
    N_Ncat = @view size_dict[:module_label_M]

    N_blocks = N_Mcat * N_Ncat
    dim_dict = Dict{Tuple{Int,Int,Int}, Tuple{Int,Int,Int}}()

    for i in 1:(N_blocks)
        M_1, N_1 = index_to_tuple(i, (N_Mcat, N_Ncat))
        for j in 1:(N_blocks)
            M_2, N_2 = index_to_tuple(j, (N_Mcat, N_Ncat))

            map_a = tubes_ij(M_2, M_1, N_1, N_2)
            size_a = length(map_a)
            if size_a == 0
                continue
            end

            for k in 1:(N_blocks)
                M_3, N_3 = index_to_tuple(k, (N_Mcat, N_Ncat))

                map_b = tubes_ij(M_3, M_2, N_2, N_3)
                size_b = length(map_b)
                if size_b == 0
                    continue
                end

                map_c = tubes_ij(M_3, M_1, N_1, N_3)
                size_c = length(map_c)
                if size_c == 0
                    continue
                end
                dim_dict[(i,j,k)] = (size_a, size_b, size_c)
            end
        end
    end

    return dim_dict
end

# ----------------------------------------
# - Post Processing
# ----------------------------------------

#using .SparseAlgebraObjects: TubeAlgebra, create_left_ijk_basis

# --- Construct module functor intertwiner
"""
Return a DOK (dict) mapping SVector{10,Int} => ComplexF64
for the irreducible representation tensor.
"""

#function construct_irreps(algebra::TubeAlgebra, irrep_projectors::Vector{Dict}, size::Dict, tubes_ij::Function, q_dim_a)
function construct_irreps(algebra, irrep_projectors, size, tubes_ij::Function, q_dim_a)
    dok = Dict{SVector{10,Int}, ComplexF64}()
    maxvals = zeros(Int, 10)   

    N_irrep = length(irrep_projectors)

    @inbounds for irrep in 1:N_irrep
        first_key = first(keys(irrep_projectors[irrep]))
        k = first_key[2]

        N_blocks = algebra.N_diag_blocks
        iproj = irrep_projectors[irrep]

        for i in 1:N_blocks
            for j in 1:N_blocks
                if !haskey(algebra.dimension_dict, (i,j,k)); continue; end
                if !(haskey(iproj,(i,k)) && haskey(iproj,(j,k))); continue; end

                Q_ik = iproj[(i,k)]
                Q_jk = iproj[(j,k)]

                T_a = create_left_ijk_basis(algebra, i,j,k).basis
                d_a = length(T_a)
                    
                M1, N1 = index_to_tuple(i, (size[:module_label_N], size[:module_label_M]))
                M2, N2 = index_to_tuple(j, (size[:module_label_N], size[:module_label_M]))

                # Cachethis inverse mapping??
                idx_a_map = tubes_ij(M2, M1, N1, N2)
                reverse_tubes_ij = Dict{Int, NTuple{3,Int}}()
                for (tup_key, lin_val) in idx_a_map
                    reverse_tubes_ij[lin_val] = tup_key
                end

                for a in 1:d_a
                    Y, m, n = reverse_tubes_ij[a]
                    scale = 1.0 / sqrt(Float64(q_dim_a[Y]))
                    ρ = scale * (adjoint(Q_ik) * T_a[a] * Q_jk)

                    nnz_indices = findall(!iszero, ρ)
                    vals = ρ[nnz_indices]

                    for t in eachindex(vals)
                        row, col = Tuple(nnz_indices[t])
                        key = SVector{10,Int}(
                            irrep, M1, Y, N2, N1, M2,
                            row, n, m, col
                        )
                        dok[key] = vals[t]

                        # --- update maxvals on the fly ---
                        @inbounds for d in 1:10
                            if key[d] > maxvals[d]
                                maxvals[d] = key[d]
                            end
                        end
                    end
                end
            end
        end
    end

    shape = Tuple(maxvals)

    dok_CI = Dict{CartesianIndex{10}, ComplexF64}()
    #@inbounds
    @inbounds for (k, v) in dok
        dok_CI[CartesianIndex(Tuple(k))] = v # Key k is an SVector - maybe change but supposedly very good for insertion
    end

    ω = SparseArray{ComplexF64,10}(dok_CI, shape) 
    return ω
end # construct_irreps

# Compute FP dimension
FP_dimension(d_quantum) = sum(d_quantum .^ 2)

# Computing CG coefficients
function sparse_clebsch_gordon_coefficients(ω, fusion_cat_quantum_dims)
    X_dim = size(ω, 1)
    FP_dim = FP_dimension(fusion_cat_quantum_dims)

    U_abc = Dict{NTuple{3,Int}, Array}()

    for a in 1:X_dim
        ω_a_doubled = reindexdims(ω[a, ntuple(_ -> :, ndims(ω)-1)...], (1,1,2,2,3,4,5,5,6,7,8,9))
        for b in 1:X_dim
            ω_b = ω[b, ntuple(_ -> :, ndims(ω)-1)...]  #@view ω[b, :]
            @tensor P_ab[ a,y,e,g,c, h,f,d, j,l, i,k,m ] := ω_a_doubled[a, a_,y, y_,e,g,c, c_, h,f,b,d] * ω_b[j,y_,c_,a_,l, i,b,k,m]
            for c in 1:X_dim
                #ω_c = conj.(@view ω[c, :])
                # Double this first? then slice in loop?
                ω_c_doubled = reindexdims( ω[c, ntuple(_ -> :, ndims(ω)-1)...] , (1,1,2,2,3,3,4,4,5,5,6,7,8,9)) 
                @tensor P_abc[g,e, a,c, j,l, h,d, i,m, s,n ] := P_ab[a,y,e,g,c, h,f,d, j,l, i,k,m] * ω_c_doubled[j,j_,y, y_,e,e_,g,g_,l, l_,s,f,k,n] *  fusion_cat_quantum_dims[y_]
                P_abc ./= FP_dim
                @tensor trace_P_abc[] := P_abc[g,g, a,a, j,j, h,h, i,i, s,s]

                N_eigs = round(Int, abs(trace_P_abc))

                if abs(N_eigs) > 1e-10
                    #println("abc = ($a,$b,$c) has N_eigs = $N_eigs")

                    perm = (1,7,3,9,5,11, 2,8,4,10,6,12)
                    P_perm = permutedims(P_abc, perm)

                    old_shape = size(P_perm)
                    half = length(old_shape) ÷ 2
                    dL = prod(old_shape[1:half])
                    dR = prod(old_shape[half+1:end])

                    # Reshape into 2D matrix
                    P_mat = reshape(P_perm, dL, dR)
                    eigvals, eigvecs = eigs(P_mat; nev=N_eigs)
                    reshaped = reshape(eigvecs, old_shape[1:half]..., size(eigvecs,2))

                    U_abc[(a,b,c)] = reshaped
                end
            end
        end
    end

    return U_abc
end


"""

tic;
labels = cell(nM1*nM2,nM1*nM2);
f = cell(nM1*nM2,nM1*nM2,nM1*nM2);
flip = cell(nM1*nM2,nM1*nM2);
for i = 1:nM1*nM2
    [m1,n1] = ind2sub([nM1,nM2],i);
    temp = ArrayContractino({peps1{m1,m1,m1},peps2{n1,n1,n1}},{[-1 -3 -5 1],[-2 -4 -6 1]});
    f{i,i,i} = full(reduce(groupind(temp,[2 2 2])));
    for j = 1:nM1*nM2
        [m2,n2] = ind2sub([nM1,nM2],j);
        temp = groupind(ArrayContractino({flip1{m1,m2},flip2{n1,n2}},{[-1 -3 1],[-2 -4 1]}),[2 2]);
        if nnz(temp) ~= 0
            ind_temp = temp.ind;
            labels{i,j} = unique(ind_temp(:,1));
            flip{i,j} = full(reduce(temp));
        end
    end
end
fprintf('Computed diagonal structure factors and dagger structure in %f seconds\n',toc);

"""
# Do I need the full dim_dict?
# Comp

end # Module