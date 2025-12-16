module IntertwinerTools

export make_fusion_rules, make_tubes_ij, make_f_ijk_sparse, compute_dim_dict, make_dim_dict, sparse_clebsch_gordon_coefficients #, construct_irreps

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

function remove_zeros(sparse_tensor::SparseArray; tol=1e-9)
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

function make_fusion_rules_(F::SparseArray, size_dict::Dict)
    N_fusion_elements = size_dict[:fusion_label]

    #cache = Dict{Tuple{Int,Int}, Dict{Int,Vector{Int}}}()
    cache = Dict{SVector{Int,2}, Dict{Int,Int}}()


    function fusion_rules(M2::Int, M1::Int; tol = 1e-10)
        #key = (M2, M1)
        key = @SVector[M2, M1]


        if haskey(cache, key)
            return cache[key]
        end

        # Preallocate result container
        #nonzero_M2M1 = Dict{Int,Vector{Int}}()
        #nonzero_M2M1 = Dict{Int,Vector{Int}}()

        nonzero_M2M1 = Dict{Int,Int}()
        nonzero_M2M1 = Dict{Int,Int}()

        for Y in 1:N_fusion_elements
            #hom_space_view = @view F[M2, 1,Y, M1, M2, Y, 1,:,1,:]
            hom_space_view = SparseSliceView(F, Dict(1=>M2, 2=>1, 3=>Y, 4=>M1, 5=>M2, 6=>Y, 7=>1, 9=>1))
            
            for k in keys(hom_space_view)             
                #kt = Tuple(k)   
                #println(k)         
                val = hom_space_view[Tuple(k)...]  

                if abs(val) > tol
                    @show alpha = k[1]
                    push!(get!(nonzero_M2M1, Y, Int[]), alpha)
                end
            end
        end

        cache[key] = nonzero_M2M1
        return nonzero_M2M1
    end

    return fusion_rules
end

# --- Tubes Factory ---function make_tubes_ij(fusion_rules_M, fusion_rules_N)
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

        """for Y in common_keys
            vals_N = nonzero_N[Y]
            vals_M = nonzero_M[Y]

            for (a, b) in Iterators.product(vals_N, vals_M)
                joined[(Y, a, b)] = linear_index
                linear_index += 1
            end
        end  """

        cache[key] = joined
        return joined
    end

    return tubes_ij
end

# WIP :()
function make_tubes_ij_(fusion_rules_M, fusion_rules_N)
    cache = Dict{SVector{4,Int}, Dict{SVector{3,Int},Int}}()

    function tubes_ij(N2::Int, N1::Int, M1::Int, M2::Int)
        key_outer = @SVector [N2, N1, M1, M2]
        if haskey(cache, key_outer)
            return cache[key_outer]
        end

        # Call fusion rules only once
        #println("N2, N1, M1, M2: $((N2, N1, M1, M2))")
        nonzero_N = fusion_rules_N(N1, N2)
        if isempty(nonzero_N)
            return Dict{SVector{3,Int},Int}()  
        end
        nonzero_M = fusion_rules_M(M1, M2)
        if isempty(nonzero_M)
            return Dict{SVector{3,Int},Int}()  
        end

        #println("nonzero_N: $((nonzero_N))")
        #println("nonzero_M: $((nonzero_M))")

        # Fast set intersection
        common_Y = intersect!(collect(keys(nonzero_N)), keys(nonzero_M))
        #sort!(common_Y) 

        joined = Dict{SVector{3,Int},Int}()
        linear_index = 1

        # Avoid repeated sort allocations
        for Y in common_Y
            valsN = nonzero_N[Y]
            valsM = nonzero_M[Y]
            #sort!(valsN)
            #sort!(valsM)

            # Iterators.product is fast but we can unroll manually
            for a in valsN
                for b in valsM
                    key = @SVector [Y, a, b]
                    joined[key] = linear_index
                    linear_index += 1
                end
            end
        end

        cache[key_outer] = joined
        return joined
    end

    return tubes_ij
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

        # Cache and return
        cache[key] = reindexed_f_symbol
        return reindexed_f_symbol
    end

    return f_ijk_sparse
end

# WIP :()
function make_f_ijk_sparse__(F_N::SparseArray{ComplexF64, 10}, F_M::SparseArray{ComplexF64, 10}, 
                           F_quantum_dims::Vector{Float64}, size_dict::Dict{Symbol, Int}, 
                           tubes_ij)
    cache = Dict{Tuple{Int,Int,Int}, SparseArray}()

    # --- Quantum dimension prefactors ---
    sqrtd = sqrt.(F_quantum_dims)
    dY1 = SparseArray(sqrtd)
    dY2 = SparseArray(sqrtd)
    dY3 = SparseArray(1.0 ./ sqrtd)


    function f_ijk_sparse(i::Int, j::Int, k::Int)
        key = (i,j,k)
        println("i,j,k: $((key))")
        if haskey(cache, key)
            return cache[key]
        end

        # --- Decode flattened indices ---
        M_1, N_1 = index_to_tuple(i, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_2, N_2 = index_to_tuple(j, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_3, N_3 = index_to_tuple(k, (size_dict[:module_label_N], size_dict[:module_label_M]))

        # --- Slice tensors ---
        F_N_slice = slice_sparse_tensor(F_N, Dict(1=>N_1, 4=>N_3, 5=>N_2))
        F_M_slice = slice_sparse_tensor(F_M, Dict(1=>M_1, 4=>M_3, 5=>M_2))

        # --- Get linear indices from tube maps ---
        idx_a_map = tubes_ij(M_2, M_1, N_1, N_2)
        idx_b_map = tubes_ij(M_3, M_2, N_2, N_3)
        idx_c_map = tubes_ij(M_3, M_1, N_1, N_3)

        # --- COllect Nonzero Indices ---
        keys_N = collect(nonzero_keys(F_N_slice))
        vals_N = collect(nonzero_values(F_N_slice))
        keys_M = collect(nonzero_keys(F_M_slice))
        vals_M = collect(nonzero_values(F_M_slice))

        # Thread-local storage
        nthreads = Threads.nthreads()
        thread_DOK = [Dict{CartesianIndex{3}, ComplexF64}() for _ in 1:nthreads]

        # --- Outer loop multithreaded ---
        @threads for idx_N in eachindex(keys_N)
            tid = threadid()
            CI_N = keys_N[idx_N]
            val_N = vals_N[idx_N]

            #Y1, Y2, Y3, n1, n2, n4, m1, m2, m3 = y,p,x,r,s,n,a,m,b

            Y1, Y2, Y3, n1, l, n2, n4 = Tuple(CI_N)

            @inbounds @simd for idx_M in eachindex(keys_M)
                CI_M = keys_M[idx_M]
                val_M = vals_M[idx_M]

                Y1_, Y2_, Y3_, m1, m2, l_, m3 = Tuple(CI_M)

                if Y1_ == Y1 && Y2_ == Y2 && Y3_ == Y3 && l_ == l
                    factor = dY1[Y1] * dY2[Y2] * dY3[Y3]

                    #key_out = CartesianIndex(Y1, Y2, Y3, n1, n2, n4, m1, m2, m3)
                    key_out = CartesianIndex(idx_a_map[(Y1, m1, n1)], idx_b_map[(Y2, m2, n2)], idx_c_map[(Y3, m3, n4)])
                    thread_DOK[tid][key_out] = get(thread_DOK[tid], key_out, 0.0) + val_N * val_M * factor
                end
            end
        end

        # --- Merge thread-local dictionaries ---
        F_out_DOK = Dict{CartesianIndex{3}, ComplexF64}()
        for d in thread_DOK
            for (k,v) in d
                F_out_DOK[k] = get(F_out_DOK, k, 0.0) + v
            end
        end

        shape_out = (length(tubes_ij(M_2,M_1,N_1,N_2)), length(tubes_ij(M_3,M_2,N_2,N_3)), length(tubes_ij(M_3,M_1,N_1,N_3)))

        reindexed_f_symbol = SparseArray{ComplexF64,3}(F_out_DOK, shape_out)

        # Cache and return
        cache[key] = reindexed_f_symbol
        return reindexed_f_symbol
    end

    return f_ijk_sparse
end

function make_f_ijk_sparse___(F_N::SparseArray{ComplexF64, 10}, F_M::SparseArray{ComplexF64, 10}, F_quantum_dims::Vector{Float64}, size_dict::Dict{Symbol, Int}, tubes_ij)

    cache = Dict{Tuple{Int,Int,Int}, SparseArray{ComplexF64,3}}()

    # Precompute quantum dimension factors
    sqrtd = sqrt.(F_quantum_dims)
    dY1 = SparseArray(sqrtd)
    dY2 = SparseArray(sqrtd)
    dY3 = SparseArray(1.0 ./ sqrtd)

    function f_ijk_sparse(i::Int, j::Int, k::Int)

        # --- Caching ---
        key = (i,j,k)
        if haskey(cache, key)
            return cache[key]
        end

        # --- Decode flattened indices ---
        M_1, N_1 = index_to_tuple(i, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_2, N_2 = index_to_tuple(j, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_3, N_3 = index_to_tuple(k, (size_dict[:module_label_N], size_dict[:module_label_M]))

        # --- Sparse views ---
        """
        F_N_view = view(F_N, :, N_1, :, :, N_2, :, :, N_3, :, :)
        F_M_view = view(F_M, :, M_1, :, :, M_2, :, :, M_3, :, :)
        """
        F_N_slice = slice_sparse_tensor(F_N, Dict(1=>N_1, 4=>N_3, 5=>N_2))
        F_M_slice = slice_sparse_tensor(F_M, Dict(1=>M_1, 4=>M_3, 5=>M_2))

        # --- Tube maps ---
        idx_a_map = tubes_ij(M_2, M_1, N_1, N_2)
        idx_b_map = tubes_ij(M_3, M_2, N_2, N_3)
        idx_c_map = tubes_ij(M_3, M_1, N_1, N_3)

        # --- Nonzero keys and values ---
        keys_N = collect(nonzero_keys(F_N_slice))
        vals_N = collect(nonzero_values(F_N_slice))
        keys_M = collect(nonzero_keys(F_M_slice))
        vals_M = collect(nonzero_values(F_M_slice))

        #YN_array = reduce(hcat, Tuple.(keys_N))
        #YM_array = reduce(hcat, Tuple.(keys_M))

        nthreads = Threads.nthreads()
        thread_keys = [Vector{CartesianIndex{3}}() for _ in 1:nthreads]
        thread_vals = [Vector{ComplexF64}() for _ in 1:nthreads]

        # --- Parallel outer loop ---
        @threads for idx_N in 1:length(keys_N)
        #for idx_N in 1:length(keys_N)
            tid = Threads.threadid()
            #println(Tuple(keys_N[idx_N]))
            Y1, Y2, Y3, n1, l, n2, n4 = Tuple(keys_N[idx_N]) #YN_array[:,idx_N] 

            local_keys = thread_keys[tid]
            local_vals = thread_vals[tid]
            
            M_groups = Dict{NTuple{4,Int}, Vector{Int}}()
            @inbounds for idx_M in 1:length(keys_M)
                """Y1_, Y2_, Y3_, m1, m2, l_, m3 = Tuple(keys_M[idx_M]) #YM_array[:, idx_M] 

                if Y1_ == Y1 && Y2_ == Y2 && Y3_ == Y3 && l_ == l
                    factor = dY1[Y1] * dY2[Y2] * dY3[Y3]

                    push!(local_keys, CartesianIndex(
                        idx_a_map[(Y1, m1, n1)],
                        idx_b_map[(Y2, m2, n2)],
                        idx_c_map[(Y3, m3, n4)]
                    ))
                    push!(local_vals, vals_N[idx_N] * vals_M[idx_M] * factor)
                end"""

                
                Y1_,Y2_,Y3_,m1,m2,l_,m3 = Tuple(keys_M[idx_M])
                key4 = (Y1_,Y2_,Y3_,l_)
                push!(get!(M_groups, key4, Int[]), idx_M)
                
                for idx_N in 1:length(keys_N)
                    Y1,Y2,Y3,n1,l,n2,n4 = Tuple(keys_N[idx_N])
                    key4 = (Y1,Y2,Y3,l)

                    matched_idxs_M = get(M_groups, key4, nothing)
                    matched_idxs_M === nothing && continue

                    @inbounds for idx_M in matched_idxs_M
                        Y1_,Y2_,Y3_,m1,m2,l_,m3 = Tuple(keys_M[idx_M])

                        factor = dY1[Y1] * dY2[Y2] * dY3[Y3]
                        push!(local_keys, CartesianIndex( idx_a_map[(Y1, m1, n1)], idx_b_map[(Y2, m2, n2)], idx_c_map[(Y3, m3, n4)] ))
                        push!(local_vals, vals_N[idx_N] * vals_M[idx_M] * factor)
                    end
                end
                

            end
        end

        # --- Merge thread-local results ---
        keys_out = reduce(vcat, thread_keys)
        vals_out = reduce(vcat, thread_vals)

        shape_out = (length(idx_a_map), length(idx_b_map), length(idx_c_map))
        reindexed_f_symbol = SparseArray{ComplexF64,3}(Dict(zip(keys_out, vals_out)), shape_out)

        cache[key] = reindexed_f_symbol
        return reindexed_f_symbol
    end

    return f_ijk_sparse
end

"""function make_f_ijk_sparse( F_N::SparseArray{ComplexF64, 10},
                            F_M::SparseArray{ComplexF64, 10},
                            F_quantum_dims::Vector{Float64},
                            size_dict::Dict{Symbol, Int},
                            tubes_ij)

    cache = Dict{Tuple{Int,Int,Int}, SparseArray{ComplexF64,3}}()

    # Precompute quantum dimension factors (plain vectors for fast indexing)
    sqrtd = sqrt.(F_quantum_dims)
    dY1 = sqrtd
    dY2 = sqrtd
    dY3 = 1.0 ./ sqrtd

    function f_ijk_sparse(i::Int, j::Int, k::Int)
        key = (i,j,k)
        if haskey(cache, key)
            return cache[key]
        end

        # decode flattened indices
        M_1, N_1 = index_to_tuple(i, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_2, N_2 = index_to_tuple(j, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_3, N_3 = index_to_tuple(k, (size_dict[:module_label_N], size_dict[:module_label_M]))

        # preslice (use your slice function / view)
        F_N_slice = slice_sparse_tensor(F_N, Dict(1=>N_1, 4=>N_3, 5=>N_2))
        F_M_slice = slice_sparse_tensor(F_M, Dict(1=>M_1, 4=>M_3, 5=>M_2))

        # tube maps (dicts mapping (Y,m,n) -> linear index)
        idx_a_map = tubes_ij(M_2, M_1, N_1, N_2)
        idx_b_map = tubes_ij(M_3, M_2, N_2, N_3)
        idx_c_map = tubes_ij(M_3, M_1, N_1, N_3)

        # collect nonzero keys/values (materialize once)
        keys_N = collect(nonzero_keys(F_N_slice))   # Vector{CartesianIndex{7}} or similar
        println(size(keys_N))
        vals_N = collect(nonzero_values(F_N_slice))
        keys_M = collect(nonzero_keys(F_M_slice))
        vals_M = collect(nonzero_values(F_M_slice))

        # fast return if empty
        if isempty(keys_N) || isempty(keys_M)
            shape_out = (length(idx_a_map), length(idx_b_map), length(idx_c_map))
            reindexed_f_symbol = SparseArray{ComplexF64,3}(Dict{CartesianIndex{3},ComplexF64}(), shape_out)
            cache[key] = reindexed_f_symbol
            return reindexed_f_symbol
        end

        # Convert keys to tuple form and pre-extract components to arrays for O(1) indexing
        keysN_tuples = Tuple.(keys_N)  # Vector{NTuple{7,Int}}
        keysM_tuples = Tuple.(keys_M)

        # components arrays for M set (so inner loop avoids Tuple allocation)
        Y1m = [t[1] for t in keysM_tuples]
        Y2m = [t[2] for t in keysM_tuples]
        Y3m = [t[3] for t in keysM_tuples]
        m1_arr = [t[4] for t in keysM_tuples]
        m2_arr = [t[5] for t in keysM_tuples]
        lm_arr = [t[6] for t in keysM_tuples]
        m3_arr = [t[7] for t in keysM_tuples]

        # Build buckets: map (Y1,Y2,Y3,l) -> Vector{Int} of indices into keys_M
        M_groups = Dict{NTuple{4,Int}, Vector{Int}}()
        for idxM in 1:length(keysM_tuples)
            k4 = (Y1m[idxM], Y2m[idxM], Y3m[idxM], lm_arr[idxM])
            push!(get!(M_groups, k4, Int[]), idxM)
        end

        # Prepare thread-local result buffers
        nthreads = Threads.nthreads()
        thread_keys = [Vector{CartesianIndex{3}}() for _ in 1:nthreads]
        thread_vals = [Vector{ComplexF64}() for _ in 1:nthreads]

        # Parallel outer loop (each N-key handled independently)
        @threads for iN in 1:length(keysN_tuples)
            tid = threadid()
            local_keys = thread_keys[tid]
            local_vals = thread_vals[tid]

            tN = keysN_tuples[iN]
            # unpack N-tuple: adjust according to your actual tuple layout (here assumed 7)
            Y1, Y2, Y3, n1, l, n2, n4 = tN
            valN = vals_N[iN]
            # precompute factor and the map-get key
            factor = @inbounds dY1[Y1] * dY2[Y2] * dY3[Y3]
            key4 = (Y1, Y2, Y3, l)

            # retrieve matching M indices (fast) — skip if none
            matched = get(M_groups, key4, nothing)
            matched === nothing && continue

            # iterate only matching M entries
            @inbounds for idxM in matched
                # get components and value quickly
                m1 = m1_arr[idxM]
                m2 = m2_arr[idxM]
                m3 = m3_arr[idxM]
                valM = vals_M[idxM]
                
                # map to linear indices using tube maps
                idx_a = idx_a_map[@SVector [Y1, m1, n1]]
                idx_b = idx_b_map[@SVector [Y2, m2, n2]]
                idx_c = idx_c_map[@SVector [Y3, m3, n4]]

                push!(local_keys, CartesianIndex(idx_a, idx_b, idx_c))
                push!(local_vals, valN * valM * factor)
            end
        end

        # Merge thread-local buffers into single arrays
        keys_out = reduce(vcat, thread_keys)
        vals_out = reduce(vcat, thread_vals)

        # Aggregate duplicates by summing values for same CartesianIndex
        F_out_DOK = Dict{CartesianIndex{3}, ComplexF64}()
        @inbounds for t in 1:length(keys_out)
            k_out = keys_out[t]
            F_out_DOK[k_out] = get(F_out_DOK, k_out, 0.0 + 0im) + vals_out[t]
        end

        shape_out = (length(idx_a_map), length(idx_b_map), length(idx_c_map))
        reindexed_f_symbol = SparseArray{ComplexF64,3}(F_out_DOK, shape_out)

        # cache + return
        cache[key] = reindexed_f_symbol
        return reindexed_f_symbol
    end

    return f_ijk_sparse
end
"""

function make_f_ijk_sparse____(F_N::SparseArray{ComplexF64, 10}, F_M::SparseArray{ComplexF64, 10},F_quantum_dims::Vector{Float64},size_dict::Dict{Symbol, Int},tubes_ij)

    cache = Dict{Tuple{Int,Int,Int}, SparseArray{ComplexF64,3}}()

    # Precompute quantum dimension factors
    sqrtd = sqrt.(F_quantum_dims)
    dY1 = sqrtd
    dY2 = sqrtd
    dY3 = 1.0 ./ sqrtd

    function f_ijk_sparse(i::Int, j::Int, k::Int)
        key = (i,j,k)
        if haskey(cache, key)
            return cache[key]
        end

        # decode flattened indices
        M_1, N_1 = index_to_tuple(i, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_2, N_2 = index_to_tuple(j, (size_dict[:module_label_N], size_dict[:module_label_M]))
        M_3, N_3 = index_to_tuple(k, (size_dict[:module_label_N], size_dict[:module_label_M]))

        # preslice
        F_N_slice = slice_sparse_tensor(F_N, Dict(1=>N_1, 4=>N_3, 5=>N_2))
        F_M_slice = slice_sparse_tensor(F_M, Dict(1=>M_1, 4=>M_3, 5=>M_2))

        # tube maps
        idx_a_map = tubes_ij(M_2, M_1, N_1, N_2)
        idx_b_map = tubes_ij(M_3, M_2, N_2, N_3)
        idx_c_map = tubes_ij(M_3, M_1, N_1, N_3)

        # collect nonzero keys/values
        keys_N = collect(nonzero_keys(F_N_slice))
        vals_N = collect(nonzero_values(F_N_slice))
        keys_M = collect(nonzero_keys(F_M_slice))
        vals_M = collect(nonzero_values(F_M_slice))

        if isempty(keys_N) || isempty(keys_M)
            shape_out = (length(idx_a_map), length(idx_b_map), length(idx_c_map))
            reindexed_f_symbol = SparseArray{ComplexF64,3}(Dict{CartesianIndex{3},ComplexF64}(), shape_out)
            cache[key] = reindexed_f_symbol
            return reindexed_f_symbol
        end

        # Pre-extract components as arrays to avoid tuple allocations
        Y1M = Int[]
        Y2M = Int[]
        Y3M = Int[]
        m1_arr = Int[]
        m2_arr = Int[]
        lM_arr = Int[]
        m3_arr = Int[]

        for kM in keys_M
            push!(Y1M, kM[1])
            push!(Y2M, kM[2])
            push!(Y3M, kM[3])
            push!(m1_arr, kM[4])
            push!(m2_arr, kM[5])
            push!(lM_arr, kM[6])
            push!(m3_arr, kM[7])
        end

        # Bucket M indices by (Y1,Y2,Y3,l)
        M_groups = Dict{NTuple{4,Int}, Vector{Int}}()
        for idxM in 1:length(keys_M)
            key4 = (Y1M[idxM], Y2M[idxM], Y3M[idxM], lM_arr[idxM])
            push!(get!(M_groups, key4, Int[]), idxM)
        end

        # Result arrays
        keys_out = Vector{CartesianIndex{3}}()
        vals_out = Vector{ComplexF64}()

        # Sequential loop over N
        for iN in 1:length(keys_N)
            kN = keys_N[iN]
            Y1, Y2, Y3 = kN[1], kN[2], kN[3]
            n1, l, n2, n4 = kN[4], kN[6], kN[5], kN[7] # adjust order if needed
            valN = vals_N[iN]

            factor = dY1[Y1] * dY2[Y2] * dY3[Y3]
            key4 = (Y1, Y2, Y3, l)

            matched = get(M_groups, key4, nothing)
            matched === nothing && continue

            for idxM in matched
                m1, m2, m3 = m1_arr[idxM], m2_arr[idxM], m3_arr[idxM]
                valM = vals_M[idxM]

                # map to linear indices
                idx_a = idx_a_map[@SVector [Y1, m1, n1]]
                idx_b = idx_b_map[@SVector [Y2, m2, n2]]
                idx_c = idx_c_map[@SVector [Y3, m3, n4]]

                push!(keys_out, CartesianIndex(idx_a, idx_b, idx_c))
                push!(vals_out, valN * valM * factor)
            end
        end

        # Aggregate duplicates
        F_out_DOK = Dict{CartesianIndex{3}, ComplexF64}()
        @inbounds for t in 1:length(keys_out)
            k_out = keys_out[t]
            F_out_DOK[k_out] = get(F_out_DOK, k_out, 0.0 + 0im) + vals_out[t]
        end

        shape_out = (length(idx_a_map), length(idx_b_map), length(idx_c_map))
        reindexed_f_symbol = SparseArray{ComplexF64,3}(F_out_DOK, shape_out)

        cache[key] = reindexed_f_symbol
        return reindexed_f_symbol
    end

    return f_ijk_sparse
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