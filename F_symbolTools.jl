module FSymbolTools

export F_vec_G, F_mod_cat_Vec_Vec_G, triple_line_to_linear_index, reindexdims, remove_zeros!, slice_sparse_tensor, tuple_to_index, index_to_tuple, SparseSliceView

using SparseArrayKit: SparseArray, nonzero_values, nonzero_keys, nonzero_pairs
using LinearAlgebra
using TensorOperations
using TupleTools


"""
    triple_line_to_linear_index(X, M, N, j, shapes)

Given multi-indices (X, M, N, j) and a tuple of dimensions `shapes`,
return the linear index and the total size.
Julia’s `LinearIndices` is used to mimic NumPy’s `ravel_multi_index`.

        M
    _________
    ____X____|(j) ----- (XMN,j):= A
    _________|
        N

"""

function triple_line_to_linear_index(idx::Vector{Int}, 
                                     X::Vector{Int}, M::Vector{Int}, N::Vector{Int}, j::Vector{Int}, 
                                     shapes::NTuple{4,Int})
    li = LinearIndices(shapes)
    map!((x,m,n,j_) -> li[x,m,n,j_], idx, X, M, N, j)
    shape_A = prod(shapes)
    return idx, shape_A
end


"""
Permutes and duplicates axes of a sparse tensor `F` according to `new_axes`.
"""
function reindexdims(F::SparseArray{T,N}, new_axes::NTuple{K,Int}) where {T,N,K}
    # Determine new shape
    old_shape = size(F)
    new_shape = ntuple(i -> old_shape[new_axes[i]], length(new_axes))

    # Build new coordinate dictionary
    coord_dict = Dict{CartesianIndex{length(new_axes)}, T}()
    
    for (old_idx, val) in pairs(F.data)
        old_tuple = Tuple(old_idx)
        new_tuple = ntuple(i -> old_tuple[new_axes[i]], length(new_axes))
        coord_dict[CartesianIndex(new_tuple)] = val
    end

    return SparseArray{T, length(new_axes)}(coord_dict, new_shape)
end
"""
function reindexdims(A::SparseArray, p::IndexTuple)
    C = similar(A, TupleTools.getindices(size(A), p))
    return reindexdims!(C, A, p)
end
function reindexdims!(C::SparseArray{T, N}, A::SparseArray, p::IndexTuple{N}) where {T, N}
    #_zero!(C)
    fill!(C, 0)  
    _sizehint!(C, nonzero_length(A))
    for (IA, vA) in nonzero_pairs(A)
        IC = CartesianIndex(TupleTools.getindices(IA.I, p))
        increaseindex!(C, vA, IC)
    end
    return C
end
"""

"""
Remove small entries from a SparseArrayKit array.
"""
function remove_zeros!(A::SparseArray{T,N}, tol::Float64 = 1e-12) where {T,N}
    h = A.data   
    # Collect keys to delete to avoid modifying dict while iterating
    for key in keys(h)
        val = h[key]
        if abs(val) < tol
            delete!(h, key)
        end
    end
    return A
end

import Base: getindex, keys

"""
Slice a high-rank SparseArray F along specific axes.

# Arguments
- `F::SparseArray{T,N}`: input sparse tensor
- `fixed::Dict{Int,Int}`: dictionary of axes to fix (key = axis index, value = index to fix)

# Returns
- `F_sliced::SparseArray{T,M}`: sliced sparse tensor with axes in `fixed` removed
"""

struct SparseSliceView{T,N,M}
    F::SparseArray{T,N}         
    fixed::Dict{Int,Int}        
    remaining_axes::Vector{Int} 
    axis_map::Vector{Int}      
end

function SparseSliceView(F::SparseArray{T,N}, fixed::Dict{Int,Int}) where {T,N}
    remaining_axes = [ax for ax in 1:N if !haskey(fixed, ax)]
    axis_map = [haskey(fixed,i) ? 0 : findfirst(==(i), remaining_axes) for i in 1:N]
    return SparseSliceView{T,N,length(remaining_axes)}(F, fixed, remaining_axes, axis_map)
end

function Base.getindex(s::SparseSliceView{T,N,M}, idx::Vararg{Int,M}) where {T,N,M}
    @assert M == length(s.remaining_axes) "Wrong number of indices for slice"
    full_idx = ntuple(i -> s.axis_map[i] == 0 ? s.fixed[i] : idx[s.axis_map[i]], N)
    return s.F[full_idx...]  # splat tuple
end

function Base.keys(s::SparseSliceView)
    filtered = Iterators.filter(k -> all(k[ax] == val for (ax,val) in s.fixed), Tuple.(keys(s.F.data)))
    return (Tuple(k[s.remaining_axes]) for k in filtered)
end


function slice_sparse_tensor(F::SparseArray{T,N}, fixed::Dict{Int,Int}) where {T,N}

    remaining_axes = setdiff(1:N, collect(keys(fixed)))
    new_dims = ntuple(i -> F.dims[remaining_axes[i]], length(remaining_axes))

    sliced_data = Dict{CartesianIndex{length(new_dims)}, T}()

    for (idx, val) in pairs(F.data)
        
        match = all(idx[ax] == fixed[ax] for ax in keys(fixed))
        if match
            new_idx = CartesianIndex(ntuple(i -> idx[remaining_axes[i]], length(remaining_axes)))
            sliced_data[new_idx] = val
        end
    end

    return SparseArray{T,length(new_dims)}(sliced_data, new_dims)
end


"""
Convert a multi-dimensional tuple `tup` to a linear index
given the array `shape`.
"""
function tuple_to_index(tup::NTuple{N, Int}, shape::NTuple{N, Int}) where N
    # Convert tuple to CartesianIndex and get linear index
    return LinearIndices(shape)[CartesianIndex(tup)]
end

"""
Convert a linear index `idx` to a multi-dimensional tuple
given the array `shape`.
"""
function index_to_tuple(idx::Int, shape::NTuple{N, Int}) where N
    return Tuple(CartesianIndices(shape)[idx])
end

"""
Convert a rank-10 sparse tensor F into a rank-4 MPO-type tensor with
coordinates A,B,C,D computed from the triple-line flattening.
Assumes F.coords is a tuple of index arrays and F.data is the value array.

          D
          |
          |
    A---(mpo)---C
          |
          |
          B
"""
function MPO_tensor(F::SparseArray{ComplexF64,10})
    keys_array = collect(nonzero_keys(F))                     
    nnz = length(keys_array)

    coords_matrix = Array{Int,2}(undef, 10, nnz)
    for (col, CI) in enumerate(keys_array)
        @inbounds coords_matrix[:, col] .= Tuple(CI)
    end
    X, N1, Y, M2, M1, N2, i, k, l, j = ntuple(d -> coords_matrix[d, :], 10)
    Xs, N1s, Ys, M2s, M1s, N2s, isz, ksz, lsz, jsz = size(F)
    

    A_vec = Vector{Int}(undef, nnz)
    A, shapeA = triple_line_to_linear_index(A_vec, X, M1, N1, i, (Xs, M1s, N1s, isz))
    B_vec = Vector{Int}(undef, nnz)
    B, shapeB = triple_line_to_linear_index(B_vec, Y, M1, M2, l, (Ys, M1s, M2s, lsz))
    C_vec = Vector{Int}(undef, nnz)
    C, shapeC = triple_line_to_linear_index(C_vec, X, M2, N2, j, (Xs, M2s, N2s, jsz))
    D_vec = Vector{Int}(undef, nnz)
    D, shapeD = triple_line_to_linear_index(D_vec, Y, N1, N2, k, (Ys, N1s, N2s, ksz))


    shape = (shapeA, shapeB, shapeC, shapeD)
    vals = collect(nonzero_values(F))
    #print(vals)
    coord_dict = Dict{CartesianIndex{4}, ComplexF64}()
    for i in 1:nnz
        coord_dict[CartesianIndex(A[i], B[i], C[i], D[i])] = vals[i]
    end

    MPO_F = SparseArray{ComplexF64, 4}(coord_dict, shape)
    return MPO_F
end

"""
Convert a rank-10 fusion tensor F into a rank-4 tensor using triple-line flattening.
             l
            /
           /
    A---(PEPS)---C
          |
          |
          B
"""

function fusion_tensor(F::SparseArray{ComplexF64,10})

    keys_array = collect(nonzero_keys(F))
    nnz = length(keys_array)
    
    coords_matrix = Array{Int,2}(undef, 10, nnz)
    for (col, CI) in enumerate(keys_array)
        @inbounds coords_matrix[:, col] .= Tuple(CI)
    end
    M1, Y1, Y2, M2, M3, Y3, i, k, l, j = ntuple(d -> coords_matrix[d, :], 10)
    M1s, Y1s, Y2s, M2s, M3s, Y3s, isz, ksz, lsz, jsz = size(F)

    A_vec = Vector{Int}(undef, nnz)
    A, shapeA = triple_line_to_linear_index(A_vec, Y1, M1, M3, i, (Y1s, M1s, M3s, isz))
    B_vec = Vector{Int}(undef, nnz)
    B, shapeB = triple_line_to_linear_index(B_vec, Y3, M1, M2, j, (Y3s, M1s, M2s, jsz))
    C_vec = Vector{Int}(undef, nnz)
    C, shapeC = triple_line_to_linear_index(C_vec, Y2, M2, M3, k, (Y2s, M2s, M3s, ksz))

    vals = collect(nonzero_values(F))
    coord_dict = Dict{CartesianIndex{4}, ComplexF64}()
    for idx in 1:nnz
        coord_dict[CartesianIndex(l[idx], A[idx], B[idx], C[idx])] = vals[idx]
    end

    shape = (lsz, shapeA, shapeB, shapeC)
    X = SparseArray{ComplexF64,4}(coord_dict, shape)

    return X
end


function F_vec_G(cayley_table)
    order = size(cayley_table, 1)
    shape = (order, order, order, order, order, order, 1,1,1,1)
    F = DOK{Float64}(shape...)

    for g1 in 1:order
        for g2 in 1:order
            g12 = cayley_table[g1,g2]
            for g3 in 1:order
                g23  = cayley_table[g2,g3]
                g123 = cayley_table[g1, g23]

                idx = (g1, g2, g3, g123, g12, g23, 1,1,1,1)
                F[idx] = 1.0
            end
        end
    end

    return F
end

function F_mod_cat_Vec_Vec_G(cayley_table)
    order = size(cayley_table, 1)
    shape = (1, order, order, 1, 1, order, 1,1,1,1)
    F = DOK{Float64}(shape...)

    for g1 in 1:order
        for g2 in 1:order
            g1g2 = cayley_table[g1,g2]
            idx = (1, g1, g2, 1, 1, g1g2, 1,1,1,1)
            F[idx] = 1.0
        end
    end

    return F
end

end # module