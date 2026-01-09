using SparseArrayKit: SparseArray, nonzero_values, nonzero_keys
using MAT
using LinearAlgebra
using TensorOperations
using time

vars = matread("Luke_F.mat")
F = SparseArray{ComplexF64}(vars["F"])
#print(F)

function triple_line_to_linear_index(idx::Vector{Int}, 
                                     X::Vector{Int}, M::Vector{Int}, N::Vector{Int}, j::Vector{Int}, 
                                     shapes::NTuple{4,Int})
    li = LinearIndices(shapes)
    map!((x,m,n,j_) -> li[x,m,n,j_], idx, X, M, N, j)
    shape_A = prod(shapes)
    return idx, shape_A
end


"""
    MPO_tensor(F)

Convert a rank-10 sparse tensor F into a rank-4 MPO-type tensor with
coordinates A,B,C,D computed from the triple-line flattening.
Assumes F.coords is a tuple of index arrays and F.data is the value array.
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
    
    mpo = reindexdims(F,(5,1,2,7,4,1,6,10,2,3,6,9,5,3,4,8))
    mpo = reshape(mpo,(128,128,128,128))

    return MPO_F
end

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

@time begin
    mpo = reindexdims(F,(5,1,2,7,4,1,6,10,2,3,6,9,5,3,4,8))
    mpo = reshape(mpo,(128,128,128,128))

    peps = reindexdims(F,(1,2,5,7,5,3,4,8,1,6,4,10,2,3,6,9))
    peps = reshape(peps,(128,128,128,128))

    @tensor lhs[-1 -2 -3 -4 -5 -6] := mpo[-1 -2 -3 1]*peps[-4 -5 1 -6]
    @tensor rhs[-1 -2 -3 -4 -5 -6] := peps[1 2 -3 -6]*mpo[-1 3 1 -4]*mpo[3 -2 2 -5]
    
    print(norm(lhs-rhs))
end