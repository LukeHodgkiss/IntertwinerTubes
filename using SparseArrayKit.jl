using SparseArrayKit
using MAT
using LinearAlgebra
using TensorOperations

vars = matread("Luke_F.mat")
Fsparse = SparseArray{ComplexF64}(vars["F"])

@time begin
    mpo = reindexdims(Fsparse,(5,1,2,7,4,1,6,10,2,3,6,9,5,3,4,8))
    mpo = reshape(mpo,(128,128,128,128))

    peps = reindexdims(Fsparse,(1,2,5,7,5,3,4,8,1,6,4,10,2,3,6,9))
    peps = reshape(peps,(128,128,128,128))

    @tensor lhs[-1 -2 -3 -4 -5 -6] := mpo[-1 -2 -3 1]*peps[-4 -5 1 -6]
    @tensor rhs[-1 -2 -3 -4 -5 -6] := peps[1 2 -3 -6]*mpo[-1 3 1 -4]*mpo[3 -2 2 -5]

    display(norm(lhs-rhs))
end
