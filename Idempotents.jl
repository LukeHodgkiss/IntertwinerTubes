module Idempotents

using ..SparseAlgebraObjects: AlgebraVec, EigVec, SubAlgebraElementBlockL,
        SubAlgebraElementBlockR, TubeAlgebra, Vec, inner_product,
        random_left_linear_combination_ijk, random_right_linear_combination_ijk
using LinearAlgebra, Random
using SparseArrayKit: SparseArray, nonzero_values, nonzero_keys, nonzero_pairs

export eigen_decomposition_subalgebra_block, build_block, remove_overlapping_evec, remove_evec_in_same_block_ii, build_out_irrep, find_idempotents

"""
include("SparseAlgebraObjects.jl")
using .SparseAlgebraObjects: TubeAlgebra, EigVec, Vec, random_left_linear_combination_ijk, random_right_linear_combination_ijk
"""

# ---------- eigen decomposition of (i,i,i) single block ---- 
"""
    eigen_decomposition_subalgebra_block(algebra::TubeAlgebra, i::Int; rng=GLOBAL_RNG)
"""
function eigen_decomposition_subalgebra_block(algebra::TubeAlgebra,random_left_linear_combination_ijk::Function, i::Int; rng::AbstractRNG = Random.GLOBAL_RNG)
    LX = random_left_linear_combination_ijk(algebra, i, i, i; rng=rng)
    vals, vecs = eigen(LX.LX)
    #println(vals)
    #println(vecs)

    #vals, vecs = eigen(Hermitian(LX.LX))

    results = Vector{EigVec}(undef, length(vals))
    #@show vals, vecs
    for c in eachindex(vals)
        v = vecs[:, c]  # column eigenvector
        #println("evaltype: $(typeof(vals[c]))")
        results[c] = EigVec(vec(v), (i,i), i, ComplexF64(vals[c]))
        #results[c] = EigVec(Matrix(v), (i,i), i, ComplexF64(vals[c]))
        #results[c] = EigVec(v, (i,i), i, ComplexF64(vals[c]))
    end
    return results

end

# ---------- build_block: expand subspace by repeated random L multiplication ----------
"""
    build_block(v::EigVec, L_X::Function, j::Int, rng; tol_rank=1e-10) -> Matrix{ComplexF64}

Start from eigenvector `v` (in subalgebra (i,i)) and build an orthonormal basis Q
for the block (j, i) by repeatedly applying random left elements `L_X(j, left, right; rng=rng)`.
Returns Q (columns = orthonormal basis vectors).
"""
function build_block(v::EigVec, algebra::TubeAlgebra, L_X::Function, j::Int, d_irrep::Int, d_subalgebra_iii::Int, rng::AbstractRNG; 
    tol_rank::Float64 = 1e-10)
    # initial vector produced by applying one random L
    L0 = L_X(algebra, j, v.subalgebra[1], v.subalgebra[2]; rng=rng)
    #@show j, v.subalgebra[1], v.subalgebra[2]
    first_vec = (L0 * v).vector             # Vec returned by block * v
    Vcols = [first_vec]                     # column vectors as Vector{Vector{ComplexF64}}
    # initial QR
    M = hcat(Vcols...)
    qr_decomp = qr(M)
    Q_old = Matrix(qr_decomp.Q)
    R = qr_decomp.R
    rank_old = count(abs.(diag(R)) .> tol_rank)

    while true
        L = L_X(algebra, j, v.subalgebra[1], v.subalgebra[2]; rng=rng)
        new_v = (L * v).vector
        M_new = hcat(Vcols..., new_v)
        qr_decomp = qr(M_new)
        Rnew = qr_decomp.R
        rank_new = count(abs.(diag(Rnew)) .> tol_rank)

        
        if rank_new > rank_old
            push!(Vcols, new_v)
            Q_old = Matrix(qr_decomp.Q)
            rank_old = rank_new
            d_irrep_new = d_irrep + size(Q_old)[1]^2

            if d_irrep_new>=d_subalgebra_iii
                break
            end

        else
            break
        end
    end
    #println(size(Q_old))
    return Q_old, d_irrep
end

# ---------- remove_overlapping_evec ----------
"""
remove_overlapping_evec(algebra::TubeAlgebra, ED_ii::Vector{EigVec}, ED::Vector{EigVec}; tol_overlap=1e-9)

Return ED_ii with eigenvectors removed that have overlap > tol_overlap with any vector in ED.
"""
function remove_overlapping_evec(algebra::TubeAlgebra, random_left_linear_combination_ijk::Function, random_right_linear_combination_ijk::Function, ED_ii::Vector{EigVec}, ED::Vector{EigVec}; 
    tol_overlap::Float64 = 1e-10)
    trimmed = Vector{EigVec}()
    for evec in ED_ii
        overlaps = 0.0 + 0im
        s, t = evec.subalgebra
        for old_evec in ED
            i, j = old_evec.subalgebra
            if algebra.dim_ijk(j,t,s) !== nothing && algebra.dim_ijk(s,i,j) !== nothing
            #if haskey(algebra.dimension_dict, (j, t, s)) && haskey(algebra.dimension_dict, (s, i, j))
                LX_sij = random_left_linear_combination_ijk(algebra, s, i, j)
                RX_jts = random_right_linear_combination_ijk(algebra, j, t, s)
                
                vl = RX_jts * evec
                vr = LX_sij * old_evec
                """
                print("Problem Here")
                @show j, t, s, s, i, j
                @show algebra.dimension_dict[(j, t, s)], algebra.dimension_dict[(s, i, j)]
                @show size(RX_jts.LX), size(evec.vector), size(vl.vector)
                @show RX_jts.subalgebra, evec.subalgebra, vl.subalgebra
                @show size(LX_sij.LX), size(old_evec.vector), size(vr.vector)
                @show LX_sij.subalgebra, old_evec.subalgebra, vr.subalgebra
                """
                overlaps = abs(inner_product(vl, vr))
                #println(overlaps)
            else
                overlaps = 0.0
            end

            if abs(overlaps) > tol_overlap
                break
            end
        end

        if abs(overlaps) < tol_overlap
            push!(trimmed, evec)
        end
    end
    return trimmed
end

# ---------- remove_evec_in_same_block_ii ----------
"""
    remove_evec_in_same_block_ii(ED_ii, RX_iii, LX_iii; tol_overlap=1e-12)

Keep only eigenvectors in ED_ii that are pairwise orthogonal under the action of RX_iii and LX_iii.
"""
function remove_evec_in_same_block_ii(ED_ii::Vector{EigVec}, RX_iii, LX_iii; 
    tol_overlap::Float64 = 1e-10)
    kept = Vector{EigVec}()
    for e1 in ED_ii
        ok = all(abs(inner_product(RX_iii * e1, LX_iii * e2)) < tol_overlap for e2 in kept)
        if ok
            push!(kept, e1)
        end
    end
    return kept
end

# ---------- build_out_irrep ----------
"""
build_out_irrep(v::EigVec, i::Int, algebra::TubeAlgebra, rng) -> Dict{Tuple{Int,Int}, Matrix}

For each j in 1:algebra.N_diag_blocks, if (j, v.subalgebra[1], v.subalgebra[2]) exists,
build the block basis Q via build_block and store it under key (j, v.subalgebra[2]).
"""
function build_out_irrep(v::EigVec, i::Int, algebra::TubeAlgebra, random_left_linear_combination_ijk::Function, rng::AbstractRNG)
    irrep_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    d_subalgebra_iii = algebra.dim_ijk(i,i,i)[1]^2
    d_irrep = 0
    
    for j in i:algebra.N_diag_blocks
        if algebra.dim_ijk(j, v.subalgebra[1], v.subalgebra[2]) !== nothing
            #println("alg: $((j, v.subalgebra[1], v.subalgebra[2])))")
            #println(" building out block ")
            Q, d_irrep = build_block(v, algebra, random_left_linear_combination_ijk, j, d_irrep, d_subalgebra_iii, rng)
            irrep_blocks[(j, v.subalgebra[2])] = Q
        end
    end
    return irrep_blocks, d_irrep
end

# ---------- helpers ----------
d_sum(d_list::Vector{Int}) = sum(x->x, d_list)

# ---------- find_idempotents (main) ----------
"""
    find_idempotents(algebra::TubeAlgebra) -> Vector{Dict{(Int,Int)=>Matrix}}

Diagonalize each diagonal subalgebra, remove overlaps, build irreps,
stop when total sum of squared irrep dims reaches algebra.d_algebra_squared.
"""
function find_idempotents(algebra::TubeAlgebra)
    irrep_projectors = Vector{Dict{Tuple{Int,Int}, Matrix{ComplexF64}}}()
    ED_global = Vector{EigVec}()
    d_algebra_squared = algebra.d_algebra_squared
    rng = MersenneTwister(42)
    d_irrep_list = Int[]

    #dim_ii_start = np.zeros(algebra.N_diag_blocks)
    #size_block = algebra.dim_dict(ii,ii,ii)
    #dim_ii_start[ii] += size(QR)[1]
    #if dim_ii_start > size_block
    #   break


    for ii in 1:algebra.N_diag_blocks

        # diagonalize block (ii,ii)
        #@show ii, algebra.N_diag_blocks
        println("Considering block ED: $(ii)")
        ED_ii = eigen_decomposition_subalgebra_block(algebra, random_left_linear_combination_ijk, ii; rng=rng)

        # remove overlapping vectors between the blocks
        #println("remove overlaps with global ED")
        ED_ii_ortho = remove_overlapping_evec(algebra, random_left_linear_combination_ijk, random_right_linear_combination_ijk, ED_ii, ED_global)

        # remove overlapping vectors within the block
        #println("remove overlaps within the block")
        RX_iii = random_right_linear_combination_ijk(algebra, ii, ii, ii; rng=rng)
        LX_iii = random_left_linear_combination_ijk(algebra, ii, ii, ii; rng=rng)
        ED_ii_trimmed = remove_evec_in_same_block_ii(ED_ii_ortho, RX_iii, LX_iii)

        append!(ED_global, ED_ii_trimmed)

        # build irreps from each unique eigenvector
        for vec in ED_ii_trimmed
            println("build irreps from each unique eigenvector")
            irrep, d_irrep = build_out_irrep(vec, ii, algebra, random_left_linear_combination_ijk, rng)
            d_irrep = sum(size(Q)[1]^2 for Q in values(irrep))
            println("d_irrep $d_irrep")
            #d_irrep = length(values(irrep))
            push!(irrep_projectors, irrep)
            push!(d_irrep_list, d_irrep)   
            if d_sum(d_irrep_list) >= d_algebra_squared
                print("We saved time? :)")
                return irrep_projectors
            end
        end
    end

    return irrep_projectors
end

end # Module

#=
if d_sum(d_irrep_list) >= d_algebra_squared
                print("We saved time? :)")
                #return irrep_projectors
            end
=#

#=



=#