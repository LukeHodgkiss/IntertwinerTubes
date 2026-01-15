module Idempotents

using ..SparseAlgebraObjects: AlgebraVec, EigVec, SubAlgebraElementBlockL,
        SubAlgebraElementBlockR, TubeAlgebra, Vec, inner_product,
        random_left_linear_combination_ijk, random_right_linear_combination_ijk
using LinearAlgebra, Random
using SparseArrayKit: SparseArray, nonzero_values, nonzero_keys, nonzero_pairs

export eigen_decomposition_subalgebra_block, build_block, remove_overlapping_evec, remove_evec_in_same_block_ii, build_out_irrep, find_idempotents, dim_calc

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
    tol_rank::Float64 = 1e-9)
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
    #println("rank_old: $((rank_old))")
    if sum(abs.(diag(R))) < tol_rank
        return zeros(0, 0), d_irrep
    end
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

            #=
            if d_irrep_new>=d_subalgebra_iii
                break
            end
            =#
        else
            break
        end
    end
    
    Q_old[abs.(Q_old) .< 1e-12] .= 0
    Q_old = Q_old * Diagonal(exp.(-im .* angle.(Q_old[1, :])))
    #==#
    return Q_old, d_irrep
end

# ---------- remove_overlapping_evec ----------
"""
remove_overlapping_evec(algebra::TubeAlgebra, ED_ii::Vector{EigVec}, ED::Vector{EigVec}; tol_overlap=1e-9)

Return ED_ii with eigenvectors removed that have overlap > tol_overlap with any vector in ED.
"""
function remove_overlapping_evec(algebra::TubeAlgebra, random_left_linear_combination_ijk::Function, random_right_linear_combination_ijk::Function, ED_ii::Vector{EigVec}, ED::Vector{EigVec}; 
    tol_overlap::Float64 = 1e-11)
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
                overlaps = abs(inner_product(RX_jts * evec, LX_sij * old_evec))
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
        overlaps = 0
        for e2 in kept
            overlaps += abs(inner_product(RX_iii * e1, LX_iii * e2))
            if overlaps>tol_overlap
                break
            end
        end

        if overlaps < tol_overlap 
            push!(kept, e1)
        end
    end
    return kept
end

# ---------- build_out_irrep ----------
function build_out_irrep(v::EigVec, i::Int, algebra::TubeAlgebra, random_left_linear_combination_ijk::Function, rng::AbstractRNG)
    irrep_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    d_subalgebra_iii = algebra.dim_ijk(i,i,i)[1]^2
    d_irrep = 0

    for j in i:algebra.N_diag_blocks
        if algebra.dim_ijk(j, v.subalgebra[1], v.subalgebra[2]) !== nothing
            
            Q, d_irrep = build_block(v, algebra, random_left_linear_combination_ijk, j, d_irrep, d_subalgebra_iii, rng)
            if size(Q,1) > 0; irrep_blocks[(j, v.subalgebra[2])] = Q; end 
        end
    end
    return irrep_blocks, d_irrep
end

# ---------- helpers ----------
d_sum(d_list::Vector{Int}) = sum(x->x, d_list)

# ---------- find_idempotents (main) ----------
"""
 1.   Diagonalize each diagonal subalgebra, 
 2.   remove overlaps, 
 3.   build out irreps,
 4   stop when total sum of squared irrep dims reaches algebra.d_algebra_squared.
"""
function find_idempotents(algebra::TubeAlgebra)
    irrep_projectors = Vector{Dict{Tuple{Int,Int}, Matrix{ComplexF64}}}()
    ED_global = Vector{EigVec}()
    rng = MersenneTwister(42)
    d_irrep_list = Int[]


    for ii in 1:algebra.N_diag_blocks
        
        ED_ii = eigen_decomposition_subalgebra_block(algebra, random_left_linear_combination_ijk, ii; rng=rng)
        ED_ii_ortho = remove_overlapping_evec(algebra, random_left_linear_combination_ijk, random_right_linear_combination_ijk, ED_ii, ED_global)

        RX_iii = random_right_linear_combination_ijk(algebra, ii, ii, ii; isHermitian=false, rng=rng)
        LX_iii = random_left_linear_combination_ijk(algebra, ii, ii, ii; isHermitian=false, rng=rng)
        ED_ii_trimmed = remove_evec_in_same_block_ii(ED_ii_ortho, RX_iii, LX_iii)

        append!(ED_global, ED_ii_trimmed)

        for vec in ED_ii_trimmed
            irrep, d_irrep = build_out_irrep(vec, ii, algebra, random_left_linear_combination_ijk, rng)
            if length(irrep) > 0 # this shuldnt be neccesary
                push!(irrep_projectors, irrep)
                d_irrep = sum(size(Q)[2]^2 for Q in values(irrep))
                push!(d_irrep_list, d_irrep)
            end
               
            #=
            if d_sum(d_irrep_list) >= algebra.d_algebra_squared
                print("We saved time? :)")
                return irrep_projectors
            end
            =#
        end
    end
    #@show sum(d_irrep_list)
    return irrep_projectors
end


function dim_calc(idempotents_dict)
    dim_alg_glob = 0
    for irrep in idempotents_dict
        println("Irrep has size: $(length(irrep))")
        
        for (ij, proj) in irrep
            println("Projector $(ij) has shape $(size(proj)) ")
            dim_alg_glob=dim_alg_glob+((size(proj)[2])^2)
            #println(proj)
        end 
    end
    #@show dim_alg_glob
end

end # Module

#=
if d_sum(d_irrep_list) >= d_algebra_squared
                print("We saved time? :)")
                #return irrep_projectors
            end
=#

