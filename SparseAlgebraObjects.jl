module SparseAlgebraObjects
using LinearAlgebra, Random
using SparseArrayKit: SparseArray, nonzero_values, nonzero_keys, nonzero_pairs

export Vec, EigVec, AlgebraVec, inner_product, TubeAlgebra, random_left_linear_combination_ijk, random_right_linear_combination_ijk, create_left_ijk_basis, create_right_ijk_basis

# -------------------------------
# Vec and Eigenvector Types
# -------------------------------

abstract type AbstractVec end

mutable struct Vec <: AbstractVec
    vector#::Matrix{ComplexF64}
    subalgebra::Tuple{Int,Int}
    irrep_index::Int
end

# Constructor with keyword args
function Vec(vector::Vector{ComplexF64}; subalgebra=nothing, irrep_index=nothing)
    Vec(vector, subalgebra, irrep_index)
end

# Zero vector constructor
VecZero(dim::Int, irrep_index::Int, subalgebra::Tuple{Int,Int}) =
    Vec(zeros(ComplexF64, dim), subalgebra, irrep_index)

# EigVec type, subtype of AbstractVec
mutable struct EigVec <: AbstractVec
    vector#::AbstractMatrix{<:Complex}
    subalgebra::Tuple{Int,Int}
    irrep_index::Int
    eigenvalue::ComplexF64
    eigindex::Union{Nothing,Int}
    degenindex::Union{Nothing,Int}
end

# Constructor
function EigVec(vector, subalgebra::Tuple{Int,Int}, irrep_index::Int, eigenvalue::ComplexF64; #eigenvalue should be real since hermitian matrix
                eigindex=nothing, degenindex=nothing)
    EigVec(vector, subalgebra, irrep_index, eigenvalue, eigindex, degenindex)
end

# Vec addition
function Base.:+(v1::Vec, v2::Vec)
    if v1.subalgebra == v2.subalgebra
        Vec(v1.vector + v2.vector, subalgebra=v1.subalgebra, irrep_index=v1.irrep_index)
    else
        error("Cannot add Vec from different subalgebras")
    end
end

struct AlgebraVec
    vecs::Dict{Tuple{Int,Int}, Vec}
end


# -------------------------------
# Complex Inner Product ⟨v2 | v1⟩
# -------------------------------
""" inner_product(v1, v2) -> ComplexF64 """

#  Vec – Vec
function inner_product(v1::Vec, v2::Vec)
    if v1.subalgebra == v2.subalgebra
        return dot(conj.(v2.vector), v1.vector)
        
    else
        return 0.0 + 0im
    end
end

#  Vec – AlgebraVec
function inner_product(v1::Vec, v2::AlgebraVec)
    vec2 = get(v2.vecs, v1.subalgebra, nothing)
    vec2 === nothing && return 0.0 + 0im
   
    return dot(vec2.vector, v1.vector)

end

#  AlgebraVec – Vec
function inner_product(v1::AlgebraVec, v2::Vec)
    vec1 = get(v1.vecs, v2.subalgebra, nothing)
    vec1 === nothing && return 0.0 + 0im
    return dot(v2.vector, vec1.vector)

end

#  AlgebraVec – AlgebraVec
function inner_product(v1::AlgebraVec, v2::AlgebraVec)
    inner = 0.0 + 0im
    for (subalg, vec1) in v1.vecs
        inner += inner_product(vec1, v2)
    end
    return inner
end

# -------------------------------
# Subalgebra Irrep Base
# -------------------------------
abstract type AbstractSubalgebraIrrep end

mutable struct SubalgebraIrrep <: AbstractSubalgebraIrrep
    subalgebra::Tuple{Int,Int}          # (i,j)
    irrep::Int                           # k
    basis::Vector{Matrix{ComplexF64}}   # T_aijk_list
    d_a::Int
    irrep_space_shape::Tuple{Int,Int}   # shape of each basis matrix

    function SubalgebraIrrep(i::Int, j::Int, k::Int, T_aijk_list::Vector{Matrix{ComplexF64}})
        d_a = length(T_aijk_list)
        irrep_space_shape = size(T_aijk_list[1])
        new((i,j), k, T_aijk_list, d_a, irrep_space_shape)
    end
end

# Show and linear combination
function Base.show(io::IO, irrep::SubalgebraIrrep)
    println(io, "<SubalgebraIrrep(subalgebra=$(irrep.subalgebra), irrep=$(irrep.irrep), ",
            "basis_dim=$(length(irrep.basis)), irrep_space_dim=$(irrep.irrep_space_shape))>")
end

function linear_combination(irrep::SubalgebraIrrep, coeffs::Vector)
    sum(coeffs[i] * irrep.basis[i] for i in 1:length(coeffs))
end

# -------------------------------
# Subalgebra Element Block Base - (i,j) subalgebra element
# -------------------------------
abstract type AbstractSubAlgebraElementBlock end

function Base.show(io::IO, block::AbstractSubAlgebraElementBlock)
    println(io, "<SubAlgebraElementBlock (i=$(block.subalgebra[1]), j=$(block.subalgebra[2]), k=$(block.irrep)) | shape=$(size(block.LX))>")
end

function copy(block::AbstractSubAlgebraElementBlock)
    SubAlgebraElementBlock(block.subalgebra[1], block.subalgebra[2], block.irrep, copy(block.LX))
end

function dagger(block::AbstractSubAlgebraElementBlock)
    SubAlgebraElementBlock(block.subalgebra[2], block.subalgebra[1], block.irrep, conj.(block.LX'))
end

function herm_conj(block::AbstractSubAlgebraElementBlock)
    SubAlgebraElementBlock(block.subalgebra[1], block.subalgebra[2], block.irrep, conj.(block.LX))
end

function Base.:+(A::AbstractSubAlgebraElementBlock, B::AbstractSubAlgebraElementBlock)
    if A.subalgebra != B.subalgebra
        throw(ArgumentError("Cannot add blocks from different subalgebras"))
    end
    if A.irrep != B.irrep
        throw(ArgumentError("Cannot add blocks from different irreps"))
    end
    if size(A.LX) != size(B.LX)
        throw(ArgumentError("Cannot add blocks of different shapes"))
    end
    SubAlgebraElementBlock(A.subalgebra[1], A.subalgebra[2], A.irrep, A.LX + B.LX)
end

# -------------------------------
# Subalgebra Element Block L / R
# -------------------------------

mutable struct SubAlgebraElementBlockL <: AbstractSubAlgebraElementBlock
    subalgebra::Tuple{Int,Int}
    irrep::Int
    LX::Matrix{ComplexF64}
end

mutable struct SubAlgebraElementBlockR <: AbstractSubAlgebraElementBlock
    subalgebra::Tuple{Int,Int}
    irrep::Int
    LX::Matrix{ComplexF64}
end

SubAlgebraElementBlockL(i::Int, j::Int, k::Int, LX::Matrix{ComplexF64}) =
    SubAlgebraElementBlockL((i,j), k, LX)

SubAlgebraElementBlockR(i::Int, j::Int, k::Int, LX::Matrix{ComplexF64}) =
    SubAlgebraElementBlockR((i,j), k, LX)

# L-action on Vec
function Base.:*(block::SubAlgebraElementBlockL, v::AbstractVec)
    if block.subalgebra[2] != v.subalgebra[1]
        println("Cannot act: L_($(block.subalgebra)) on Vec in subalgebra $(v.subalgebra)")
        return VecZero(size(block.LX,1), block.irrep, (block.subalgebra[1], v.subalgebra[2]))
    end

    Vec(block.LX * v.vector, (block.subalgebra[1], v.subalgebra[2]), v.irrep_index)
end 

# R-action on Vec
function Base.:*(block::SubAlgebraElementBlockR, v::AbstractVec)
    if block.subalgebra[2] != v.subalgebra[2]
        println("Cannot act: R_($(block.subalgebra)) on Vec in subalgebra $(v.subalgebra)")
        return VecZero(size(block.LX,2), block.irrep, (v.subalgebra[1], block.subalgebra[1]))
    end
    
    Vec(  block.LX * v.vector, (v.subalgebra[1], block.subalgebra[1]), v.irrep_index)
end


# -------------------------------
# TubeAlgebra 
# -------------------------------

function linear_combination(irrep::SubalgebraIrrep, coeffs::Vector)
    sum(coeffs[i] * irrep.basis[i] for i in 1:length(coeffs))
end


mutable struct TubeAlgebra
    #dimension_dict::Dict{Tuple{Int,Int,Int}, Tuple{Int,Int,Int}}
    N_diag_blocks::Int
    d_algebra_squared::Int
    dim_ijk::Function
    f_ijk_sparse::Function
    left_cache::Dict{Tuple{Int,Int,Int}, SubalgebraIrrep}
    right_cache::Dict{Tuple{Int,Int,Int}, SubalgebraIrrep}
end


#function TubeAlgebra(dims::Dict, f_s::Function)
function TubeAlgebra(N_diag_blocks::Int, d_algebra_squared::Int, dims::Function, f_s::Function)
    #N_diag_blocks = maximum([k[1] for k in keys(dims)])
    #d_algebra_squared = sum([dims[(i,j,k)][2] for (i,j,k) in keys(dims) if i==j])
    TubeAlgebra(N_diag_blocks, d_algebra_squared, dims, f_s, 
                Dict{Tuple{Int,Int,Int},SubalgebraIrrep}(),
                Dict{Tuple{Int,Int,Int},SubalgebraIrrep}())
end



# -------------------------------
# Cached creation of L/R basis
# -------------------------------

function create_left_ijk_basis(t::TubeAlgebra, i,j,k)
    key = (i,j,k)
    if haskey(t.left_cache, key)
        return t.left_cache[key]
    end
    d_a, d_b, d_c = t.dim_ijk(i,j,k)
    f_dense = Array(t.f_ijk_sparse(i,j,k))
    T_a_ijk = [Matrix(transpose(f_dense[a,:,:])) for a in 1:d_a]

    ijk_irrep_basis = SubalgebraIrrep(i, j, k, T_a_ijk)
    t.left_cache[key] = ijk_irrep_basis
    return ijk_irrep_basis
end

function create_right_ijk_basis(t::TubeAlgebra, i,j,k)
    key = (i,j,k)
    if haskey(t.right_cache, key)
        return t.right_cache[key]
    end

    d_a, d_b, d_c = t.dim_ijk(i,j,k)
    f_dense = Array(t.f_ijk_sparse(i,j,k))
    T_b_ijk = [Matrix(f_dense[:,:,c]) for c in 1:d_c]
    ijk_irrep_basis = SubalgebraIrrep(i, j, k, T_b_ijk)
    t.right_cache[key] = ijk_irrep_basis

    return ijk_irrep_basis
end

# -------------------------------
# Random linear combinations
# -------------------------------

function random_left_linear_combination_ijk(t::TubeAlgebra, i,j,k; isHermitian=true, rng=Random.GLOBAL_RNG)
    block_basis = create_left_ijk_basis(t,i,j,k).basis
    d_a, d_b, d_c = t.dim_ijk(i,j,k)
    
    x_a = rand(d_a) .+ im .* rand(d_a)
    LX = sum(x_a[i] * block_basis[i] for i in eachindex(x_a))
    if isHermitian && i==j
        LX += conj!(LX')
        LX=Matrix(LX)/2.0
    end
    SubAlgebraElementBlockL(i, j, k, LX)
end

function random_right_linear_combination_ijk(t::TubeAlgebra, i,j,k; isHermitian=false, rng=Random.GLOBAL_RNG)
    block_basis = create_right_ijk_basis(t,i,j,k).basis
    d_a, d_b, d_c = t.dim_ijk(i,j,k)
    x_c = rand(d_c) .+ im .* rand(d_c)

    RX = Matrix(sum(block_basis[i] * x_c[i] for i in eachindex(x_c)))
    
    if isHermitian && i==j
        RX += RX'
        RX = Matrix(RX)
    end
    SubAlgebraElementBlockR(i, j, k, RX)
end

end # Module
