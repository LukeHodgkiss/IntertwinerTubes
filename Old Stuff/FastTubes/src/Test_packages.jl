module TestPackages

# Importing Paxkages
using SparseArrayKit: SparseArray, nonzero_values, nonzero_keys, nonzero_pairs # Ghent's sparse tensor helper
using MAT # reading in .Mat files
using LinearAlgebra # self explanatory
using TensorOperations # contractions
using CSV, DataFrames # For saving to file
using StaticArrays: SVector # Faster loops
using Arpack #spars 


# Importing Locally
include("SparseAlgebraObjects.jl")
using .SparseAlgebraObjects: TubeAlgebra, random_left_linear_combination_ijk, random_right_linear_combination_ijk, create_left_ijk_basis, create_right_ijk_basis
include("F_symbolTools.jl")
using .FSymbolTools
include("IntertwinerTools.jl")
using .IntertwinerTools     
include("Idempotents.jl")
using .Idempotents: find_idempotents


@time begin
# Read in F-symbol data
vars = matread("Luke_F.mat")
F = SparseArray{ComplexF64}(vars["F"])
quantum_dims = vec(vars["dim"])
size_dict = Dict(:module_label => size(F, 1),
                 :module_label_N => size(F, 1),
                 :module_label_M => size(F, 1),
                 :fusion_label => size(F, 2),
                 :multiplicity_label => size(F)[end])

# Build sparse block algebra
fusion_rules_M = make_fusion_rules(F, size_dict)
fusion_rules_N = make_fusion_rules(F, size_dict)
tubes_ij = make_tubes_ij(fusion_rules_M, fusion_rules_N)
f_ijk_sparse = make_f_ijk_sparse(F, F, quantum_dims, size_dict, tubes_ij)
dimension_dict = compute_dim_dict(size_dict, tubes_ij)
tubealgebra = TubeAlgebra(dimension_dict, f_ijk_sparse)

# Save Dictionary toCSV
df = DataFrame(i=Int[], j=Int[], k=Int[], d_a=Int[], d_b=Int[], d_c=Int[])
for ((i,j,k), (d_a,d_b,d_c)) in dimension_dict
    push!(df, (i,j,k,d_a,d_b,d_c))
end
sorted_df = sort(df, [:i, :j, :k])
CSV.write("dimension_dict_output.csv", sorted_df)

# Save f_ijk to csv
df = DataFrame(i=Int[], j=Int[], k=Int[], a=Int[], b=Int[], c=Int[], val=ComplexF64[])
for i in 1:16
    for j in 1:16
        for k in 1:16
            for (I, val) in f_ijk_sparse(i,j,k).data
                (a,b,c) = Tuple(I)
                push!(df, (i,j,k,a,b,c, val))
            end
        end
    end
end
sorted_df = sort(df, [:i, :j, :k,:a, :b, :c])
CSV.write("f_ijk_sparse_output.csv", sorted_df)

# Calculate idempotents
idempotents_dict = find_idempotents(tubealgebra)
@show idempotents_dict

"""#function construct_irreps(algebra::TubeAlgebra, irrep_projectors::Vector{Dict}, size::Dict, tubes_ij::Function, q_dim_a)
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

                M1, N1 = index_to_tuple(i, (size[:module_label], size[:module_label]))
                M2, N2 = index_to_tuple(j, (size[:module_label], size[:module_label]))

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
    @inbounds for (k, v) in dok
        dok_CI[CartesianIndex(Tuple(k))] = v # Key k is an SVector - maybe change but supposedly very good for insertion
    end

    ω = SparseArray{ComplexF64,10}(dok_CI, shape) 
    return ω
end"""

ω = construct_irreps(tubealgebra, idempotents_dict, size_dict, tubes_ij, quantum_dims)

# Save ω to CSV
df = DataFrame(irrep=Int[], M1=Int[], Y=Int[], N2=Int[], N1=Int[], M2=Int[], row=Int[], n=Int[], m=Int[], col=Int[], val=ComplexF64[])
for (I, val) in ω.data
    (irrep, M1, Y, N2, N1, M2, row, n, m, col) = Tuple(I)
    push!(df, (irrep, M1, Y, N2, N1, M2, row, n, m, col, val))
end
sorted_df = sort(df, [:irrep, :M1, :Y, :N2, :N1, :M2, :row, :n, :m, :col])
CSV.write("bim_mod_funct.csv", sorted_df)



end # Module



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

function make_dim_dict(size_dict::Dict{Symbol, Int}, tubes_ij)
    cache = Dict{Tuple{Int,Int,Int}, Tuple{Int,Int,Int}}()

    function dim_ijk(i,j,k)
        key = (i,j,k)
        if haskey(cache, key)
            return cache[key]
        end

        N_Mcat = size_dict[:module_label_N]
        N_Ncat = size_dict[:module_label_M]
        M_1, N_1 = index_to_tuple(i, (size_dict[:module_label_M], size_dict[:module_label_N]))
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


#using .SparseAlgebraObjects: TubeAlgebra, create_left_ijk_basis

# --- Construct module functor intertwiner
"""
Return a DOK (dict) mapping SVector{10,Int} => ComplexF64
for the irreducible representation tensor.
"""
#=
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
                    # refix here

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
=#
#=
function construct_irreps(algebra, irrep_projectors, size_dict, tube_map_inv::Dict, q_dim_a, create_left_ijk_basis)
    dok = Dict{SVector{10,Int}, ComplexF64}()
    #sizehint!(dok,)
    maxvals = zeros(Int, 10)   

    N_irrep = length(irrep_projectors)
    N_blocks = algebra.N_diag_blocks

    @inbounds for irrep in 1:N_irrep
        k = first(keys(irrep_projectors[irrep]))[2]

        iproj = irrep_projectors[irrep]

        for i in 1:N_blocks
            for j in 1:N_blocks
                if algebra.dim_ijk(i,j,k) == nothing; continue; end
                if !(haskey(iproj,(i,k)) && haskey(iproj,(j,k))); continue; end
                
                Q_ik = iproj[(i,k)]
                Q_jk = iproj[(j,k)]

                T_a = create_left_ijk_basis(algebra, i,j,k).basis
                d_a = length(T_a)
                    
                M1, N1 = index_to_tuple(i, (size_dict[:module_label_N], size_dict[:module_label_M]))
                M2, N2 = index_to_tuple(j, (size_dict[:module_label_N], size_dict[:module_label_M]))
                
                for a in 1:d_a
                    Y, m, n = tube_map_inv[(M1, M2, N2, N1, a)]
                    scale = 1.0 / sqrt(Float64(q_dim_a[Y]))

                    #@show i, j, size(adjoint(Q_ik)), size(T_a[a]), size(Q_jk)
                    ρ = scale * (adjoint(Q_ik) * T_a[a] * Q_jk)
                    # refix here

                    nnz_indices = findall(!iszero, ρ)
                    vals = ρ[nnz_indices]

                    for t in eachindex(vals)
                        row, col = Tuple(nnz_indices[t])
                        key = SVector{10,Int}(irrep, M1, Y, N2, N1, M2, row, n, m, col)
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
    #=
    dok_CI = Dict{CartesianIndex{10}, ComplexF64}()
    @inbounds for (k, v) in dok
        dok_CI[CartesianIndex(Tuple(k))] = v # Key k is an SVector - maybe change but supposedly very good for insertion
    end
    =#
    dok_CI = Dict(CartesianIndex(Tuple(k)) => v for (k, v) in dok)

    ω = SparseArray{ComplexF64,10}(dok_CI, shape) 
    return ω
end # construct_irreps
=#
#=
function construct_irreps(algebra, irrep_projectors, size_dict, tube_map_inv::Dict, q_dim_a, create_left_ijk_basis, F)
    # 1. Concrete type for DOK and sizehint to avoid re-allocations
    dok = Dict{SVector{10,Int}, ComplexF64}()
    sizehint!(dok, length(F.data))
    
    maxvals = zeros(Int, 10)   
    N_irrep = length(irrep_projectors)
    N_blocks = algebra.N_diag_blocks
    
    mod_N = size_dict[:module_label_N]
    mod_M = size_dict[:module_label_M]

    @inbounds for irrep in 1:N_irrep
        iproj = irrep_projectors[irrep]
        k = first(keys(iproj))[2]

        for i in 1:N_blocks
            Q_ik = get(iproj, (i, k), nothing)
            isnothing(Q_ik) && continue
            adj_Q_ik = adjoint(Q_ik) 
            
            M1, N1 = index_to_tuple(i, (mod_N, mod_M))

            for j in 1:N_blocks
                isnothing(algebra.dim_ijk(i, j, k)) && continue
                Q_jk = get(iproj, (j, k), nothing)
                isnothing(Q_jk) && continue
                
                M2, N2 = index_to_tuple(j, (mod_N, mod_M))
                
                T_a = create_left_ijk_basis(algebra, i, j, k).basis
                for a in eachindex(T_a)
                    Y, m, n = tube_map_inv[(M1, M2, N2, N1, a)]
                    ρ = (1.0 / sqrt(q_dim_a[Y])) .* (adj_Q_ik * T_a[a] * Q_jk)

                    for idx in CartesianIndices(ρ)
                        val = ρ[idx]
                        if !iszero(val)
                            row, col = idx.I
                            key = SVector{10,Int}(irrep, M1, Y, N2, N1, M2, row, n, m, col)
                            dok[key] = val

                            for d in 1:10
                                @inbounds maxvals[d] = max(maxvals[d], key[d])
                            end
                        end
                    end
                end
            end
        end
    end

    shape = Tuple(maxvals)
    dok_CI = Dict(CartesianIndex(Tuple(k)) => v for (k, v) in dok)
    return SparseArray{ComplexF64,10}(dok_CI, shape)
end
=#


fusion_rules_M = make_fusion_rules(F, size_dict)
fusion_rules_N = make_fusion_rules(F, size_dict)
tubes_ij = make_tubes_ij(fusion_rules_M, fusion_rules_N)
f_ijk_sparse = make_f_ijk_sparse(F, conj!(F), quantum_dims, size_dict, tubes_ij)
#@show length(f_ijk_sparse(12,12,1))
#dimension_dict = compute_dim_dict(size_dict, tubes_ij)
dimension_dict = make_dim_dict(size_dict, tubes_ij)


N_diag_blocks = size_dict[:module_label_N]*size_dict[:module_label_M]
#d_algebra_squared = 27863#N_diag_blocks*N_diag_blocks # calculate by contracting fusion tensor then summing up all the entries
d_algebra = 27863  #N_diag_blocks*N_diag_blocks # calculate by contracting fusion tensor then summing up all the entries
tubealgebra = TubeAlgebra(N_diag_blocks, d_algebra, dimension_dict, f_ijk_sparse)
println("This is no longer the way")


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
                        
                    end
                end
            end
        end
    end
    
    return tube_map, tube_map_shape, tube_map_inv
end
