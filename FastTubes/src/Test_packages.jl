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
