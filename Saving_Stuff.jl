
module Saving_Stuff

export  save_dim_dict, save_f_ijk, save_ω

using CSV, DataFrames

# Save Dictionary to CSV
function save_dim_dict(dimension_dict) 
    df = DataFrame(i=Int[], j=Int[], k=Int[], d_a=Int[], d_b=Int[], d_c=Int[])
  
    for i in 1:16
        for j in 1:16
            for k in 1:16
                #@show i, j, k
                if dimension_dict(i,j,k) !== nothing
                    d_a,d_b,d_c = dimension_dict(i,j,k)
                    push!(df, (i,j,k,d_a,d_b,d_c))
                end
            end
        end
    end
    sorted_df = sort(df, [:i, :j, :k])
    CSV.write("dimension_dict_output.csv", sorted_df)
end

# Save f_ijk to csv
function save_f_ijk(algebra)
    df = DataFrame(i=Int[], j=Int[], k=Int[], a=Int[], b=Int[], c=Int[], val=ComplexF64[])
    for i in 1:algebra.N_diag_blocks
        for j in 1:algebra.N_diag_blocks
            for k in 1:algebra.N_diag_blocks
                #@show i, j, k
                if algebra.dim_ijk(i,j,k) !== nothing
                    for (I, val) in algebra.f_ijk_sparse(i,j,k).data 
                        (a,b,c) = Tuple(I)
                        push!(df, (i,j,k,a,b,c, val))
                    end
                end
            end
        end
    end
    sorted_df = sort(df, [:i, :j, :k,:a, :b, :c])
    CSV.write("f_ijk_sparse_output_RrepA4PepA4.csv", sorted_df)
end

# Save ω to CSV
function save_ω(ω)
    df = DataFrame(irrep=Int[], M1=Int[], Y=Int[], N2=Int[], N1=Int[], M2=Int[], row=Int[], n=Int[], m=Int[], col=Int[], val=ComplexF64[])
    for (I, val) in ω.data
        (irrep, M1, Y, N2, N1, M2, row, n, m, col) = Tuple(I)
        push!(df, (irrep, M1, Y, N2, N1, M2, row, n, m, col, val))
    end
    sorted_df = sort(df, [:irrep, :M1, :Y, :N2, :N1, :M2, :row, :n, :m, :col])
    CSV.write("CG.csv", sorted_df)
end

end # Module