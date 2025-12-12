
module Saving_Stuff

export  save_dim_dict, save_f_ijk, save_ω

using CSV, DataFrames

# Save Dictionary to CSV
function save_dim_dict(dimension_dict)
    df = DataFrame(i=Int[], j=Int[], k=Int[], d_a=Int[], d_b=Int[], d_c=Int[])
    for ((i,j,k), (d_a,d_b,d_c)) in dimension_dict
        push!(df, (i,j,k,d_a,d_b,d_c))
    end
    sorted_df = sort(df, [:i, :j, :k])
    CSV.write("dimension_dict_output.csv", sorted_df)
end

# Save f_ijk to csv
function save_f_ijk(f_ijk_sparse)
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
end

# Save ω to CSV
function save_ω(ω)
    df = DataFrame(irrep=Int[], M1=Int[], Y=Int[], N2=Int[], N1=Int[], M2=Int[], row=Int[], n=Int[], m=Int[], col=Int[], val=ComplexF64[])
    for (I, val) in ω.data
        (irrep, M1, Y, N2, N1, M2, row, n, m, col) = Tuple(I)
        push!(df, (irrep, M1, Y, N2, N1, M2, row, n, m, col, val))
    end
    sorted_df = sort(df, [:irrep, :M1, :Y, :N2, :N1, :M2, :row, :n, :m, :col])
    CSV.write("bim_mod_funct.csv", sorted_df)
end

end # Module