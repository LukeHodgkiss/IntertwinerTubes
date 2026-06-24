module CoordCompress

export compress_nd, build_sparse_compressed, compress_sparse, compress_sparse_matrix

using SparseArrayKit

function compress_nd(coords::Vector{NTuple{N,T}}) where {N,T}

    maps = [Dict{T,Int}() for _ in 1:N]
    counters = ones(Int, N)

    compressed = Vector{NTuple{N,Int}}(undef, length(coords))

    for i in eachindex(coords)

        c = coords[i]
        newc = ntuple(d -> begin
            m = maps[d]

            if !haskey(m, c[d])
                m[c[d]] = counters[d]
                counters[d] += 1
            end

            m[c[d]]
        end, N)

        compressed[i] = newc
    end

    return compressed, maps
end


# coords = [(100,42,7),(5000,42,9),(100,99,7)]
# vals = [1.0,2.0,3.0]

# compressed, _ = compress_nd(coords)

# tensor = SparseArray{Float64}(undef, 2,2,2)

# for (c,v) in zip(compressed, vals)
#     tensor[c...] = v
# end

# original = first(keys(maps[1]))

function compress_sparse(A)

    N = ndims(A)

    maps = [Dict{Int,Int}() for _ in 1:N]
    invmaps = [Int[] for _ in 1:N]

    counters = ones(Int, N)

    # ---- pass 1: build maps ----
    for (I, _) in nonzero_pairs(A)

        for d in 1:N
            orig = I[d]
            m = maps[d]

            if !haskey(m, orig)

                comp = counters[d]
                counters[d] += 1

                m[orig] = comp
                push!(invmaps[d], orig)

            end
        end
    end

    dims = counters .- 1
    @show dims

    # ---- allocate compressed tensor ----
    B = SparseArray{eltype(A)}(undef, dims...)

    # ---- pass 2: fill tensor ----
    for (I, v) in nonzero_pairs(A)

        newI = ntuple(d -> maps[d][Tuple(I)[d]], N)
        B[newI...] = v

    end

    println(size(A))
    println(size(B))    

    return B, maps, invmaps
end

function compress_sparse_matrix(P)

    
    used = Set{Int}()

    for (I, _) in nonzero_pairs(P)
        push!(used, I[1])
        push!(used, I[2])
    end

    used_list = sort!(collect(used))

    map = Dict{Int,Int}()
    invmap = Vector{Int}(undef, length(used_list))

    for (i, orig) in enumerate(used_list)
        map[orig] = i
        invmap[i] = orig
    end

    n = length(used_list)

    B = SparseArray{eltype(P)}(undef, n, n)

    for (I, v) in nonzero_pairs(P)
        r = map[I[1]]
        c = map[I[2]]
        B[r,c] = v
    end

    return B, map, invmap
end

end # module