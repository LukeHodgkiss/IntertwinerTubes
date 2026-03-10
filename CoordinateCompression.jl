using SparseArrayKit

function compress_nd(coords::Vector{NTuple{N,T}}) where {N,T}

    # One map per dimension
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


coords = [(100,42,7),(5000,42,9),(100,99,7)]
vals = [1.0,2.0,3.0]

compressed, _ = compress_nd(coords)

tensor = SparseArray{Float64}(undef, 2,2,2)

for (c,v) in zip(compressed, vals)
    tensor[c...] = v
end

original = first(keys(maps[1]))

using SparseArrayKit

function build_sparse_compressed(coords::Vector{NTuple{N,T}}, vals) where {N,T}

    maps = [Dict{T,Int}() for _ in 1:N]
    counters = ones(Int, N)

    tensor = SparseArray{eltype(vals)}()

    for (c,v) in zip(coords, vals)

        newc = ntuple(d -> begin
            m = maps[d]

            if !haskey(m, c[d])
                m[c[d]] = counters[d]
                counters[d] += 1
            end

            m[c[d]]
        end, N)

        tensor[newc...] = v
    end

    return tensor, maps
end