module Model

export AbstractMoire, Moire, hk, vx, vy, GaugeField

using LinearAlgebra

abstract type AbstractMoire end

struct GaugeField
    A0::Float64#vF*e*A0, eV
    ω::Float64#hbar*omega, eV
end

function hk(moire::AbstractMoire, k::Matrix{<:Number})
    error("hkp for single layer is not defined!")
end

function vx(moire::AbstractMoire, k::Matrix{<:Number})
    error("velocity operator vx is not defined")
end

function vy(moire::AbstractMoire, k::Matrix{<:Number})
    error("velocity operator vy is not defined")
end

function hk(moire::AbstractMoire, k::Matrix{<:Number}, gf::GaugeField, t::Float64)
    error("driven hkp for single layer is not defined!")
end

function vx(moire::AbstractMoire, k::Matrix{<:Number}, gf::GaugeField, t::Float64)
    error("driven velocity operator vx is not defined")
end

function vy(moire::AbstractMoire, k::Matrix{<:Number}, gf::GaugeField, t::Float64)
    error("driven velocity operator vy is not defined")
end

struct Moire <: AbstractMoire
    theta::Float64
    bnorm::Float64
    bscale::Matrix{Float64}
    hrdict::Dict{Tuple{Int, Int}, Matrix{ComplexF64}}
    fshell::Matrix{Int}
    hfshell::Matrix{Int}
end

function Moire(a0::Float64, theta::Float64, bscale::Matrix{Float64}, hrdict::Dict{Tuple{Int, Int}, Matrix{ComplexF64}})
    bnorm = 8*sin(theta*π/180/2)*π/(√3*a0)
    temp = fs_hf(bscale)
    fshell = temp[1]
    hfshell = temp[2]
    return Moire(theta, bnorm, bscale, hrdict, fshell, hfshell)
end

function finduf(ls::Array, equival::Function)
    uf = [-1 for i in 1:length(ls)]
    for i = 1:length(ls)
        for j = i+1:length(ls)
            if equival(ls[i], ls[j])
                uf[j] = i
            end
        end
    end
    return uf
end

function dev_by_uf(ls, uf)
    classes = []
    for i = 1:length(ls)
        if uf[i] == -1
            push!(classes, [ls[i]])
        else
            for c in classes
                if ls[uf[i]] in c
                    push!(c, ls[i])
                end
            end
        end
    end
    return classes
end

function fs_hf(basis::Matrix{Float64}, tol::Float64 = 10^-4)
    function mydot(v::Vector{<:Number}, w::Vector{<:Number})
        return dot(v, transpose(basis) * basis, w)
    end
    function eq(x::Vector{<:Number}, y::Vector{<:Number})
        return abs(mydot(x, x) - mydot(y, y)) < tol
    end
    cands = [[i,j] for i = -1:1 for j = -1:1]
    uf = finduf(cands, eq)
    shells = dev_by_uf(cands, uf)
    sort!(shells, by = (x) -> mydot(x[1], x[1]))
    fs = shells[2]

    function eq1(x, y)
        return mydot(x .+ y, x .+ y) < tol
    end
    uf = finduf(fs, eq1)
    temp = dev_by_uf(fs, uf)
    hf = [x[1] for x in temp]
    return hcat(fs...), hcat(hf...)
end

# function (gf::GaugeField)(t::Float64)
#     return gf.A0 * [-sin(gf.ω * t), cos(gf.ω * t)]
# end

end # module