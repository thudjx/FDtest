using LinearAlgebra, PhysicalConstants.CODATA2018, Unitful

trunc3(x::Float64) = trunc(x, digits=3)

get_hbarv() = -6.65 # Å*eV

get_a0() = 2.46 # Å

function hrdict_gen()
    z = exp(im * 2 * π / 3)
    zc = exp(-im * 2 * π / 3)
    u = 0.11 * 0.85 + 0 * im
    up = 0.11 + 0 * im
    t0 = [u up; up u]
    Hr0 = kron([0 1; 1 0], t0)
    t = [u z * up; zc * up u]
    tc = [u zc * up; z * up u]
    Hr2 = kron([0 1; 0 0], tc)
    Hr3 = kron([0 1; 0 0], t)
    Hr5 = kron([0 0; 1 0], tc)
    Hr6 = kron([0 0; 1 0], t)
    return Dict((0, 0) => Hr0, (1, 0) => Hr5, (1, 1) => Hr6, (-1, 0) => Hr2, (-1, -1) => Hr3)
end

function hk(moire::Moire, k::Vector{Float64})
    hbarv = get_hbarv() # Å*eV
    m1 = 0.01
    m2 = 0.01
    #Y choice
    K1 = [-1/3; 1/3]
    K2 = [1/3; 2/3]
    k1_c = moire.bnorm * moire.bscale *  (k - K1)
    k2_c = moire.bnorm * moire.bscale *  (k - K2)
    σs = [[0 1; 1 0], [0 -im; im 0]]
    σz = [1 0; 0 -1]
    return kron([1 0; 0 0], hbarv * sum(k1_c .* σs) .+ (m1 .* σz)) .+
           kron([0 0; 0 1], hbarv * sum(k2_c .* σs) .+ (m2 .* σz))
end

function hk(moire::Moire, k::Vector{Float64}, gf::GaugeField, t::Float64)
    hbarv = get_hbarv() # Å*eV
    m1 = 0.01
    m2 = 0.01
    #Y choice
    K1 = [-1/3; 1/3]
    K2 = [1/3; 2/3]
    k1_c = moire.bnorm * moire.bscale *  (k - K1) .+ gf(t)
    k2_c = moire.bnorm * moire.bscale *  (k - K2) .+ gf(t)
    σs = [[0 1; 1 0], [0 -im; im 0]]
    σz = [1 0; 0 -1]
    return kron([1 0; 0 0], hbarv * sum(k1_c .* σs) .+ (m1 .* σz)) .+
           kron([0 0; 0 1], hbarv * sum(k2_c .* σs) .+ (m2 .* σz))
end

function vx(moire::Moire, k::Vector{Float64}, gf::GaugeField, t::Float64)
    hbarv = get_hbarv() # Å*eV
    return hbarv * kron([1. + 0im 0; 0 1], [0 1; 1. 0])
end

function vy(moire::Moire, k::Vector{Float64}, gf::GaugeField, t::Float64)
    hbarv = get_hbarv() # Å*eV
    return hbarv * kron([1. 0; 0 1], [0 -im; im 0])
end

(gf::GaugeField)(t::Float64) = gf.A0 .* [-sin(gf.ω * t); cos(gf.ω * t)]

function index_cb(b_scale, fshell, qcut::Float64 = 4.0, tol::Float64 = 10^-4)
    d = Dict((0, 0) => 1)
    index = 2
    shell = [(0, 0)]
    g = fshell
    while length(shell) > 0
        shell_copy = shell[:]
        shell = []
        for s in shell_copy
            q_cand = collect(s) .+ g
            qc_cand = b_scale * q_cand
            for ig in 1:size(q_cand, 2)
                if dot(qc_cand[:, ig], qc_cand[:, ig]) <= qcut^2 +tol
                    key = Tuple(q_cand[:, ig])
                    if !(key in keys(d))
                        d[key] = index
                        index += 1
                        push!(shell, key)
                    end
                end
            end
        end
    end
    return d
end