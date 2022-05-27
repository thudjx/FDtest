using Tullio, FiniteDifferences

function HF_gen(k::Vector{Float64}, Hr_part::Matrix{ComplexF64}, qdict::Dict{Tuple{Int, Int}, Int}, moire::AbstractMoire, FTcut::Int, gf::GaugeField, Nt::Int)
    dim_spinor = size(collect(values(moire.hrdict))[1], 1)
    dim = length(qdict) * dim_spinor
    # Fourier components of H, Vx, Vy
    H_fourier = zeros(ComplexF64, dim, dim, 2*FTcut+1)
    T = 2 * π / gf.ω
    tlist = [range(0, T, length = Nt + 1);][1:end-1]
    for l in -2*FTcut:0
        # l-th component
        if l==0
            H = copy(Hr_part)
        else
            H = zeros(ComplexF64, dim, dim)
        end
        # integrate over t
        for t in tlist
            # construct matrices
            for (q, iq) in qdict
                q = collect(q)
                kp = k + q
                H[(iq - 1) * dim_spinor + 1:iq * dim_spinor, (iq - 1) * dim_spinor + 1:iq * dim_spinor] .+= hk(moire, kp, gf, t) * cis(l * gf.ω * t) * 1/Nt
            end
        end
        H_fourier[:,:,l+2*FTcut+1] = H
    end
    HF = zeros(ComplexF64, (2*FTcut+1)*dim, (2*FTcut+1)*dim)
    # upper right part of Ht, VFx, VFy
    for ir = 1:(2 * FTcut + 1) # row
        for ic = ir:(2 * FTcut + 1) # colomn
            HF[(ir-1)*dim+1:ir*dim, (ic-1)*dim+1:ic*dim] .+= H_fourier[:, :, ir - ic + 2 * FTcut + 1]
        end
    end
    # diagnol part of -iħ*∂/∂t
    for ir = 1:2 * FTcut + 1
        HF[(ir-1)*dim+1:ir*dim, (ir-1)*dim+1:ir*dim] .+= - gf.ω * (ir - FTcut - 1) * I(dim)
    end
    return Hermitian(HF)
end

function calc_df(k::Vector{Float64}, qdict::Dict{Tuple{Int, Int}, Int}, moire::AbstractMoire, FTcut::Int, gf::GaugeField, phi::Real, Nt::Int = 20, ZERO_CUT::Float64=1e-6)
    dim_spinor = size(collect(values(moire.hrdict))[1], 1)
    dim = length(qdict) * dim_spinor
    Hr_part = Hr(moire, qdict)
    occuprk = repeat([1., 0.], inner=(dim÷2))
    HF = HF_gen(k, Hr_part, qdict, moire, FTcut, gf, Nt)
    egvals, egvecs = eigen(HF)
    # choose branch of quasienergies
    center = (minimum(egvals) + maximum(egvals)) / 2
    ϵ = egvals[FTcut*dim+1:(FTcut+1)*dim]
    all(center-gf.ω/2 .<= ϵ .<= center+gf.ω/2) || error("egvals are not in one branch cut!")
    egarr = reshape(egvecs[:, FTcut*dim+1:(FTcut+1)*dim], dim, :, dim)
    # ---- df_nk ----
    # arbitary direction
    vcart_unit = [cos(deg2rad(phi)), sin(deg2rad(phi))]
    vdir = inv(moire.bscale) * vcart_unit
    J = 1/moire.bnorm * norm(vdir)
    vdir_unit = vdir ./ norm(vdir)
    function daf_gen(dk)
        return calc_f(k + (dk .* vdir_unit), Hr_part, qdict, moire, occuprk, FTcut, gf, Nt) * J
    end
    return daf_gen
end

function calc_f(k::Vector{Float64}, Hr_part::Matrix{ComplexF64}, qdict::Dict{Tuple{Int, Int}, Int}, moire::AbstractMoire, occuprk::Vector{Float64}, FTcut::Int, gf::GaugeField, Nt::Int)
    println("calc_f @kdir: $(k)")
    dim_spinor = size(collect(values(moire.hrdict))[1], 1)
    dim = length(qdict) * dim_spinor
    HF = HF_gen(k, Hr_part, qdict, moire, FTcut, gf, Nt)
    egvals, egvecs = eigen(HF)
    # choose branch of quasienergies
    center = (minimum(egvals) + maximum(egvals)) / 2
    ϵ = egvals[FTcut*dim+1:(FTcut+1)*dim]
    all(center-gf.ω/2 .<= ϵ .<= center+gf.ω/2) || error("egvals are not in one branch cut!")
    egarr = reshape(egvecs[:, FTcut*dim+1:(FTcut+1)*dim], dim, :, dim)
    Hpr = Hpr_gen(k, Hr_part, qdict, moire)
    egvecs_pr = eigvecs(Hpr)
    f = quench_occu(egvecs_pr, egarr, occuprk)
    # return [f; ϵ]
    return f
end

function quench_occu(egvecs_pr::Matrix{ComplexF64}, egarr::Array{ComplexF64, 3}, occuprk::Vector{Float64})
    @tullio g[n, α] := conj(egarr[i, p, n]) * egvecs_pr[i, α]
    @tullio occuk[n] := abs2(g[n, α]) * occuprk[α]
    return occuk
end

function calc_dfϵ(k::Vector{Float64}, qdict::Dict{Tuple{Int, Int}, Int}, moire::AbstractMoire, FTcut::Int, gf::GaugeField, phi::Real, fdm, Nt::Int = 20, ZERO_CUT::Float64=1e-6)
    dim_spinor = size(collect(values(moire.hrdict))[1], 1)
    dim = length(qdict) * dim_spinor
    Hr_part = Hr(moire, qdict)
    occuprk = repeat([1., 0.], inner=(dim÷2))
    HF = HF_gen(k, Hr_part, qdict, moire, FTcut, gf, Nt)
    egvals, egvecs = eigen(HF)
    # choose branch of quasienergies
    center = (minimum(egvals) + maximum(egvals)) / 2
    ϵ = egvals[FTcut*dim+1:(FTcut+1)*dim]
    all(center-gf.ω/2 .<= ϵ .<= center+gf.ω/2) || error("egvals are not in one branch cut!")
    egarr = reshape(egvecs[:, FTcut*dim+1:(FTcut+1)*dim], dim, :, dim)
    # ---- df_nk & dϵ_nk ----
    # arbitary direction
    vcart_unit = [cos(deg2rad(phi)), sin(deg2rad(phi))]
    vdir = inv(moire.bscale) * vcart_unit
    J = 1/moire.bnorm * norm(vdir)
    vdir_unit = vdir ./ norm(vdir)
    function daf_gen(dk)
        return calc_fϵ(k + (dk .* vdir_unit), Hr_part, qdict, moire, occuprk, FTcut, gf, Nt)
    end
    daf = fdm(daf_gen, 0) * J
    return daf
end

function calc_fϵ(k::Vector{Float64}, Hr_part::Matrix{ComplexF64}, qdict::Dict{Tuple{Int, Int}, Int}, moire::AbstractMoire, occuprk::Vector{Float64}, FTcut::Int, gf::GaugeField, Nt::Int)
    println("calc_f_ϵ @kdir: $(k)")
    dim_spinor = size(collect(values(moire.hrdict))[1], 1)
    dim = length(qdict) * dim_spinor
    HF = HF_gen(k, Hr_part, qdict, moire, FTcut, gf, Nt)
    egvals, egvecs = eigen(HF)
    # choose branch of quasienergies
    center = (minimum(egvals) + maximum(egvals)) / 2
    ϵ = egvals[FTcut*dim+1:(FTcut+1)*dim]
    all(center-gf.ω/2 .<= ϵ .<= center+gf.ω/2) || error("egvals are not in one branch cut!")
    egarr = reshape(egvecs[:, FTcut*dim+1:(FTcut+1)*dim], dim, :, dim)
    # calc f
    Hpr = Hpr_gen(k, Hr_part, qdict, moire)
    egvecs_pr = eigvecs(Hpr)
    f = quench_occu(egvecs_pr, egarr, occuprk)
    # return [f; ϵ]
    return f
end

#return the pristine hamiltonian
function Hpr_gen(k, Hr_part, qdict, moire)
    dim_spinor = size(collect(values(moire.hrdict))[1], 1)
    Htot = copy(Hr_part)
    for (q, iq) in qdict
        q = collect(q)
        kp = k + q
        Htot[(iq - 1) * dim_spinor + 1:iq * dim_spinor, (iq - 1) * dim_spinor + 1:iq * dim_spinor] += hk(moire, kp)
    end
    return Htot
end

function Hr(moire::AbstractMoire, qdict::Dict{Tuple{Int, Int}, Int})
    hr_dict = copy(moire.hrdict)
    dim_spinor = size(collect(values(hr_dict))[1], 1)
    if (0, 0) in keys(hr_dict)
        ma = kron(Matrix{ComplexF64}(I(length(qdict))), hr_dict[(0, 0)])
        pop!(hr_dict, (0, 0))
    else
        dim = length(qdict) * dim_spinor
        ma = zeros(ComplexF64, dim, dim)
    end
    qs = keys(qdict)
    for (q, iq) in qdict
        for (g, hrg) in hr_dict
            qp = (q[1] - g[1], q[2] - g[2])
            if qp in qs
                iq = qdict[q]
                iqp = qdict[qp]
                ma[(iq - 1) * dim_spinor + 1:iq * dim_spinor, (iqp - 1) * dim_spinor + 1:iqp * dim_spinor] = hrg
            end
        end
    end
    return ma
end