include("model.jl")
using .Model

include("ham.jl")
include("K-l.jl")

using DelimitedFiles

function main_df()
    # -------- model set; no need to read --------
    A_scale = 0.1
    omega_mult = 2.0
    theta = 1.1
    FTcut = 3
    qcut = 3.1
    Nt = 20
	q = 6 
    phi = 90 # the direction of partial derivative, in degree
    γ = 3.1214465773260254 # eV
    ω = γ * omega_mult # hbar*omega, eV
    A0 = A_scale * √3 / 2.46 # e*A0/ħ, 1/Å
    tbg = Moire(2.46, theta, [transpose([-1/2 -√3/2; 1 0]);], hrdict_gen())
    qdict = index_cb(tbg.bscale, tbg.fshell, qcut)
    gf = GaugeField(A0, ω)
    k = [0., 0.]
    df = calc_df(k, qdict, tbg, FTcut, gf, phi, Nt)
    # --------------------------------------------
    ##############################################
    #### define finite difference method here ####
    ##############################################
    q = 10
    adapt = 1
    fexp = 10
    fdm = central_fdm(q, 1, adapt=adapt, factor=10.0^fexp)
    # ---- calc derivative ----
    data = fdm(df, 0)
    # ---- save data ----
    open("df_q$(q)_adapt$(adapt)_fexp$(fexp).txt", "w") do io
        writedlm(io, data, ';')
    end
end

main_df()