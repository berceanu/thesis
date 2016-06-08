using PyPlot
PyPlot.ioff()
push!(LOAD_PATH, "/home/berceanu/Development/topo-photon/anc/scripts")
import BP
using PyCall
@pyimport matplotlib.gridspec as gspec

# system parameters
const N = 45
const q = 11
const κ = 0.02
const γ = 0.001
const ν = collect(-3.45:0.001:-0.47)
prm = (1/q,γ,κ);
const r = 11 # points in MBZ
# Nk should be an odd multiple of q
const Nk = r*q # zero-padded system size = no of k points
const l = div(Nk-1,2)
const x = -l:l
const δk = 2π/Nk # resolution in mom space
# for fft, full BZ
const k = x * δk
randpmp(s::Int) = BP.randpmp(N; seed=s)
sprans = BP.Spectrum(ν,randpmp(1234), :symmetric, prm...)

# default exact spectrum, first 100 eigenvalues
exdef = BP.ExactStates(100, :symmetric, N, 1/q, κ)
# selected states for plotting
βs = [9, 20, 30,
      38, 59, 99]
ηs = βs + 1
sω0s = [exdef.νs[state]::Float64 for state in ηs] # 6 frequencies

# matplotlib parameters
matplotlib["rcParams"][:update](Dict("axes.labelsize" => 22,
                                     "axes.titlesize" => 20,
                                     "font.size" => 18,
                                     "legend.fontsize" => 14,
                                     "axes.linewidth" => 1.5,
                                     "font.family" => "serif",
                                     "font.serif" => "Computer Modern Roman",
                                     "xtick.labelsize" => 20,
                                     "xtick.major.size" => 5.5,
                                     "xtick.major.width" => 1.5,
                                     "ytick.labelsize" => 20,
                                     "ytick.major.size" => 5.5,
                                     "ytick.major.width" => 1.5,
                                     "text.usetex" => true,
                                     "figure.autolayout" => true))

fig, axes = plt[:subplots](2,3, figsize=(10,7.3))
for i = 1:3 #loop over columns
    # top row
    ax = axes[1,i]
    img = ax[:imshow](abs2(BP.myfft2(BP.getstate(sprans, sω0s[i]),
       k,k)), origin="upper", ColorMap("viridis"),
       interpolation="none",
       extent=[-π, π, -π, π])
    ax[:set_xticklabels]([])
    ax[:set_xticks]([-π,0,π])
    ax[:set_yticks]([-π,0,π])
    if i == 1 #leftmost panel
        ax[:set_ylabel](L"$k_y$")
        ax[:set_yticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
    else
        ax[:set_yticklabels]([])
    end
    #bottom row
    ax = axes[2,i]
    ax[:imshow](abs2(BP.myfft2(BP.getstate(sprans, sω0s[i+3]), k,k)),
       origin="upper", ColorMap("viridis"), interpolation="none",
       extent=[-π, π, -π, π])
    ax[:set_xlabel](L"$k_x$")
    ax[:set_xticks]([-π,0,π])
    ax[:set_yticks]([-π,0,π])
    ax[:set_xticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
    if i == 1 # leftmost panel
        ax[:set_ylabel](L"$k_y$")
        ax[:set_yticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
    else
        ax[:set_yticklabels]([])
    end
end
fig[:savefig]("../../figures/sym_ring.pdf", transparent=true, pad_inches=0.0, bbox_inches="tight")
plt[:close](fig)

# moving the trap
β = 4 # selected state
ω0 = exdef.νs[β + 1]
# trap positions
pos = [(0.0, 0.0) (2.0, 0.0) (5.5, 0.0) (11.0, 0.0);
       (0.0, 0.0) (2.0, 0.0) (5.5, 0.0) (11.0, 0.0);
       (0.0, 2.0) (0.0, 5.5) (0.0, 11.0) (11.0, 11.0)]
bz = Array(Float64, (3,4,Nk,Nk))
for col = 1:4, row = 1:3
    if row == 1 # landau gauge
        spran = BP.Spectrum([ω0], randpmp(1234), :landau, prm...,
           pos[row, col][1], pos[row, col][2])
    else # symmetric gauge
        spran = BP.Spectrum([ω0], randpmp(1234), :symmetric, prm...,
           pos[row, col][1], pos[row, col][2])
    end
    state = BP.getstate(spran, ω0)
    statefft = BP.myfft2(state, k,k)
    bz[row, col, :, :] = abs2(statefft)
end

fig = plt[:figure]()
gs = gspec.GridSpec(3, 4)
gs[:update](top=0.99, bottom=0.08, left=0.07, right=0.99, hspace=0.05)
axes = Array(PyObject, (3,4))
for col = 0:3, row = 0:2
    axes[row + 1, col + 1] = plt[:subplot](get(gs, (row, col)))
end
for col = 1:4, row = 1:3
    axes[row, col][:imshow](squeeze(bz[row, col, :, :], (1,2)),
                            origin="upper", ColorMap("viridis"),
                            interpolation="none", extent=[-π, π, -π, π])
    axes[row, col][:set_xticks]([-π,0,π])
    axes[row, col][:set_yticks]([-π,0,π])
end
# disable tick labels for bulk panels
for col = 2:4, row = 1:2
    axes[row, col][:set_yticklabels]([])
    axes[row, col][:set_xticklabels]([])
end
# left margin
for row = 1:2
    axes[row, 1][:set_yticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
    axes[row, 1][:set_xticklabels]([])
    axes[row, 1][:set_ylabel](L"$k_y$", labelpad=-9)
end
# bottom margin
for col = 2:4
    axes[3, col][:set_xticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
    axes[3, col][:set_yticklabels]([])
    axes[3, col][:set_xlabel](L"$k_x$", labelpad=-3)
end
# bottom left corner
axes[3, 1][:set_xticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
axes[3, 1][:set_yticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
axes[3, 1][:set_xlabel](L"$k_x$", labelpad=-3)
axes[3, 1][:set_ylabel](L"$k_y$", labelpad=-9)
fig[:savefig]("../../figures/fringe_trap.pdf", transparent=true, pad_inches=0.0, bbox_inches="tight")
plt[:close](fig)
