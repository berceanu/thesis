using PyPlot
PyPlot.ioff()
push!(LOAD_PATH, "/home/berceanu/Development/topo-photon/anc/scripts")
import BP
import HH
using PyCall
@pyimport matplotlib.gridspec as gspec

# matplotlib parameters
matplotlib["rcParams"][:update](Dict("axes.labelsize" => 14,
                                     "axes.titlesize" => 20,
                                     "font.size" => 18,
                                     "legend.fontsize" => 14,
                                     "axes.linewidth" => 1.5,
                                     "font.family" => "serif",
                                     "font.serif" => "Computer Modern Roman",
                                     "xtick.labelsize" => 12,
                                     "xtick.major.size" => 5.5,
                                     "xtick.major.width" => 1.5,
                                     "ytick.labelsize" => 12,
                                     "ytick.major.size" => 5.5,
                                     "ytick.major.width" => 1.5,
                                     "text.usetex" => true,
                                     "figure.autolayout" => true))

# system parameters
sN=11 # true system size
# where we pump
n=5
m=0
# inverse lifetime
sγ=0.05
# strength of the trap
sκ = 0.2
q = 7
r = 11 # points in MBZ
# N should be an odd multiple of q
N = r*q # zero-padded system size
l = div(N-1,2)
x = -l:l
δk = 2π/N #resolution in mom space
k = x * δk
#generate all p values inside MBZ
xmbz = -div(r-1,2):div(r-1,2)
kxmbz = xmbz * δk

# exact spectrum, first 15 eigenvalues
exexp = BP.ExactStates(15, :landau, sN, 1/q, sκ)

# calculate HH energies E₀ and E₁
ladder = Array(Float64, (25,25,q))
HH.hhladder!(ladder, 1) # p = 1

#we filter state η
η = 8
β =  η-1
#at energy
sω0=exexp.νs[η]
#with a δ-like pump
δpmp(n₀::Int,m₀::Int) = BP.δpmp(sN; n0=n₀, m0=m₀)
P = δpmp(n,m)

# compute spectrum
sω1=-3.1
sω2=-1.0
dδ=0.001
#spectral range
ν =  sω1:dδ:sω2
sp = BP.Spectrum(collect(ν), P, :landau, 1/q, sγ, sκ)

# no pumping or dissipation
state = BP.getstate(exexp, η)
ψrex = abs2(state)
ψkex = abs2(BP.myfft2(state, k, k)) #77x77 Array{Float64,2} (0.0, 21.3)
ψkmbzex = BP.mbz(ψkex, r, q, kxmbz, k) #77x11 Array{Float64,2} (0., 1.24)
# with pumping
X = BP.getstate(sp, sω0)
ψr = abs2(X)
ψk = abs2(BP.myfft2(X, k, k))
ψkmbz = BP.mbz(ψk, r, q, kxmbz, k)

#analytical w.f. in MBZ
function getχ(Np,q,β)
    α = 1/q
    ydata = linspace(-π,π,Np)
    v = Array(Complex{Float64}, Np)
    for (i,y) in enumerate(ydata)
        v[i] = BP.χ(0.,y,α,β)
    end
    radical = sqrt(sqrt(2/q) / (2π*2^β * factorial(β) * 2π*α))
    radical .* v
end

χ = getχ(N,q,β)

# plotting
ics = 4
t1 = 0.97
b3 = 0.08
uai = 1.2
alfa = 1.5
el = (t1-b3)/(2 + 2alfa + 1/ics + 2/uai)
elp = alfa*el
b1 = t1 - el
t2 = b1 - el/uai
b2 = t2 - 2elp - el/ics
t3 = b2 - el/uai

fig = plt[:figure]()
#plot spectrum
gs1 = gspec.GridSpec(1, 1)
gs1[:update](top=t1, bottom=b1, left=0.225, right=0.8)
ax1 = plt[:subplot](get(gs1, (0, 0)))
ax1[:plot](sp.νs, sp.intensity, "k", linewidth = 2.0)
k₀ = 9 # the first k₀ states are in the n=0 HH band
for (ka, ν) in enumerate(exexp.νs)
    if (ka > k₀) && (mod(ka-k₀, 2) == 1)
        ax1[:axvline](x = ν, color="green", linestyle="-.", lw=2.0)
    else
        ka != 8 && ax1[:axvline](x = ν, color="orange",
           linestyle="dashed", lw=2.0)
        # we excluded the value where we filter
    end
end
ax1[:axvline](x = sω0, color="k", ls=":", lw=2.0)
ax1[:set_xlim](sp.νs[1], sp.νs[end])
ax1[:yaxis][:set_ticks]([0.,1.7,3.5])
ax1[:set_xlabel](L"$\omega_0/J$")
ax1[:set_ylabel](L"$\sum_{m,n} |a_{m,n}|^2$ (arb. units)", fontsize=10)
#plot w.f. in real and mom space
gs2 = gspec.GridSpec(2, 3)
gs2[:update](top=t2, bottom=b2, left=0.225, right=0.8)
ax2 = plt[:subplot](get(gs2, (0, 0)))
ax3 = plt[:subplot](get(gs2, (0, 1)))
ax4 = plt[:subplot](get(gs2, (0, 2)))
ax5 = plt[:subplot](get(gs2, (1, 0)))
ax6 = plt[:subplot](get(gs2, (1, 1)))
ax7 = plt[:subplot](get(gs2, (1, 2)))
axes = [ax2 ax3 ax4;
        ax5 ax6 ax7]
#real space
for (i,ψ) in enumerate((ψrex, ψr))
    ax = axes[i,1]
    img = ax[:imshow](ψ, origin="upper", ColorMap("viridis"),
                     interpolation="none",
                     extent=[-5.5, 5.5, -5.5, 5.5], aspect=1,
                     vmin=0, vmax=0.1)
    ax[:set_ylabel](L"$n$", labelpad=-5)
    ax[:set_xticks]([-4,0,4])
    ax[:set_yticks]([-4,0,4])
    if i == 2
        ax[:set_xlabel](L"$m$")
    else
        ax[:set_xticklabels]([])
    end
end
#momentum space
for (i,ψ) in enumerate((ψkex, ψk))
    ax = axes[i,2]
    img = ax[:imshow](ψ, origin="upper", ColorMap("viridis"),
                     interpolation="none",
                     extent=[-π, π, -π, π], aspect=1,
                     vmin=0, vmax=14)
    ax[:set_ylabel](L"$k_y$", labelpad=-8)
    ax[:set_xticks]([-π,0,π])
    if i == 2
        ax[:set_xlabel](L"$k_x$")
        ax[:set_xticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
    else
        ax[:set_xticklabels]([])
    end
    ax[:set_yticks]([-π,0,π])
    ax[:set_yticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
end
#MBZ
for (i,ψ) in enumerate((ψkmbzex, ψkmbz))
    ax = axes[i,3]
    img = ax[:imshow](ψ, origin="upper", ColorMap("viridis"),
                      interpolation="none",
                      extent=[-π/q, π/q, -π, π], aspect=1/q,
                      vmin=0, vmax=1)
    # vertical line to indicate slice
    ax[:axvline](x = 0.0, color = "k", linestyle = "-.", linewidth = 2.5)
    ax[:set_ylabel](L"$k_y$", labelpad=-8)
    ax[:set_xticks]([-π/q,0,π/q])
    if i == 2
        ax[:set_xlabel](L"$k_x^0$")
        ax[:set_xticklabels]([L"$-\pi/7$",L"$0$",L"$\pi/7$"])
    else
        ax[:set_xticklabels]([])
    end
    ax[:set_yticks]([-π,0,π])
    ax[:set_yticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
end
gs3 = gspec.GridSpec(1, 1)
gs3[:update](top=t3, bottom=b3, left=0.225, right=0.8)
ax8 = plt[:subplot](get(gs3, (0, 0)))
ax8[:plot](k, abs2(χ), "blue", ls="dotted", linewidth = 2.0)
ax8[:plot](k, reverse(ψkmbz[:,6]), "k", linewidth = 2.0)
ax8[:plot](k, reverse(ψkmbzex[:,6]), color="orange", ls="--", linewidth = 2.0)
ax8[:set_xlim](-π, π)
ax8[:set_xticks]([-π,-π/2,0,π/2,π])
ax8[:set_xticklabels]([L"$-\pi$",L"$-\pi/2$",L"$0$",L"$\pi/2$",L"$\pi$"])
ax8[:set_ylabel](L"$|\chi_7(0,k_y)|^2$")
ax8[:set_xlabel](L"$k_y$")
ax8[:yaxis][:set_ticks]([0.,0.2,0.4])
fig[:savefig]("../../figures/exp_fig.pdf", pad_inches=0.0, transparent=true, bbox_inches="tight")
plt[:close](fig)
