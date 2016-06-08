using PyPlot
PyPlot.ioff()
push!(LOAD_PATH, "/home/berceanu/Development/topo-photon/anc/scripts")
using PyCall
@pyimport mpl_toolkits.axes_grid1.inset_locator as axloc
@pyimport matplotlib.gridspec as gspec
import BP

# system parameters
const N = 45
const q = 11
const κ = 0.02
const γ = 0.001
const ν = collect(linspace(-3.45,-2.47,981))

const r = 11 # points in MBZ

# Nk should be an odd multiple of q
const Nk = r*q # zero-padded system size = no of k points
const l = div(Nk-1,2)
const x = -l:l
const δk = 2π/Nk # resolution in mom space
const k = x * δk

# full plot range, both in x and y
const xm = collect(-div(N-1,2):div(N-1,2))
# zoom in
const edge = 10
const st = findin(xm, -edge)[1]
const en = findin(xm,  edge)[1]

# exact spectrum, first 29 eigenvalues
exstates = BP.ExactStates(29, :landau, N, 1/q, κ)
# half the level spacing
const hf = (exstates.νs[2] - exstates.νs[1])/2

#for plotting filter markers
βlan = [0,2,4,6]
βsym = [0,1,9,20]
βreal = [3,6,15,26]
#we filter state η
ηlan = βlan + 1
ηsym = βsym + 1
ηreal = βreal + 1
#at energy
sω0lan = [exstates.νs[state]::Float64 for state in ηlan]
sω0sym = [exstates.νs[state]::Float64 for state in ηsym]
sω0real= [exstates.νs[state]::Float64 for state in ηreal]
# energy boundaries for state with β=4
ω₁= exstates.νs[4] - 0.005
ω₂= exstates.νs[6] + 0.005

δpmp(n₀::Int,m₀::Int) = BP.δpmp(N; n0=n₀, m0=m₀)
gausspmp(n₀::Int,m₀::Int) = BP.gausspmp(N; σ=1., n0=n₀, m0=m₀)
homopmp() = BP.homopmp(N)
randpmp(s::Int) = BP.randpmp(N; seed=s) #1234

prm = (1/q,γ,κ);

spδl = BP.Spectrum(ν,δpmp(5,5), :landau, prm...)
spgaussl = BP.Spectrum(ν,gausspmp(5,5), :landau, prm...)
sphoml = BP.Spectrum(ν,homopmp(), :landau, prm...)

# calculating all real space wfs
ψ = Array(Float64, (N, N, length(βreal)))
for (i,ω) in enumerate(sω0real)
    ψ[:,:,i] = abs2(BP.getstate(spδl, ω))
end

spgausss = BP.Spectrum(ν,gausspmp(5,5), :symmetric, prm...)
sphoms = BP.Spectrum(ν,homopmp(), :symmetric, prm...)

# calculating all mom space wfs
ψL = Array(Float64, (Nk, Nk, length(βlan)))
ψS = Array(Float64, (Nk, Nk, length(βsym)))
for (i,ω) in enumerate(sω0lan)
    ψL[:,:,i] = abs2(BP.myfft2(BP.getstate(sphoml, ω), k, k))
end
for (i,ω) in enumerate(sω0sym)
    ψS[:,:,i] = abs2(BP.myfft2(BP.getstate(spgausss, ω), k, k))
end

## averaging over 100 random phase distributions ##
intvec = zeros(Float64, length(ν));
A = spzeros(Complex{Float64}, N^2,N^2);
for j=1:100
    P=randpmp(j)
    for (i,ω) in enumerate(ν)
        BP.buildham_landau!(A, N,1/q,κ,γ,ω)
        intvec[i] += sum(abs2(A\P))
    end
end
sprandl = intvec./100;

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

ics=3.0
el = 0.86/(3/ics + 5)
b1 = 0.97-el
t2 = b1 - el/ics
b2 = t2 - 2el
t3 = b2 - el/ics

fig = plt[:figure]()
gs1 = gspec.GridSpec(1, 1)
gs1[:update](top=0.97, bottom=b1, right=0.98, left=0.16)
ax1 = plt[:subplot](get(gs1, (0, 0)))
ax1[:plot](spδl.νs,spδl.intensity,"k")
for (i,ω) in enumerate(sω0real)
    ax1[:axvline](x = ω, color="orange", ls="dotted", lw=2.0)
    ax1[:text](ω + hf/4, 8e3, string(βreal[i]))
end
ax1[:set_ylim](0, 1e4)
ax1[:yaxis][:set_ticks]([0, 1e4])
ax1[:yaxis][:set_ticklabels]([L"$0$", L"$10^4$"])
ax1[:text](ν[1] + hf, 5e3, "(a)")
gs2 = gspec.GridSpec(2, 1)
gs2[:update](top=t2, bottom=b2, hspace=0.0, right=0.98, left=0.16)
ax2 = plt[:subplot](get(gs2, (0, 0)))
ax3 = plt[:subplot](get(gs2, (1, 0)))
ax2[:plot](spgaussl.νs,spgaussl.intensity,"k") 
ax2[:set_ylim](0, 3.5e3)
ax2[:yaxis][:set_ticks]([0, 3.5e3])
ax2[:yaxis][:set_ticklabels]([L"$0$", L"$3.5\!\! \times\!\! 10^3$"])
ax2[:text](ν[1] + hf, 1.75e3, "(b)")
ax3[:plot](spgausss.νs,spgausss.intensity, color="green", ls="dashed",
   linewidth=1.5)
for (i,ω) in enumerate(sω0sym)
    ax3[:axvline](x = ω, color="orange", ls="dotted", lw=2.0)
    ax3[:text](ω + hf/4, 1.6e4, string(βsym[i]))
end
ax3[:set_ylim](0, 2e4)
ax3[:yaxis][:set_ticks]([0, 1e4])
ax3[:yaxis][:set_ticklabels]([L"$0$", L"$10^4$"])
gs3 = gspec.GridSpec(2, 1)
gs3[:update](top=t3, bottom=0.11, hspace=0.4, right=0.98, left=0.16)
ax4 = plt[:subplot](get(gs3, (0, 0)))
ax4[:plot](sphoml.νs,sphoml.intensity,"k")
ax4[:plot](sphoms.νs,sphoms.intensity, color="green", ls="dashed",
   linewidth=1.5)
# insert with zoom of peak β=4
axins = axloc.inset_axes(ax4,
                        width="30%", # width = 30% of parent_bbox
                        height="50%",
                        loc=9) # located at upper middle part
# plot same thing as in parent box
axins[:plot](sphoms.νs,sphoms.intensity, color="green", ls="dashed",
   linewidth=1.5)
# but set much narrower limits
axins[:set_xlim]([ω₁, ω₂])
axins[:xaxis][:set_ticks]([ω₁, ω₂])
axins[:xaxis][:set_ticklabels]([string(round(ω₁,2)),
   string(round(ω₂,2))], fontsize=8)
axins[:set_ylim]([1e3, 1e4])
axins[:yaxis][:set_ticks]([1e3, 1e4])
axins[:yaxis][:set_ticklabels]([L"$10^3$", L"$10^4$"], fontsize=8)
# draw vertical lines at position of every exact eigenstate
axins[:axvline](x = exstates.νs[5], color="orange", ls="dotted", lw=2.0)
axins[:text](exstates.νs[5] + hf/8, 8e3, string(4), fontsize=8)
# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
axloc.mark_inset(ax4, axins, loc1=2, loc2=4, ec="0.", fc="none")
for (i,ω) in enumerate(sω0lan)
    ax4[:axvline](x = ω, color="orange", ls="dotted", lw=2.0)
    ax4[:text](ω + hf/4, 2e7, string(βlan[i]))
end
ax4[:set_ylim](0, 2.5e7)
ax4[:yaxis][:set_ticks]([0, 2.5e7])
ax4[:yaxis][:set_ticklabels]([L"$0$", L"$2.5\!\! \times\!\! 10^7$"])
ax4[:text](ν[1] + hf, 1.25e7, "(c)")
ax5 = plt[:subplot](get(gs3, (1, 0)))
ax5[:plot](ν,sprandl,"k")
ax5[:set_xlabel](L"$\omega_0/J$")
ax5[:set_ylim](0, 1.2e6)
ax5[:yaxis][:set_ticks]([0, 1.2e6])
ax5[:yaxis][:set_ticklabels]([L"$0$", L"$1.2\!\! \times\!\! 10^6$"])
ax5[:text](ν[1] + hf, 6e5, "(d)")
for (i, ax) in enumerate([ax1,ax2,ax3,ax4,ax5])
    ax[:set_xlim](ν[1], ν[end])
    i != 5 && ax[:set_xticklabels]([])
end
# set common y label to all subplots
fig[:text](0.022, 0.5, L"$\sum_{m,n} |a_{m,n}|^2$ (arb. units)",
   ha="center", va="center", rotation="vertical")
fig[:savefig]("../../figures/selection.pdf", pad_inches=0.0, transparent=true, bbox_inches="tight")
plt[:close](fig)

# plot w.fs. in real space
fig, axes = plt[:subplots](1,length(βreal), figsize=(10, 5))
for (i,ax) in enumerate(axes)

    ax[:imshow](ψ[st:en,st:en,i], origin="upper",
                     ColorMap("viridis"), interpolation="none",
                     extent=[-edge, edge, -edge, edge])
    ax[:set_xlabel](L"$m$")
    if i == 1 # leftmost panel
        ax[:set_ylabel](L"$n$")
    else
        ax[:set_yticklabels]([])
    end
end
fig[:savefig]("../../figures/real.pdf", transparent=true, pad_inches=0.0, bbox_inches="tight")
plt[:close](fig)

# plot w.fs. in  mom space
fig, axes = plt[:subplots](2,length(βlan), figsize=(10, 5))
for i = 1:length(βlan) # loop over columns
    # top row
    ax = axes[1,i]
    ax[:imshow](ψL[:,:,i], origin="upper", ColorMap("viridis"),
                     interpolation="none",
                     extent=[-π, π, -π, π])
    ax[:set_xticklabels]([])
    ax[:set_xticks]([-π,0,π])
    ax[:set_yticks]([-π,0,π])
    if i == 1 # leftmost panel
        ax[:set_ylabel](L"$k_y$")
        ax[:set_yticklabels]([L"$-\pi$",L"$0$",L"$\pi$"])
    else
        ax[:set_yticklabels]([])
    end
    # bottom row
    ax = axes[2,i]
    if i == 3 # third pannel
        ax[:imshow](ψS[:,:,i], origin="upper",
                         ColorMap("viridis"), interpolation="none",
                         extent=[-π, π, -π, π],
                         vmin=0, vmax=270000)
    else
        ax[:imshow](ψS[:,:,i], origin="upper",
                         ColorMap("viridis"), interpolation="none",
                         extent=[-π, π, -π, π])
    end
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
fig[:savefig]("../../figures/momentum.pdf", transparent=true, pad_inches=0.0, bbox_inches="tight")
plt[:close](fig)
