using PyPlot
import DSP: fftshift, fftfreq
function sider(l, n)
    Δ = 2l/(n-1)
    -l:Δ:l
end
sidek(l, n) = 2pi*fftshift(fftfreq(n, (n-1)/2l))
ε(ky::Float64, kx::Float64, Δp) = (kx^2 + ky^2)/2 - Δp
λ1(ky::Float64, kx::Float64, vp, Δp, κ) = ((ky*vp[1] + kx*vp[2])
           -im*κ + sqrt(ε(ky, kx, Δp)*(ε(ky, kx, Δp) + 2) + 0.*im))
λ2(ky::Float64, kx::Float64, vp, Δp, κ) = ((ky*vp[1] + kx*vp[2])
           -im*κ - sqrt(ε(ky, kx, Δp)*(ε(ky, kx, Δp) + 2) + 0.*im))
ψtmom(ky::Float64, kx::Float64, gv, vp, Δp, κ) = (gv*(ε(ky, kx, Δp)
                                    -(ky*vp[1] + kx*vp[2]) + im*κ)/
                     (λ1(ky, kx, vp, Δp, κ)*λ2(ky, kx, vp, Δp, κ)))
type System
    N::Int # nx = ny = N
    gv::Float64
    L::Float64 # lx = ly = L
    v::Float64
    Δ::Float64
    κ::Float64
    ψk::Matrix{Float64}
    ψr::Matrix{Float64}
    k::FloatRange{Float64}
    r::FloatRange{Float64}
end
function System(N::Int, gv::Float64, L::Float64, v::Float64,
                Δ::Float64, κ::Float64)
    r = sider(L,N)
    k = sidek(L,N)
    ψmom = Array(Complex{Float64}, N,N)
    ψmom = [ψtmom(momy, momx, gv, (0.,v), Δ, κ) for momx in k,
            momy in k]
    ψmom_cp = copy(ψmom)
    #add constant part to wavefunction at k = 0
    ψmom_cp[N/2+1,N/2+1] += sqrt(N*N)
    return System(N, gv, L, v, Δ, κ, abs2(transpose(ψmom)),
            abs2(transpose(fftshift(ifft(sqrt(N*N)*ψmom_cp)))),k,r)
end
System(L::Float64, v::Float64, Δ::Float64, κ::Float64) =
    System(512, 0.01, L, v, Δ, κ)
type Param
    Δ::Float64
    κ::Float64
    L::Float64
    v::Float64
    zrangek::(Float64,Float64)
    zranger::(Float64,Float64)
end
type ParamV
    V::Vector{Param}
end
ParamV(Δ::Vector{Float64},κ::Vector{Float64},L::Vector{Float64},
       v::Vector{Float64}, zrangek::Vector{(Float64,Float64)},
       zranger::Vector{(Float64,Float64)}) =
           ParamV([Param(Δ0,κ0,L0,v0,zrangek0,zranger0) for
                   (Δ0,κ0,L0,v0,zrangek0,zranger0) in
                   zip(Δ,κ,L,v,zrangek,zranger)])
import Base.getindex, Base.length, Base.start, Base.next
getindex(pv::ParamV,i::Int64) = pv.V[i]
length(pv::ParamV) = length(pv.V)
start(pv::ParamV) = 1
next(pv::ParamV,i::Int64) = (pv.V[i],i+1)
# nondiffusive spectra
pnond = ParamV([-0.3472, -.25, 0.],
               .03*ones(3),
               100*ones(3),
               1.5*ones(3),
               [(0.,.04),(0.,.03),(0.,.01)],
               [(.8,1.4),(.8,1.4),(.8,1.3)])
snond = [System(p.L, p.v, p.Δ, p.κ) for p in pnond]
## Plotting ##
fig, axes = plt[:subplots](2,3, figsize=(10, 6))
for i = 1:3 # loop over columns
    # top row
    ax = axes[1,i]
    ax[:imshow](clamp(snond[i].ψk,
                      pnond[i].zrangek[1], pnond[i].zrangek[2]),
                origin="upper", ColorMap("gist_heat_r"),
                interpolation="none",
                extent=[snond[i].k[1], snond[i].k[end],
                        snond[i].k[1], snond[i].k[end]])
    ax[:set_xlabel](L"$\delta k_x[m c_s]$")
    ax[:set_xticks]([-4,0,4])
    ax[:set_yticks]([-2,0,2])
    ax[:set_xlim](-4,4)
    ax[:set_ylim](-3,3)
    if i == 1 # leftmost panel
        ax[:set_ylabel](L"$\delta k_y[m c_s]$")
    else
        ax[:set_yticklabels]([])
    end
    # bottom row
    ax = axes[2,i]
    ax[:imshow](clamp(snond[i].ψr,
                      pnond[i].zranger[1], pnond[i].zranger[2]),
                origin="upper", ColorMap("gist_heat_r"),
                interpolation="none",
                extent=[snond[i].r[1], snond[i].r[end],
                        snond[i].r[1], snond[i].r[end]])
    ax[:set_xlabel](L"$x [(m c_s)^{-1}]$")
    ax[:set_xticks]([-20,0,20])
    ax[:set_yticks]([-10,0,10])
    ax[:set_xlim]([-20,20])
    ax[:set_ylim]([-15,15])
    if i == 1 # leftmost panel
        ax[:set_ylabel](L"$y [(m c_s)^{-1}]$")
    else
        ax[:set_yticklabels]([])
    end
end
fig[:savefig]("nond.pdf", transparent=true,
              pad_inches=0.0, bbox_inches="tight")
plt[:close](fig)
