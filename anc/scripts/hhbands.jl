using PyPlot


# Functions

function momsphhmat!(A::Matrix{Complex{Float64}},
                     kx0::Float64,ky::Float64,
                     p::Int)
    q::Int = size(A,1)
    A[1,q] = -exp(-im*q*ky)
    A[q,1] = -exp(im*q*ky)
    #upper subdiagonal
    for i in diagind(A,1)
        A[i] = -one(Complex{Float64})
    end
    #lower subdiagonal
    for i in diagind(A,-1)
        A[i] = -one(Complex{Float64})
    end
    #main diagonal
    α = p/q
    for (j,i) in enumerate(diagind(A))
        A[i] = -2*cos(kx0 + 2*π*α*j)
    end
    nothing
end

function hhgrstate!(ve::Matrix{Float64}, p::Int, q::Int)
    ky = linspace(-π, π, size(ve,1))
    kx₀= linspace(-π/q, π/q, size(ve,2))
    H = zeros(Complex{Float64}, (q,q))
    for col=1:length(kx₀), row=1:length(ky)
        momsphhmat!(H, kx₀[col], ky[row], p)
        ve[row,col] = eigmin(H)
    end
    nothing
end

hhgrstate!(ve::Matrix{Float64}, q::Int) = hhgrstate!(ve, 1, q)

function hhladder!(E::Array{Float64, 3}, p::Int)
    ky = linspace(-π, π, size(E,1))
    q = size(E,3)
    kx₀= linspace(-π/q, π/q, size(E,2))
    H = zeros(Complex{Float64}, (q,q))
    for col=1:length(kx₀), row=1:length(ky)
        momsphhmat!(H, kx₀[col], ky[row], p)
        E[row,col,:] = eigvals(H)
    end
    nothing
end

hhladder!(E::Array{Float64,3}) = hhladder!(E, 1)

bgap(qs::UnitRange{Int}) = bgap(collect(qs))
bgap(qs::Vector{Int}) = [bgap(q) for q in qs]
bgap(q::Int) = bgap(1, q)
function bgap(p::Int, q::Int)
    ladder = Array(Float64, (25,25,q))
    hhladder!(ladder, p)
    e1 = mean(ladder[:,:,1])
    e2 = mean(ladder[:,:,2])
    return e2-e1
end

bwidth(qs::UnitRange{Int}) = bwidth(collect(qs))
bwidth(qs::Vector{Int}) = [bwidth(q) for q in qs]
bwidth(q::Int) = bwidth(1, q)
function bwidth(p::Int, q::Int)
    gstate = Array(Float64, 25, 25)
    hhgrstate!(gstate, p, q)
    a,b = extrema(gstate)
    return b-a
end

# Computing

q = 7
qs = 3:20

gs = Array(Float64, 200,200)
hhgrstate!(gs, q)
a, b = extrema(gs)

ladder = Array(Float64, (200,201,q))
hhladder!(ladder)


# Plotting

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


f, ax = plt[:subplots](figsize=(5, 5))
img = ax[:imshow](gs, origin="upper", ColorMap("gist_heat_r"), interpolation="none",
                 extent=[-π/q, π/q, -π, π],
                 aspect="auto")

ax[:set_ylabel](L"$p_y$",labelpad=-1)
ax[:set_xlabel](L"$p_x^0$", labelpad=-1)
ax[:xaxis][:set_ticks]([-π/q,0,π/q])
ax[:xaxis][:set_ticklabels]([LaTeXString(string("\$-\\pi/",q,"\$")), L"$0$", LaTeXString(string("\$\\pi/",q,"\$"))])

ax[:yaxis][:set_ticks]([-π,0,π])
ax[:yaxis][:set_ticklabels]([L"$-\pi$", L"$0$", L"$\pi$"])

cbaxes = f[:add_axes]([0.25, 0.04, 0.65, 0.015])
cbar = f[:colorbar](img, cax=cbaxes, orientation="horizontal")
cbar[:set_ticks]([a,b])
cbar[:set_ticklabels]([L"$a$", L"$b$"])
cbar[:set_label](L"$E[J]$", rotation=0, labelpad=-20, y=0.5)
cbar[:solids][:set_edgecolor]("face")

#f[:savefig]("../figures/gs_q_7.svg", transparent=true, pad_inches=0.0, bbox_inches="tight")
#plt[:close](f)


f, ax = plt[:subplots](figsize=(8, 5))

for i in 1:q
    ax[:plot](linspace(-π,π,size(ladder,1)), ladder[:,div(size(ladder,2)-1,2)+1,i], "k")
end

ax[:yaxis][:set_ticks]([-4,-2,0,2,4])
ax[:xaxis][:set_ticks]([-π,-π/2,0,π/2,π])
ax[:xaxis][:set_ticklabels]([L"$-\pi$",L"$-\pi/2$", L"$0$",L"$\pi/2$" ,L"$\pi$"])

ax[:set_xlim](-π, π)

ax[:set_ylabel](L"$E[J]$")
ax[:set_xlabel](L"$p_y$")

#f[:savefig]("../figures/7bands.svg", transparent=true, pad_inches=0.0, bbox_inches="tight")
#plt[:close](f)


f, ax = plt[:subplots](figsize=(8, 3))
ax[:plot](qs, bgap(qs), "black", marker="o", label=L"$\Delta E$") 
ax[:plot](qs, bwidth(qs).*10, "black", marker="o", ls="dashed", label=L"$BW \times 10$") 

ax[:set_ylim](-0.1, 3)
ax[:yaxis][:set_ticks]([0,1.5,3])
ax[:set_xlim](qs[1], qs[end])

ax[:set_xlabel](L"$q$")

ax[:legend](loc="upper right")

#f[:savefig]("../figures/bands.pdf", transparent=true, pad_inches=0.0, bbox_inches="tight")
#plt[:close](f)
