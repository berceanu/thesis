module BP

using Polynomials
using Base.Test

type WaveFunction
    N::Int
    int::Float64
    ψ::Vector{Complex{Float64}}
end
WaveFunction(S::SparseMatrixCSC{Complex{Float64},Int64},
             ω::Float64,P::Vector{Complex{Float64}}) =
                WaveFunction(S,ω,P,:landau)
WaveFunction(S::SparseMatrixCSC{Complex{Float64},Int64},
             ω::Float64,P::Vector{Complex{Float64}},gauge::Symbol) =
    WaveFunction(S,ω,P,gauge,1/11,0.001,0.02,0.,0.)
WaveFunction(S::SparseMatrixCSC{Complex{Float64},Int64},
             ω::Float64,P::Vector{Complex{Float64}},gauge::Symbol,
             α::Float64,γ::Float64,κ::Float64) =
                 WaveFunction(S,ω,P,gauge,α,γ,κ, 0., 0.)

function WaveFunction(S::SparseMatrixCSC{Complex{Float64},Int64},
                      ω::Float64,P::Vector{Complex{Float64}},gauge::Symbol,
                      α::Float64,γ::Float64,κ::Float64, m₀::Float64,
                      n₀::Float64)
    N::Int = sqrt(length(P))
    eval(:($(symbol(string("buildham_", gauge, "!")))))(S, N,α,κ,γ,ω,m₀,n₀)
    X = S\P
    return WaveFunction(N,sum(abs2(X)),X)
end

type ExactStates
    N::Int
    gauge::Symbol
    νs::Vector{Float64}
    states::Matrix{Complex{Float64}}
end
ExactStates(nev::Int, gauge::Symbol) = ExactStates(nev, gauge, 45)
ExactStates(nev::Int, gauge::Symbol, N::Int) = ExactStates(nev, gauge,
   N, 1/11, 0.02, 0., 0.)
ExactStates(nev::Int, gauge::Symbol, N::Int, α::Float64, κ::Float64) =
   ExactStates(nev, gauge, N, α, κ, 0., 0.)

function ExactStates(nev::Int, gauge::Symbol, N::Int, α::Float64,
   κ::Float64, m₀::Float64, n₀::Float64)
    M = spzeros(Complex{Float64}, N^2,N^2)
    eval(:($(symbol(string("buildhamexact", gauge, "!")))))(M, N,α,κ,m₀,n₀)
    (d, v, nconv, niter, nmult, resid) = eigs(M, nev=nev, which=:SR,
       ritzvec=true)
    return ExactStates(N, gauge, real(d), v)
end

function getstate(s::ExactStates, ω::Float64)
    i::Int = indmin(abs(s.νs .- ω))
    return reshape(s.states[:,i], (s.N, s.N))
end
function getstate(s::ExactStates, η::Int)
    return reshape(s.states[:,η], (s.N, s.N))
end

type Spectrum
    N::Int
    gauge::Symbol
    pump::Vector{Complex{Float64}}
    νs::Vector{Float64}
    intensity::Vector{Float64}
    states::Vector{WaveFunction}
end
Spectrum(ν::Vector{Float64},P::Vector{Complex{Float64}}) = Spectrum(ν,P,:landau)

function Spectrum(ν::Vector{Float64},P::Vector{Complex{Float64}},gauge::Symbol)
    statevec = Array(WaveFunction, length(ν))
    intvec = Array(Float64, length(ν))
    N::Int = sqrt(length(P))
    A = spzeros(Complex{Float64}, N^2,N^2)
    for (i,ω) in enumerate(ν)
        statevec[i] = WaveFunction(A, ω, P, gauge)
        intvec[i] = statevec[i].int
    end
    return Spectrum(N, gauge, P, ν, intvec, statevec)
end
function Spectrum(ν::Vector{Float64},P::Vector{Complex{Float64}},gauge::Symbol,
                  α::Float64,γ::Float64,κ::Float64)
    statevec = Array(WaveFunction, length(ν))
    intvec = Array(Float64, length(ν))
    N::Int = sqrt(length(P))
    A = spzeros(Complex{Float64}, N^2,N^2)
    for (i,ω) in enumerate(ν)
        statevec[i] = WaveFunction(A, ω, P, gauge, α,γ,κ)
        intvec[i] = statevec[i].int
    end
    return Spectrum(N, gauge, P, ν, intvec, statevec)
end
function Spectrum(ν::Vector{Float64},P::Vector{Complex{Float64}},gauge::Symbol,
                  α::Float64,γ::Float64,κ::Float64, m₀::Float64, n₀::Float64)
    statevec = Array(WaveFunction, length(ν))
    intvec = Array(Float64, length(ν))
    N::Int = sqrt(length(P))
    A = spzeros(Complex{Float64}, N^2,N^2)
    for (i,ω) in enumerate(ν)
        statevec[i] = WaveFunction(A, ω, P, gauge, α,γ,κ, m₀,n₀)
        intvec[i] = statevec[i].int
    end
    return Spectrum(N, gauge, P, ν, intvec, statevec)
end

function getstate(s::Spectrum, ω::Float64)
    i::Int = indmin(abs(s.νs .- ω))
    return reshape(s.states[i].ψ, (s.N, s.N))
end

#Check that matrix is square
function chksquare(A::AbstractMatrix)
    m,n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square"))
    m
end

#Find radius of ring
function radius(M::Matrix{Float64}, axis::Vector)
    N = chksquare(M)
    isodd(N) || throw(DimensionMismatch("invalid matrix size N=$N. N
       must be odd"))
    length(axis) == N || throw(DimensionMismatch("axis size must match
       matrix dimensions"))
    @test_approx_eq_eps(M, transpose(M), 1e-5)
    half = div(N-1, 2) + 1
    v1 = axis[half:end]
    v2 = M[half:end, half]
    i = indmax(v2)
    r = v1[i]
    return r
end

getm(i::Int64,N::Int64) = div(i-1,N)-div(N-1,2)
getn(i::Int64,N::Int64) = div(N-1,2)-rem(i-1,N)
geti(m::Int,n::Int,N::Int)=(m+div(N-1,2))*N+(div(N-1,2)-n)+1

countentries(N::Int) = N^2 + 8 + 4*(N-2)*3 + (N-2)^2*4

macro hambody(fself, fleft, fright, fup, fdown)
    return quote
        border::Int = div(N-1,2)
        for m in -border:border, n in -border:border
            i  = geti(m,n,N)
            S[i,i] = $fself
        end
        for m in -border+1:border, n in -border:border
            i  = geti(m,n,N)
            S[i,i-N] = $fleft
        end
        for m in -border:border-1, n in -border:border
            i  = geti(m,n,N)
            S[i,i+N] = $fright
        end
        for m in -border:border, n in -border:border-1
            i  = geti(m,n,N)
            S[i,i-1] = $fup
        end
        for m in -border:border, n in -border+1:border
            i  = geti(m,n,N)
            S[i,i+1] = $fdown
        end
    end
end

buildham_landau!(S::SparseMatrixCSC{Complex{Float64},Int},
                 N::Int,α::Float64,κ::Float64,γ::Float64,ω::Float64) =
                     buildham_landau!(S,N,α,κ,γ,ω, 0.,0.)
function buildham_landau!(S::SparseMatrixCSC{Complex{Float64},Int},
   N::Int,α::Float64,κ::Float64,γ::Float64,ω::Float64, m₀::Float64,
   n₀::Float64)
    @hambody(ω + im*γ - 1/2*κ*((n-n₀)^2+(m-m₀)^2), 1, 1,
       exp(-im*2π*α*m), exp(im*2π*α*m))
end
function buildham_symmetric!(S::SparseMatrixCSC{Complex{Float64},Int},
   N::Int,α::Float64,κ::Float64,γ::Float64,ω::Float64, m₀::Float64,
   n₀::Float64)
    @hambody(ω + im*γ - 1/2*κ*((n-n₀)^2+(m-m₀)^2), exp(-im*π*α*n),
       exp(im*π*α*n), exp(-im*π*α*m), exp(im*π*α*m))
end
buildham_exact!(S::SparseMatrixCSC{Complex{Float64},Int},
   N::Int,α::Float64,κ::Float64) = buildham_exact!(S, N,α,κ, 0., 0.)
function buildham_exact!(S::SparseMatrixCSC{Complex{Float64},Int},
   N::Int,α::Float64,κ::Float64, m₀::Float64, n₀::Float64)
    @hambody(1/2*κ*((n-n₀)^2+(m-m₀)^2), -1, -1, -exp(-im*2π*α*m),
       -exp(im*2π*α*m))
end
buildhamexactlandau! = buildham_exact!
function
   buildhamexactsymmetric!(S::SparseMatrixCSC{Complex{Float64},Int},
   N::Int,α::Float64,κ::Float64, m₀::Float64, n₀::Float64)
    @hambody(1/2*κ*((n-n₀)^2+(m-m₀)^2), -exp(-im*π*α*n),
       -exp(im*π*α*n), -exp(-im*π*α*m), -exp(im*π*α*m))
end

function
   genspmat(l::Function,r::Function,u::Function,d::Function,s::Function,
   N::Int,nz::Int,α::Float64)
    iseven(N) && throw(ArgumentError("invalid system size N=$N. N must
       be odd"))
    # Preallocate
    I = Array(Int64,nz)
    J = Array(Int64,nz)
    V = Array(Complex{Float64},nz)
    function setnzelem(i::Int,n::Int,m::Int; pos::ASCIIString = "self")
        if pos=="left"
            k += 1
            J[k] = i-N; I[k] = i; V[k] = l(n,m,α)
        elseif pos=="right"
            k += 1
            J[k] = i+N; I[k] = i; V[k] = r(n,m,α)
        elseif pos=="up"
            k += 1
            J[k] = i-1; I[k] = i; V[k] = u(n,m,α)
        elseif pos=="down"
            k += 1
            J[k] = i+1; I[k] = i; V[k] = d(n,m,α)
        elseif pos=="self"
            k += 1
            J[k] = i; I[k] = i; V[k] = s(n,m,α)
        end
    end
    # maximum value of m or n indices
    maxm = div(N-1,2)
    k = 0
    for i in 1:N^2
        m = getm(i,N)
        n = getn(i,N)
        setnzelem(i,n,m; pos="self")
        #corners
        #top left
        if n==maxm && m==-maxm
            setnzelem(i,n,m; pos="right")
            setnzelem(i,n,m; pos="down")
        #top right
        elseif n==maxm && m==maxm
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="down")
        #bottom right
        elseif n==-maxm && m==maxm
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="up")
        #bottom left
        elseif n==-maxm && m==-maxm
            setnzelem(i,n,m; pos="right")
            setnzelem(i,n,m; pos="up")
        #edges
        #top
        elseif n == maxm
            setnzelem(i,n,m; pos="right")
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="down")
        #right
        elseif m == maxm
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="up")
            setnzelem(i,n,m; pos="down")
        #bottom
        elseif n == -maxm
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="up")
            setnzelem(i,n,m; pos="right")
        #left
        elseif m == -maxm
            setnzelem(i,n,m; pos="down")
            setnzelem(i,n,m; pos="up")
            setnzelem(i,n,m; pos="right")
        else #bulk
            setnzelem(i,n,m; pos="down")
            setnzelem(i,n,m; pos="up")
            setnzelem(i,n,m; pos="right")
            setnzelem(i,n,m; pos="left")
        end
    end
    return sparse(I,J,V)
end

#various pumping schemes
function δpmp(N::Int; A=1., seed=0, σ=0., n0=0, m0=0)
    i = (m0+div(N-1,2)) * N + (div(N-1,2)-n0) + 1
    f = zeros(Complex{Float64}, N^2)
    f[i] = A * one(Complex{Float64})
    f
end
function gausspmp(N::Int; A=1., seed=0, σ=1., n0=0, m0=0)
    x0=m0
    y0=n0
    f = zeros(Complex{Float64}, N,N)
    m = collect(-div(N-1,2):div(N-1,2))
    n = collect(div(N-1,2):-1:-div(N-1,2))
    for c in 1:N, l in 1:N
        x = m[c]
        y = n[l]
        f[l,c] = A*exp(-1/(2σ^2)*((x-x0)^2 + (y-y0)^2))
    end
    reshape(f, N^2)
end
function randpmp(N::Int; A=1., seed=123, σ=0., n0=0, m0=0)
    # seed the RNG #
    srand(seed)
    # generate matrix of random phases in interval [0,2π)
    ϕ = 2π .* rand(N^2)
    A .* exp(im .* ϕ)
end
function homopmp(N::Int; A=1., seed=0, σ=0., n0=0, m0=0)
    A .* ones(Complex{Float64}, N^2)
end

#arbitrary resolution fft
function myfft2(ψr::Matrix{Complex{Float64}}, k1::Float64,
   k2::Float64, xs1::Float64, xs2::Float64, Δx1::Float64, Δx2::Float64)
    (N1,N2) = size(ψr)
    s = zero(Complex{Float64})
    for n2 in 1:N2, n1 in 1:N1
        xn1 = xs1 + (n2-1)*Δx1 #x
        xn2 = xs2 + (n1-1)*Δx2 #y
        cexp = exp(-im*(k1*xn1 + k2*xn2))
        s += ψr[n1,n2]*cexp
    end
    s
end
function myfft2(ψr::Matrix{Complex{Float64}}, k1, k2)
    N1 = length(k2); N2 = length(k1)
    out = Array(Complex{Float64}, N1,N2)
    for j in 1:N2, i in 1:N1
        out[i,j] = myfft2(ψr, k1[j], k2[i], 0., 0., 1., 1.)
    end
    out
end

# maps any momentum to the first Brillouin Zone
function fbz(mom::Float64)
    m = mod(mom, 2π)
    m <= π ? m : m - 2π
end
# Magnetic Brillouin Zone
function mbz(data::Array{Float64,2}, r::Int, q::Int,
   kxmbz::FloatRange{Float64}, k::FloatRange{Float64})
    # data is |ψ(FBZ)|², input
    V = zeros(Float64, (length(k), r)) # |ψ(MBZ)|², output
    for (i,px) in enumerate(kxmbz) # loop over momenta in MBZ
        for j = 0:q-1 # sum over various components
            ptld = px - j * 2π/q
            idx = indmin(abs(k .- fbz(ptld))) # index of \tilde{p_x}
            V[:,i] += data[:, idx]
        end
    end
    V/(4π^2)
end

#comparison to analytics
function compute_hermite_polynomial(n)
    P = Poly([1])
    const x = Poly([0; 1])
    for i = 1:n
        P = 2x*P - polyder(P)
    end
    P
end
function χ(kx0, ky, α, β)
    l = sqrt(2π*α)
    sum = zero(Complex{Float64})
    for j in -20:20 #truncate the sum
        H = polyval(compute_hermite_polynomial(β), kx0/l + j*l)
        sum += exp(-im*ky*j) * exp(-(kx0 + j*l^2)^2/(2l^2)) * H
    end
    sum
end
end #module
