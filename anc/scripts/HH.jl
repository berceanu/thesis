module HH

import BP

#in-place builder of the q x q HH matrix in mom. sp.
momsphhmat!(A::Matrix{Complex{Float64}}, kx0::Float64,ky::Float64) =
   momsphhmat!(A, kx0,ky, 1)

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

# Computes the q energy levels E(p_y)
hhladder(q::Int) = hhladder(1,q)
function hhladder(p::Int, q::Int)
    ky = linspace(-π, π, 100)
    kx₀= linspace(-π/q, π/q, 100)
    E = Array(Float64, (length(ky),length(kx₀),q))
    H = zeros(Complex{Float64}, (q,q))
    for col=1:length(kx₀), row=1:length(ky)
        momsphhmat!(H, kx₀[col], ky[row], p)
        E[row,col,:] = eigvals(H)
    end
    return E
end
hhladder!(E::Array{Float64,3}) = hhladder!(E, 1)
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

# ground state energy of the HH hamiltonian
hhgrstate!(ve::Matrix{Float64}, q::Int) = hhgrstate!(ve, 1, q)
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

# zero point energy error
ηzpe(q::Int, κ::Float64) = ηzpe(1, q, κ)
ηzpe(p::Int, q::Int, κ::Float64) = (N=15; A =
   spzeros(Complex{Float64}, N^2,N^2); ηzpe(A, p, q, κ))

function ηzpe(M::SparseMatrixCSC{Complex{Float64},Int}, p::Int,
   q::Int, κ::Float64)
    N::Int = sqrt(size(M,1))
    α::Float64 = p/q
    gs = Array(Float64, 25,25)
    hhgrstate!(gs, p, q)
    e1 = mean(gs)
    et = e1 + 1/2*κ/(2π*α)
    BP.buildham_exact!(M, N,α,κ)
    er = real(eigs(M, nev=1, which=:SR, ritzvec=false)[1][1])
    return 4π*α/κ * (er - et)
end

ηzpe(q::Int, κs::Vector{Float64}) = vec(ηzpe(collect(q), κs))
ηzpe(qs::UnitRange{Int}, κ::Float64) = ηzpe(collect(qs), κ)
ηzpe(qs::Vector{Int}, κ::Float64) = vec(ηzpe(qs, collect(κ)))
ηzpe(qs::UnitRange{Int}, κs::Vector{Float64}) = ηzpe(collect(qs), κs)

function ηzpe(qs::Vector{Int}, κs::Vector{Float64})
    N=15
    A = spzeros(Complex{Float64}, N^2,N^2)
    η = Array(Float64, length(qs), length(κs))
    for col=1:length(κs), row=1:length(qs)
        η[row,col] = ηzpe(A, 1, qs[row], κs[col])
    end
    return η
end

# level error
ηlev(q::Int, κ::Float64) = ηlev(1, q, κ)
ηlev(p::Int, q::Int, κ::Float64) = (N=15; A =
   spzeros(Complex{Float64}, N^2,N^2); ηlev(A, p, q, κ))

function ηlev(M::SparseMatrixCSC{Complex{Float64},Int}, p::Int,
   q::Int, κ::Float64)
    N::Int = sqrt(size(M,1))
    α::Float64 = p/q
    BP.buildham_exact!(M, N,α,κ)
    # get first 4 levels of spectrum of HH+trap
    bilevel = real(eigs(M, nev=4, which=:SR, ritzvec=false)[1])
    #calculate level spacing
    return 2π*α/κ*diff(bilevel[1:2])[1] - 1
end
function ηlev(qs::UnitRange{Int}, κ::Float64, lowβ::Int, highβ::Int)
    N=15 # system size is NxN
    # initialize (sparse) Hamiltonian matrix
    M = spzeros(Complex{Float64}, N^2,N^2)
    diffs = Array(Float64, length(qs))
    for (i,q) in enumerate(qs)
        # construct disipationless HH+trap Hamiltonian in Landau gauge
        # q dependent, need to loop over all q!
        BP.buildham_exact!(M, N,1/q,κ)
        # get first 5 levels of spectrum of M
        levels = real(eigs(M, nev=5, which=:SR, ritzvec=false)[1])
        #calculate level spacing
        diffs[i] = 2π/(q*κ) * (levels[highβ+1] - levels[lowβ+1]) - 1.0
    end
    return diffs
end

ηlev(q::Int, κs::Vector{Float64}) = vec(ηlev([q], κs))
ηlev(qs::UnitRange{Int}, κ::Float64) = ηlev([qs], κ)
ηlev(qs::Vector{Int}, κ::Float64) = vec(ηlev(qs, [κ]))
ηlev(qs::UnitRange{Int}, κs::Vector{Float64}) = ηlev([qs], κs)

function ηlev(qs::Vector{Int}, κs::Vector{Float64})
    N=15
    A = spzeros(Complex{Float64}, N^2,N^2)
    η = Array(Float64, length(qs), length(κs))
    for col=1:length(κs), row=1:length(qs)
        η[row,col] = ηlev(A, 1, qs[row], κs[col])
    end
    return η
end

bwidth(qs::UnitRange{Int}) = bwidth([qs])
bwidth(qs::Vector{Int}) = [bwidth(q) for q in qs]
bwidth(q::Int) = bwidth(1, q)

function bwidth(p::Int, q::Int)
    gstate = Array(Float64, 25, 25)
    hhgrstate!(gstate, p, q)
    a,b = extrema(gstate)
    return b-a
end

bgap(qs::UnitRange{Int}) = bgap([qs])
bgap(qs::Vector{Int}) = [bgap(q) for q in qs]
bgap(q::Int) = bgap(1, q)

function bgap(p::Int, q::Int)
    ladder = Array(Float64, (25,25,q))
    hhladder!(ladder, p)
    e1 = mean(ladder[:,:,1])
    e2 = mean(ladder[:,:,2])
    return e2-e1
end
end #module
