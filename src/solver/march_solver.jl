# march_solver.jl

# Implements the MARCH solver introduced in https://arxiv.org/pdf/2507.02644

# Implements the MARCH solver introduced in https://arxiv.org/pdf/2507.02644

mutable struct MarchSolver{S<:AbstractSolver,Tθ<:Number,TR<:Real} <: AbstractSolver
    base::S
    μ::TR
    β::TR
    eps::TR
    λ::TR

    # state (parameter space)
    phi::Vector{Tθ}
    v::Vector{TR}
    θdot_prev::Vector{Tθ}
    initialized::Bool

    # work buffers
    Os::Matrix{Tθ}      # B×P
    A::Matrix{Tθ}       # B×B
    rhs::Vector{Tθ}     # B
    x::Vector{Tθ}       # B
    tmpP::Vector{Tθ}    # P
end

# Helper: pick the real scalar type for v from Tθ
_realtype(::Type{T}) where {T} = typeof(real(zero(float(T))))  # ComplexF64->Float64, Float32->Float32, etc.

function MarchSolver(base::S, ::Type{Tθ}; μ=0.95, β=0.995, eps=1e-8, λ=1e-3) where {S<:AbstractSolver,Tθ<:Number}
    TR = _realtype(Tθ)
    return MarchSolver{S,Tθ,TR}(
        base, TR(μ), TR(β), TR(eps), TR(λ),
        Tθ[], TR[], Tθ[], false,
        Matrix{Tθ}(undef, 0, 0),
        Matrix{Tθ}(undef, 0, 0),
        Tθ[], Tθ[], Tθ[]
    )
end

function _init!(ms::MarchSolver{S,Tθ,TR}, B::Int, P::Int) where {S,Tθ,TR}
    ms.phi       = zeros(Tθ, P)           # φ₀ = 0
    ms.v         = fill(ms.eps, P)        # v₀ = eps
    ms.θdot_prev = zeros(Tθ, P)           # θ̇₀ = 0

    ms.Os   = Matrix{Tθ}(undef, B, P)
    ms.A    = Matrix{Tθ}(undef, B, B)
    ms.rhs  = zeros(Tθ, B)
    ms.x    = zeros(Tθ, B)
    ms.tmpP = zeros(Tθ, P)

    ms.initialized = true
    return ms
end

# MARCH in the T/Woodbury form (recommended when B << P)
function solve_T(ms::MarchSolver{S,Tθ,TR}, J::Jacobian, Es::EnergySummary;
                 timer=TimerOutput(), kwargs...) where {S,Tθ,TR}

    O = centered(J)                # B×P (Matrix or Transpose wrapper)
    ε = -centered(Es)               # length B (typically ComplexF64 or Real -> promote)

    B = nr_samples(J)
    P = nr_parameters(J)

    if !ms.initialized || length(ms.phi) != P
        _init!(ms, B, P)
    elseif size(ms.Os,1) != B
        # only resize sample-space buffers
        ms.Os  = Matrix{Tθ}(undef, B, P)
        ms.A   = Matrix{Tθ}(undef, B, B)
        ms.rhs = similar(ms.rhs, B)
        ms.x   = similar(ms.x, B)
    end

    @assert size(O,1) == B && size(O,2) == P

    # --- build Os = O * diag(s), with s = v^{-1/4}  (so Os*Os' = O*diag(v^{-1/2})*O') ---
    @timeit timer "march_scale_O" begin
        # copy O into Os (handles Matrix or Transpose)
        copyto!(ms.Os, O)

        # column scaling: Os[:,j] *= s[j]
        # s[j] = v[j]^(-1/4)
        @inbounds for j in 1:P
            s = inv(sqrt(sqrt(ms.v[j]))) # = 1 / v^(1/4)
            @views ms.Os[:, j] .*= s
        end
    end

    # --- A = Os * Os' + λ I (B×B) ---
    @timeit timer "march_build_A" begin
        # gemm! does: A = 1*Os*Os' + 0*A
        BLAS.gemm!('N', 'C', one(Tθ), ms.Os, ms.Os, zero(Tθ), ms.A)
        @inbounds for i in 1:B
            ms.A[i,i] += ms.λ
        end
    end

    # --- rhs = ε - O*phi ---
    @timeit timer "march_rhs" begin
        # rhs := ε
        ms.rhs .= Tθ.(ε)

        # rhs -= O*phi   (mul! works for Matrix and Transpose)
        # tmpB = O*phi, we can reuse ms.x as a temp here
        mul!(ms.x, O, ms.phi)       # ms.x = O*phi
        ms.rhs .-= ms.x
    end

    # --- solve A x = rhs using base solver ---
    @timeit timer "march_solve" begin
        ms.x .= ms.base(ms.A, ms.rhs; kwargs...)
    end

    # --- θ̇ = -diag(D) * (O' * x) + phi, with D = v^{-1/2} ---
    @timeit timer "march_backproj" begin
        mul!(ms.tmpP, adjoint(O), ms.x)   # tmpP = O' * x  (P)

        # scale by D = v^{-1/2} and add phi
        θdot = similar(ms.phi)
        @inbounds for j in 1:P
            Dj = inv(sqrt(ms.v[j]))      # v^{-1/2}
            θdot[j] = (Dj * ms.tmpP[j]) + ms.phi[j]
        end

        # update moments
        ms.phi .= ms.μ .* θdot
        ms.v   .= ms.β .* ms.v .+ abs2.(θdot .- ms.θdot_prev)   # abs2 for complex updates
        ms.v   .= max.(ms.v, ms.eps)
        ms.θdot_prev .= θdot

        return θdot
    end
end

# Hook into your NaturalGradient call path (prefer T-form for MARCH)
function (ms::MarchSolver)(ng::NaturalGradient; method=:auto, compute_error=true, kwargs...)
    ng.θdot = solve_T(ms, ng.J, ng.Es; kwargs...)
    if compute_error
        tdvp_error!(ng)
    end
    return ng
end