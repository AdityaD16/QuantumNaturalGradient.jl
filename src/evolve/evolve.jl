# Abstract types and common utilities
abstract type AbstractIntegrator end
abstract type AbstractCompositeIntegrator <: AbstractIntegrator end
abstract type AbstractSchedule end

"""
Composite natural gradient: keeps the per-term NaturalGradient objects,
but exposes only the summary fields we want to use in evolve.
"""
struct CompositeNaturalGradient{NG, TD}
    terms::Vector{NG}      # each element is a NaturalGradient
    Es_mean::Float64       # scalar summary energy
    θdot::TD               # the update direction actually used for integration step
    Es_error::Float64      # scalar energy error summary (e.g. sqrt(sum σ_i^2) or whatever you decide)
    saved_properties::Dict{Symbol,Any}
end

# Make existing code able to ask for θdot uniformly
get_θdot(ng::CompositeNaturalGradient; θtype=nothing) = ng.θdot

function Base.getproperty(integrator::AbstractCompositeIntegrator, property::Symbol)
    if property === :lr || property === :step
        return getproperty(integrator.integrator, property)
    else
        return getfield(integrator, property)
    end
end

function clamp_and_norm!(gradients::AbstractVector{<:Complex}, clip_val::Real, clip_norm::Real)
    # Clamp magnitude of each entry to clip_val
    @inbounds for i in eachindex(gradients)
        z = gradients[i]
        r = abs(z)
        if r > clip_val && r != 0
            gradients[i] = z * (clip_val / r)
        end
    end

    nrm = norm(gradients)
    if nrm > clip_norm
        gradients .= (gradients ./ nrm) .* clip_norm
    end
    return gradients
end

# RK4 integrator structure
mutable struct RK4 <: AbstractIntegrator
    lr::Float64
    step::Integer
    use_clipping::Bool
    clip_norm::Float64
    clip_val::Float64
    RK4(;lr=0.05, step=0, use_clipping=false, clip_norm=5.0,
         clip_val=1.0) =
        new(lr, step, use_clipping, clip_norm, clip_val)
end

function (integrator::RK4)(θ::ParameterTypes, Oks_and_Eks_::Function; kwargs...)

    θtype = eltype(θ)
    h = integrator.lr

    ng1 = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
    k1 = get_θdot(ng1; θtype=θtype)
    if integrator.use_clipping
        clamp_and_norm!(k1, integrator.clip_val, integrator.clip_norm)
    end

    θ2 = deepcopy(θ)  
    θ2 .+= (h/2) .* k1
    ng2 = NaturalGradient(θ2, Oks_and_Eks_; kwargs...)
    k2 = get_θdot(ng2; θtype=θtype)
    if integrator.use_clipping
        clamp_and_norm!(k2, integrator.clip_val, integrator.clip_norm)
    end
    
    θ3 = deepcopy(θ)  
    θ3 .+= (h/2) .* k2
    ng3 = NaturalGradient(θ3, Oks_and_Eks_; kwargs...)
    k3 = get_θdot(ng3; θtype=θtype)
    if integrator.use_clipping
        clamp_and_norm!(k3, integrator.clip_val, integrator.clip_norm)
    end

    θ4 = deepcopy(θ)  
    θ4 .+= h .* k3
    ng4 = NaturalGradient(θ4, Oks_and_Eks_; kwargs...)
    k4 = get_θdot(ng4; θtype=θtype)
    if integrator.use_clipping 
        clamp_and_norm!(k4, integrator.clip_val, integrator.clip_norm)
    end
    
    θ .+= (h/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    integrator.step += 1

    return θ, ng1
end


# Euler integrator structure
mutable struct Euler <: AbstractIntegrator
    lr::Float64
    step::Integer
    use_clipping::Bool
    clip_norm::Float64
    clip_val::Float64
    write_func::Function
    basis_func::Function
    vec::Function
    gates::Tuple{Vararg{String}}
    Euler(;lr=0.05, step=0, use_clipping=false, clip_norm=5.0, clip_val=1.0,write_func=nothing,basis_func=nothing,vec=nothing,gates=nothing) = new(lr, step, use_clipping, clip_norm, clip_val,write_func,basis_func,vec,gates)
end

# Euler integrator step function
function (integrator::Euler)(θ::ParameterTypes, Oks_and_Eks_::Function; kwargs...)
    if kwargs[:timer] !== nothing 
        natural_gradient = @timeit kwargs[:timer] "NaturalGradient" NaturalGradient(θ, Oks_and_Eks_; kwargs...) 
    else
        natural_gradient = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
    end
    g = get_θdot(natural_gradient; θtype=eltype(θ))
    if integrator.use_clipping
        clamp_and_norm!(g, integrator.clip_val, integrator.clip_norm)
    end
    θ .+= integrator.lr .* g
    integrator.step += 1
    return θ, natural_gradient
end

function (integrator::Euler)(
    θ::ParameterTypes,
    Oks_and_Eks::Tuple{Vararg{Function}};
    verbosity=0,
    kwargs...
)
    nterms = length(Oks_and_Eks)

    terms = Vector{Any}(undef, nterms)  # store NaturalGradient objects

    θdot_total = nothing
    Es_total = 0.0
    err2_total = 0.0

    peps_dot = deepcopy(θ.obj)

    for (i, f) in enumerate(Oks_and_Eks)
        ng = if haskey(kwargs, :timer) && kwargs[:timer] !== nothing
            @timeit kwargs[:timer] "NaturalGradient" NaturalGradient(θ, f; kwargs...)
        else
            NaturalGradient(θ, f; kwargs...)
        end
        terms[i] = ng

        g = get_θdot(ng; θtype=eltype(θ))
        if integrator.use_clipping
            clamp_and_norm!(g, integrator.clip_val, integrator.clip_norm)
        end

        # Optional basis change per term
        if integrator.gates !== nothing && integrator.gates[i] != "Id"
            integrator.write_func(peps_dot, g)
            peps_dot = integrator.basis_func(peps_dot; gate=integrator.gates[i])
            g = integrator.vec(peps_dot)
        end

        if θdot_total === nothing
            θdot_total = similar(g)
            θdot_total .= 0
        end
        θdot_total .+= g

        Ei = real(mean(ng.Es))
        Es_total += Ei
        err2_total += energy_error(ng.Es)^2

    end

    θ .+= integrator.lr .* θdot_total

    integrator.step += 1

    Es_error = sqrt(err2_total)

    comp = CompositeNaturalGradient(
        terms,
        Es_total,
        θdot_total,
        Es_error,
        Dict{Symbol,Any}(:composite => true, :nterms => nterms)
    )

    return θ, comp
end

# Initialize and manage the optimization state
mutable struct OptimizationState
    Oks_and_Eks::Union{Function, Tuple{Vararg{Function}}}
    callback::Function
    transform!::Function
    θ
    integrator::AbstractIntegrator
    solver::AbstractSolver
    energy::Real
    niter::Int
    history::DataFrame
    timer::TimerOutput
    discard_outliers
    sample_nr::Int
    maxiter::Int
    gradtol::Real
    verbosity::Int
end

get_misc(state::OptimizationState) = Dict("energy" => state.energy, "niter" => state.niter, "history" => state.history, "rng" => get_rng())

function get_rng()
    t = current_task()
    return [t.rngState0, t.rngState1, t.rngState2, t.rngState3]
end

function OptimizationState(Oks_and_Eks, θ::T, integrator; 
    sample_nr=1000, 
    solver=EigenSolver(1e-6),
    callback=(args...; kwargs...) -> nothing, 
    transform! = (o, x) -> x,
    logger_funcs=[], 
    misc_restart=nothing, 
    timer=TimerOutput(), discard_outliers=0, maxiter=50, gradtol=1e-10, verbosity=0) where {T}
    
    energy(; energy) = energy
    var_energy(; var_energy) = var_energy
    norm_natgrad(; norm_natgrad) = norm_natgrad
    norm_θ(; norm_θ) = norm_θ
    niter(; niter) = niter
    logger_funcs = [niter, energy, var_energy, norm_natgrad, norm_θ, logger_funcs...]
    history = Observer(logger_funcs...)
    state = OptimizationState(
        Oks_and_Eks, callback, transform!, θ, integrator, solver,
        1e10, 1, history, timer, discard_outliers, sample_nr, maxiter, gradtol, verbosity
    )
    
    if misc_restart !== nothing
        set_misc(state, misc_restart)
        fixbrokenObserver!(state.history, logger_funcs)
        state.niter += 1
    end

    return state
end

function fixbrokenObserver!(history, logger_funcs)
    for f in logger_funcs
        if !(Observers.get_function(history, string(f)) isa Function)
            Observers.set_function!(history, f)
        end
    end
    return history
end

function set_misc(state::OptimizationState, misc)

    if misc isa DataFrame # This is for backward compatibility
        @warn "The misc dictionary is a DataFrame. This is deprecated. Please use a dictionary instead."
        state.history = misc
        enes = state.history[!, "energy"]
        state.niter = length(enes)
        state.energy = enes[end]
        println("Restarting from niter = $(state.niter)")
        return nothing
    end

    if haskey(misc, "history")
        @assert state.history isa DataFrame "The history is not a DataFrame."
        state.history = misc["history"]
    else
        error("The misc dictionary does not have a history key")
    end
    
    if haskey(misc, "niter")
        state.niter = misc["niter"]
        println("Restarting from niter = $(state.niter)")
    else
        error("The misc dictionary does not have a niter key")
    end
    
    if haskey(misc, "energy")
        state.energy = misc["energy"]
    end

    if haskey(misc, "rng")
        set_rng(misc["rng"])
    end
end

function set_rng(r)
    t = current_task()
    t.rngState0 = r[1]
    t.rngState1 = r[2]
    t.rngState2 = r[3]
    t.rngState3 = r[4]
end

function evolve(construct_mps, θ::T, H; integrator=Euler(0.1), maxiter=10, callback=(args...; kwargs...) -> nothing, 
    logger_funcs=[], copy=true, misc_restart=nothing, timer=TimerOutput(), kwargs...) where {T}
    
    Oks_and_Eks_ = (θ, sample_nr) -> Oks_and_Eks(θ, construct_mps, H, sample_nr; kwargs...)
    energy, θ, misc = evolve(Oks_and_Eks_, θ; integrator, maxiter, callback, logger_funcs, copy, misc_restart, timer)
    return energy, θ, construct_mps(θ), misc
end

function evolve(Oks_and_Eks_, θ::T; 
    integrator=Euler(0.1),
    solver=EigenSolver(1e-6),
    maxiter=10, 
    sample_nr=1000,
    callback=(args...; kwargs...) -> nothing, 
    transform! = (o, x) -> x,
    verbosity=0,
    logger_funcs=[], 
    copy=false, 
    misc_restart=nothing, 
    discard_outliers=0.,
    timer=TimerOutput(), gradtol=1e-10) where {T}

    if copy
        θ = deepcopy(θ)
    end

    # Prepare the optimization state    
    state = OptimizationState(
        Oks_and_Eks_, 
        θ, 
        integrator; 
        sample_nr,
        solver,
        callback, 
        transform!,
        logger_funcs, 
        misc_restart,
        timer=timer, discard_outliers,
        maxiter, gradtol, verbosity
    )

    return evolve!(state)
end

function evolve!(state::OptimizationState)
    # Main optimization loop
    dynamic_kwargs = Dict()
    while step!(state, dynamic_kwargs)
    end

    if state.verbosity >= 2
        @info "evolve: Done"
        show(state.timer)
    end

    # Collect the results
    return state.energy, state.θ, get_misc(state)
end

function step!(o::OptimizationState, dynamic_kwargs)

    if o.niter-1 >= o.maxiter
        if o.verbosity >= 1
            @info "$(typeof(o.integrator)): Maximum number of iterations reached ($(o.niter))"
        end
        return false
    end
    t0 = time_ns()
    o.θ, natural_gradient = @timeit o.timer "integrator" o.integrator(o.θ, o.Oks_and_Eks; sample_nr=o.sample_nr, solver=o.solver, discard_outliers=o.discard_outliers, timer=o.timer, dynamic_kwargs...)
    dt = (time_ns() - t0) * 1e-9  # seconds
    
    # Transform the optimization state and natural_gradient
    natural_gradient = o.transform!(o, natural_gradient)
    
    # Extract energy and norms
    if natural_gradient isa CompositeNaturalGradient
        o.energy = natural_gradient.Es_mean
        norm_natgrad = norm(get_θdot(natural_gradient; θtype=eltype(o.θ)))
        norm_θ = norm(o.θ)

        # Update observer WITHOUT var_energy
        Observers.update!(o.history;
            natural_gradient,
            θ=o.θ,
            niter=o.niter,
            energy=o.energy,
            var_energy=NaN,  # var_energy is not well-defined for CompositeNaturalGradient
            norm_natgrad,
            norm_θ,
            natural_gradient.saved_properties...
        )

    else
        o.energy = real(mean(natural_gradient.Es))
        var_energy = real(var(natural_gradient.Es))
        norm_natgrad = norm(get_θdot(natural_gradient; θtype=eltype(o.θ)))
        norm_θ = norm(o.θ)

        Observers.update!(o.history;
            natural_gradient,
            θ=o.θ,
            niter=o.niter,
            energy=o.energy,
            var_energy,
            norm_natgrad,
            norm_θ,
            natural_gradient.saved_properties...
        )
    end

    stop = o.callback(;state=o, energy_value=o.energy, model=o.θ, misc=get_misc(o), niter=o.niter)
    
    if o.verbosity >= 2
        if natural_gradient isa CompositeNaturalGradient

            msg = "iter $(o.niter): E=$(o.energy) ± $(natural_gradient.Es_error), " *
                "‖θdot‖=$(norm_natgrad), ‖θ‖=$(norm_θ), time_per_step=$(round(dt, sigdigits=3))s " 

            if o.verbosity >= 3
                comp_lines = join([
                    "Component $i: $(ng.Es), " *
                    "‖θ̇‖=$(norm(get_θdot(ng)))"
                    for (i, ng) in enumerate(natural_gradient.terms)
                ], "\n")

                msg *= "\n" * comp_lines
            end

            @info msg

        else
            @info "iter $(o.niter): $(natural_gradient.Es), " *
                "‖θdot‖=$(norm_natgrad), ‖θ‖=$(norm_θ), time_per_step=$(round(dt, sigdigits=3))s, " *
                "tdvp_error=$(natural_gradient.tdvp_error)"
        end

        flush(stdout)
        flush(stderr)
    end
    if stop === false
        return false
    end

    if norm_natgrad < o.gradtol
        if o.verbosity >= 1
            @info "$(typeof(o.integrator)): Gradient tolerance reached"
        end
        return false
    end

    o.niter += 1
    return true
end

include("heun.jl")
include("decay.jl")