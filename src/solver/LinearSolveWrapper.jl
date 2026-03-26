using LinearSolve

mutable struct LinearSolveWrapper <: QuantumNaturalGradient.AbstractSolver
    diagshift::Real
    alg
    verbose::Bool
    save_info::Bool
    info
    LinearSolveWrapper(diagshift::Real=1e-5, alg=KrylovJL_CG(); verbose=true, save_info=false) = new(diagshift, alg, verbose, save_info, nothing)
end

function (solver::LinearSolveWrapper)(M::AbstractMatrix, v::AbstractVector)
    T = promote_type(eltype(M), eltype(v))
    vv = eltype(v) === T ? v : convert(Vector{T}, v)

    shift = solver.diagshift
    if eltype(solver.diagshift) != T
        solver.diagshift =T.(solver.diagshift)
    end
    sol = nothing

    for _ in 1:3
        Mreg = Hermitian(M + shift * I)
        prob = LinearProblem(Mreg, vv)
        sol = solve(prob, solver.alg)

        if  string(sol.retcode) == "Success"
            if solver.save_info
                solver.info = sol
            end
            solver.diagshift = shift
            return sol.u
        end

        if solver.verbose
            @warn "Linear solve failed, increasing diagshift" retcode=sol.retcode diagshift=shift alg=solver.alg
        end
        shift *= 10
        if shift > 2e-1
            @warn "Diagshift exceeded 2e-1, stopping attempts" diagshift=shift
            error(" Maximum diagshift exceeded without successful linear solve.")
        end
    end

    if solver.save_info
        solver.info = sol
    end

    error("Linear solve failed even after increasing diagshift")
end
