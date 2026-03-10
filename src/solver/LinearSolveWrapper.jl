using LinearSolve

mutable struct LinearSolveWrapper <: QuantumNaturalGradient.AbstractSolver
    diagshift::Float64
    alg
    verbose::Bool
    save_info::Bool
    info
    LinearSolveWrapper(diagshift::Float64=1e-5, alg=KrylovJL_CG(); verbose=true, save_info=false) = new(diagshift, alg, verbose, save_info, nothing)
end


# function (solver::LinearSolveWrapper)(M::AbstractMatrix, v::AbstractArray)
#     #@assert ishermitian(M) "LinearSolveWrapper: M is not Hermitian"
#     if solver.diagshift != 0
#         M = Hermitian(M + I(size(M, 1)) * solver.diagshift)
#     else
#         M = Hermitian(M)
#     end

#     if eltype(M)!= eltype(v)
#         if eltype(M) == ComplexF64 
#             v = ComplexF64.(v)
#         end
#     end
    
#     prob = LinearProblem(M, v)
#     sol = solve(prob, solver.alg)
#     if solver.save_info
#         solver.info = sol
#     end
#     return sol.u
# end

function (solver::LinearSolveWrapper)(M::AbstractMatrix, v::AbstractVector)
    T = promote_type(eltype(M), eltype(v))
    vv = eltype(v) === T ? v : convert(Vector{T}, v)

    shift = solver.diagshift
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
    end

    if solver.save_info
        solver.info = sol
    end

    error("Linear solve failed even after increasing diagshift")
end