module stochastic

using ..bases, ..states, ..operators
using ..operators_dense, ..operators_sparse
using ..timeevolution
import ..timeevolution.integrate_stoch
import ..timeevolution.timeevolution_schroedinger: dschroedinger, dschroedinger_dynamic, check_schroedinger
import StochasticDiffEq

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Void}

"""
    stochastic.schroedinger_dynamic(tspan, state0, fdeterm, fstoch[; fout, ...])

Integrate stochastic Schrödinger equation.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `psi0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `fdeterm`: Function `f(t, psi, u) -> H` returning the deterministic
    (time- or state-dependent) part of the Hamiltonian.
* `fstoch`: Function or vector of functions `f(t, psi, u, du)` returning the stochastic part
    of the Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger_dynamic(tspan, psi0::Ket, fdeterm::Function, fstoch::Function;
                fout::Union{Function,Void}=nothing,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    dschroedinger_determ(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger_dynamic(t, psi, fdeterm, dpsi)
    dschroedinger_stoch(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger_stochastic(t, psi, fstoch, dpsi)
    x0 = psi0.data
    state = copy(psi0)
    dstate = copy(psi0)
    integrate_stoch(tspan_, dschroedinger_determ, dschroedinger_stoch, x0, state, dstate, fout; kwargs...)
end

function dschroedinger_stochastic(t::Float64, psi::Ket, f::Function, dpsi::Ket)
    ops = f(t, psi)
    if isa(ops, Operator)
        check_schroedinger(psi, ops)
        dschroedinger(psi, ops, dpsi)
    elseif isa(ops, Tuple)
        out = Array{Ket}(1, length(ops))
        @inbounds for i=1:length(ops)
            check_schroedinger(psi, ops[i])
            out[1, i] = dschroedinger(psi, ops[i], dpsi)
        end
        out
    else
        ArgumentError("fstoch returns invlaid type $(typeof(ops))!")
    end
end

end # module
