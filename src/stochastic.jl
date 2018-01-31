module stochastic

using ..bases, ..states, ..operators
using ..operators_dense, ..operators_sparse
using ..timeevolution
import ..timeevolution: integrate_stoch, timeevolution_schroedinger.dschroedinger_dynamic
#using ...operators_lazysum, ...operators_lazytensor, ...operators_lazyproduct
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
    dschroedinger_stoch(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger_dynamic(t, psi, fstoch, dpsi)
    x0 = psi0.data
    state = copy(psi0)
    dstate = copy(psi0)
    integrate_stoch(tspan_, dschroedinger_determ, dschroedinger_stoch, x0, state, dstate, fout; kwargs...)
end


end # module
