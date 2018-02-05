module stochastic

using ..bases, ..states, ..operators
using ..operators_dense, ..operators_sparse
using ..timeevolution
import ..timeevolution: integrate_stoch, recast!
import ..timeevolution.timeevolution_schroedinger: dschroedinger, dschroedinger_dynamic, check_schroedinger
import StochasticDiffEq

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Void}

"""
    stochastic.schroedinger(tspan, state0, H, Hs[; fout, ...])

Integrate stochastic Schrödinger equation with dynamic Hamiltonian.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `psi0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `H`: Deterministic part of the Hamiltonian.
* `Hs`: Stochastic part(s) of the Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger(tspan, psi0::Ket, H::Operator, Hs::Vector{T};
                fout::Union{Function,Void}=nothing,
                kwargs...) where T <: Operator
    tspan_ = convert(Vector{Float64}, tspan)

    n = length(Hs)
    dstate = copy(psi0)
    x0 = psi0.data
    state = copy(psi0)

    check_schroedinger(psi0, H)
    dschroedinger_determ(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger(psi, H, dpsi)
    dschroedinger_stoch(t::Float64, psi::Ket, dpsi::Ket, index::Int) = dschroedinger_stochastic(psi, Hs, dpsi, index)

    integrate_stoch(tspan_, dschroedinger_determ, dschroedinger_stoch, x0, state, dstate, fout, n; kwargs...)
end

function schroedinger(tspan, psi0::Ket, H::Operator, Hs::Operator;
                fout::Union{Function,Void}=nothing,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)

    dstate = copy(psi0)
    x0 = psi0.data
    state = copy(psi0)

    check_schroedinger(psi0, H)
    check_schroedinger(psi0, Hs)
    dschroedinger_determ(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger(psi, H, dpsi)
    dschroedinger_stoch(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger(psi, Hs, dpsi)

    integrate_stoch(tspan_, dschroedinger_determ, dschroedinger_stoch, x0, state, dstate, fout, nothing; kwargs...)
end

"""
    stochastic.schroedinger_dynamic(tspan, state0, fdeterm, fstoch[; fout, ...])

Integrate stochastic Schrödinger equation with dynamic Hamiltonian.

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

    stoch_type = pure_inference(fstoch, Tuple{eltype(tspan),typeof(psi0)})
    n = stoch_type <: Tuple ? nfields(stoch_type) : nothing

    dstate = copy(psi0)
    x0 = psi0.data
    state = copy(psi0)

    dschroedinger_determ(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger_dynamic(t, psi, fdeterm, dpsi)
    schroedinger_dynamic_(tspan, state, dstate, x0, dschroedinger_determ, fstoch, n; fout=fout, kwargs...)
end

function schroedinger_dynamic_(tspan::Vector{Float64}, state::Ket, dstate::Ket, x0::Vector{Complex128},
                dschroedinger_determ::Function, fstoch::Function, n::Void;
                fout::Union{Void, Function}=nothing, kwargs...)
    dschroedinger_stoch(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger_stochastic(t, psi, fstoch, dpsi)
    integrate_stoch(tspan, dschroedinger_determ, dschroedinger_stoch, x0, state, dstate, fout, n; kwargs...)
end
function schroedinger_dynamic_(tspan::Vector{Float64}, state::Ket, dstate::Ket, x0::Vector{Complex128},
                dschroedinger_determ::Function, fstoch::Function, n::Int;
                fout::Union{Void, Function}=nothing, kwargs...)
    dschroedinger_stoch(t::Float64, psi::Ket, dpsi::Ket, index::Int) = dschroedinger_stochastic(t, psi, fstoch, dpsi, index)
    integrate_stoch(tspan, dschroedinger_determ, dschroedinger_stoch, x0, state, dstate, fout, n; kwargs...)
end

function dschroedinger_stochastic(psi::Ket, Hs::Vector{T}, dpsi::Ket, index::Int) where T <: Operator
    check_schroedinger(psi, Hs[index])
    dschroedinger(psi, Hs[index], dpsi)
end

function dschroedinger_stochastic(t::Float64, psi::Ket, f::Function, dpsi::Ket)
    ops = f(t, psi)
    check_schroedinger(psi, ops)
    dschroedinger(psi, ops, dpsi)
end

function dschroedinger_stochastic(t::Float64, psi::Ket, f::Function, dpsi::Ket, index::Int)
    ops = f(t, psi)
    check_schroedinger(psi, ops[index])
    dschroedinger(psi, ops[index], dpsi)
end

recast!(dstate::Ket, dx::Array{Complex128, 2}) = nothing

Base.@pure pure_inference(fout,T) = Core.Inference.return_type(fout, T)


end # module
