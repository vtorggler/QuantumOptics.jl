module stochastic_semiclassical

export schroedinger_semiclassical#, semiclassical.master_dynamic

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using ...semiclassical
import ...semiclassical: recast!, State
using ...timeevolution
import ...timeevolution: integrate_stoch
import ...timeevolution.timeevolution_schroedinger: dschroedinger, dschroedinger_dynamic
# import ...timeevolution.timeevolution_master: dmaster_h, dmaster_nh, dmaster_h_dynamic, check_master

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Void}
const QuantumState = Union{Ket, DenseOperator}
Base.@pure pure_inference(f, T) = Core.Inference.return_type(f, T)

"""
    semiclassical.schroedinger_stochastic(tspan, state0, fquantum, fclassical[; fout, ...])

Integrate time-dependent Schrödinger equation coupled to a classical system.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `psi0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `fquantum`: Function `f(t, psi, u) -> H` returning the time and or state
        dependent Hamiltonian.
* `fclassical`: Function `f(t, psi, u, du)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the vector `du`.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger_semiclassical(tspan, state0::State{Ket}, fquantum::Function,
                fclassical::Function; fstoch_quantum::Union{Void, Function}=nothing,
                fstoch_classical::Union{Void, Function}=nothing,
                fout::Union{Function,Void}=nothing,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    dschroedinger_det(t::Float64, state::State{Ket}, dstate::State{Ket}) = semiclassical.dschroedinger_dynamic(t, state, fquantum, fclassical, dstate)

    if isa(fstoch_quantum, Void) && isa(fstoch_classical, Void)
        warn("No stochastic functions provided!")
    end

    x0 = Vector{Complex128}(length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)

    n = 0
    if isa(fstoch_quantum, Function)
        stoch_type = pure_inference(fstoch_quantum, Tuple{eltype(tspan), typeof(state0.quantum), typeof(state0.classical)})
        n += stoch_type <: Tuple ? nfields(stoch_type) : 1
    end
    if isa(fstoch_classical, Function)
        n += 1
    end

    dschroedinger_stoch(t::Float64, state::State{Ket}, dstate::State{Ket}, index::Int) =
    if n == 1
        dschroedinger_stochastic(t, state, fstoch_quantum, fstoch_classical, dstate)
    else
        dschroedinger_stochastic(t, state, fstoch_quantum, fstoch_classical, dstate, index)
    end
    integrate_stoch(tspan_, dschroedinger_det, dschroedinger_stoch, x0, state, dstate, fout, n; kwargs...)
end

# """
#     stochastic.master_semiclassical(tspan, rho0, H, Hs, J; <keyword arguments>)
#
# Time-evolution according to a stochastic master equation.
#
# For dense arguments the `master` function calculates the
# non-hermitian Hamiltonian and then calls master_nh which is slightly faster.
#
# # Arguments
# * `tspan`: Vector specifying the points of time for which output should
#         be displayed.
# * `rho0`: Initial density operator. Can also be a state vector which is
#         automatically converted into a density operator.
# * `H`: Deterministic part of the Hamiltonian.
# * `Hs`: Operator or vector of operators specifying the stochastic part of the
#         Hamiltonian.
# * `J`: Vector containing all deterministic
#         jump operators which can be of any arbitrary operator type.
# * `Js`: Vector containing all stochastic jump operators.
# * `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
#         for the jump operators. If nothing is specified all rates are assumed
#         to be 1.
# * `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
#         operators. If they are not given they are calculated automatically.
# * `Jsdagger=dagger.(Js)`: Vector containing the hermitian conjugates of the
#         stochastic jump operators.
# * `fout=nothing`: If given, this function `fout(t, rho)` is called every time
#         an output should be displayed. ATTENTION: The given state rho is not
#         permanent! It is still in use by the ode solver and therefore must not
#         be changed.
# * `kwargs...`: Further arguments are passed on to the ode solver.
# """
# function master_semiclassical(tspan::Vector{Float64}, rho0::DenseOperator, fdeterm::Function, fstoch::Function;
#                 fstoch_H::Union{Function, Void}=nothing, fstoch_J::Union{Function, Void}=nothing,
#                 rates::DecayRates=nothing, rates_s::DecayRates=nothing,
#                 fout::Union{Function,Void}=nothing,
#                 kwargs...)
#
#     tmp = copy(rho0)
#
#     if rates_s == nothing && rates != nothing
#         rates_s = sqrt.(rates)
#     end
#     if isa(rates_s, Matrix{Float64})
#         throw(ArgumentError("A matrix of stochastic rates is ambiguous! Please provide a vector of stochastic rates.
#         You may want to set them as ones or use diagonaljumps."))
#     end
#
#     fs_out = fstoch(0, rho0)
#     n = length(fs_out[1])
#
#
#     dmaster_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_h_dynamic(t, rho, fdeterm, rates, drho, tmp)
#     if isa(fstoch_H, Void) && isa(fstoch_J, Void)
#         dmaster_stoch_std(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) =
#         dmaster_stoch_dynamic(t, rho, fstoch, rates_s, drho, tmp, index)
#         integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_std, rho0, fout, n; kwargs...)
#     else
#         if isa(fstoch_H, Function)
#             n += length(fstoch_H(0, rho0))
#         end
#         if isa(fstoch_J, Function)
#             n += length(fstoch_J(0, rho0)[1])
#         end
#         dmaster_stoch_gen(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) =
#         dmaster_stoch_dynamic_general(t, rho, fstoch, fstoch_H, fstoch_J, rates, rates_s, drho, tmp, index)
#         integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_gen, rho0, fout, n; kwargs...)
#     end
# end

function dschroedinger_stochastic(t::Float64, state::State{Ket}, fstoch_quantum::Function,
            fstoch_classical::Function, dstate::State{Ket}, index::Int)
    H = fstoch_quantum(t, state.quantum, state.classical)
    Hvec = isa(H, Operator) ? [H] : H
    if index <= length(Hvec)
        dschroedinger(state.quantum, Hvec[index], dstate.quantum)
    else
        fstoch_classical(t, state.quantum, state.classical, dstate.classical)
    end
end
function dschroedinger_stochastic(t::Float64, state::State{Ket}, fstoch_quantum::Function,
            fstoch_classical::Void, dstate::State{Ket})
    fquantum_(t, psi) = fstoch_quantum(t, state.quantum, state.classical)
    dschroedinger_dynamic(t, state.quantum, fquantum_, dstate.quantum)
end
function dschroedinger_stochastic(t::Float64, state::State{Ket}, fstoch_quantum::Function,
            fstoch_classical::Void, dstate::State{Ket}, index::Int)
    H = fstoch_quantum(t, state.quantum, state.classical)
    dschroedinger(t, state.quantum, H[index], dstate.quantum)
end
function dschroedinger_stochastic(t::Float64, state::State{Ket}, fstoch_quantum::Void,
            fstoch_classical::Function, dstate::State{Ket})
    fstoch_classical(t, state.quantum, state.classical, dstate.classical)
end


function recast!(state::State, x::SubArray{Complex128, 1})
    N = length(state.quantum)
    copy!(x, 1, state.quantum.data, 1, N)
    copy!(x, N+1, state.classical, 1, length(state.classical))
    x
end
function recast!(x::SubArray{Complex128, 1}, state::State)
    N = length(state.quantum)
    copy!(state.quantum.data, 1, x, 1, N)
    copy!(state.classical, 1, x, N+1, length(state.classical))
end

end # module
