module stochastic

using ..bases, ..states, ..operators
using ..operators_dense, ..operators_sparse
using ..timeevolution
import ..timeevolution: integrate_stoch, recast!
import ..timeevolution.timeevolution_schroedinger: dschroedinger, dschroedinger_dynamic, check_schroedinger
import ..timeevolution.timeevolution_master: dmaster_h, dmaster_nh, dmaster_h_dynamic, check_master
import StochasticDiffEq

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Void}
Base.@pure pure_inference(fout,T) = Core.Inference.return_type(fout, T)

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
function schroedinger(tspan, psi0::Ket, H::Operator, Hs::Vector;
                fout::Union{Function,Void}=nothing,
                kwargs...)
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


"""
    stochastic.master(tspan, rho0, H, Hs, J; <keyword arguments>)

Time-evolution according to a master equation.

There are two implementations for integrating the master equation:

* [`master_h`](@ref): Usual formulation of the master equation.
* [`master_nh`](@ref): Variant with non-hermitian Hamiltonian.

For dense arguments the `master` function calculates the
non-hermitian Hamiltonian and then calls master_nh which is slightly faster.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `H`: Deterministic part of the Hamiltonian.
* `Hs`: Operator or vector of operators specifying the stochastic part of the
        Hamiltonian.
* `J`: Vector containing all deterministic
        jump operators which can be of any arbitrary operator type.
* `Js`: Vector containing all stochastic jump operators.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
        operators. If they are not given they are calculated automatically.
* `Jsdagger=dagger.(Js)`: Vector containing the hermitian conjugates of the
        stochastic jump operators.
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master(tspan, rho0::DenseOperator, H::Operator, Hs::Operator,
                J::Vector, Js::Vector;
                rates::DecayRates=nothing, rates_s::DecayRates=nothing,
                Jdagger::Vector=dagger.(J), Jsdagger::Vector=dagger.(Js),
                fout::Union{Function,Void}=nothing,
                kwargs...)

    tmp = copy(rho0)

    if rates_s == nothing && rates != nothing
        rates_s = sqrt.(rates)
    end
    if isa(rates_s, Matrix{Float64})
        throw(ArgumentError("A matrix of stochastic rates is ambiguous! Please provide a vector of stochastic rates.
        You may want to use diagonaljumps."))
    end

    n = length(Js) + 1
    dmaster_stoch(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) = dmaster_stochastic(rho, Hs, rates_s, Js, Jsdagger, drho, tmp, index)

    isreducible = check_master(rho0, H, J, Jdagger, rates)
    check_master(rho0, Hs, Js, Jsdagger, rates_s)
    if !isreducible
        dmaster_h_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
        integrate_master_stoch(tspan, dmaster_h_determ, dmaster_stoch, rho0, fout, n; kwargs...)
    else
        Hnh = copy(H)
        if typeof(rates) == Matrix{Float64}
            for i=1:length(J), j=1:length(J)
                Hnh -= 0.5im*rates[i,j]*Jdagger[i]*J[j]
            end
        elseif typeof(rates) == Vector{Float64}
            for i=1:length(J)
                Hnh -= 0.5im*rates[i]*Jdagger[i]*J[i]
            end
        else
            for i=1:length(J)
                Hnh -= 0.5im*Jdagger[i]*J[i]
            end
        end
        Hnhdagger = dagger(Hnh)

        dmaster_nh_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
        integrate_master_stoch(tspan, dmaster_nh_determ, dmaster_stoch, rho0, fout, n; kwargs...)
    end
end

function master(tspan, rho0::DenseOperator, H::Operator, Hs::Vector,
                J::Vector, Js::Vector;
                rates::DecayRates=nothing, rates_s::DecayRates=nothing,
                Jdagger::Vector=dagger.(J), Jsdagger::Vector=dagger.(Js),
                fout::Union{Function,Void}=nothing,
                kwargs...)

    tmp = copy(rho0)

    if rates_s == nothing && rates != nothing
        rates_s = sqrt.(rates)
    end
    if isa(rates_s, Matrix{Float64})
        throw(ArgumentError("A matrix of stochastic rates is ambiguous! Please provide a vector of stochastic rates.
        You may want to use diagonaljumps."))
    end

    n = length(Js) + length(Hs)
    dmaster_stoch(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) = dmaster_stochastic(rho, Hs, rates_s, Js, Jsdagger, drho, tmp, index)

    isreducible = check_master(rho0, H, J, Jdagger, rates)
    (check_master(rho0, h, Js, Jsdagger, rates_s) for h=Hs)
    if !isreducible
        dmaster_h_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
        integrate_master_stoch(tspan, dmaster_h_determ, dmaster_stoch, rho0, fout, n; kwargs...)
    else
        Hnh = copy(H)
        if typeof(rates) == Matrix{Float64}
            for i=1:length(J), j=1:length(J)
                Hnh -= 0.5im*rates[i,j]*Jdagger[i]*J[j]
            end
        elseif typeof(rates) == Vector{Float64}
            for i=1:length(J)
                Hnh -= 0.5im*rates[i]*Jdagger[i]*J[i]
            end
        else
            for i=1:length(J)
                Hnh -= 0.5im*Jdagger[i]*J[i]
            end
        end
        Hnhdagger = dagger(Hnh)

        dmaster_nh_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
        integrate_master_stoch(tspan, dmaster_nh_determ, dmaster_stoch, rho0, fout, n; kwargs...)
    end
end

"""
    stochastic.master_dynamic(tspan, rho0, f, fs; <keyword arguments>)

Time-evolution according to a master equation with a dynamic Hamiltonian and J.

There are two implementations for integrating the master equation with dynamic
operators:

* [`master_dynamic`](@ref): Usual formulation of the master equation.
* [`master_nh_dynamic`](@ref): Variant with non-hermitian Hamiltonian.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `f`: Function `f(t, rho) -> (H, J, Jdagger)` or `f(t, rho) -> (H, J, Jdagger, rates)`
* `fs`: Function `fs(t, rho) -> (Hs, Js, Jsdagger)` or `fs(t, rho) -> (Hs, Js, Jsdagger, rates)`
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.
* `rates_s=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the stochastic jump operators. If nothing is specified all rates are assumed
        to be 1.
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_dynamic(tspan, rho0::DenseOperator, f::Function, fs::Function;
                rates::DecayRates=nothing, rates_s::DecayRates=nothing,
                fout::Union{Function,Void}=nothing,
                kwargs...)

    tmp = copy(rho0)

    if rates_s == nothing && rates != nothing
        rates_s = sqrt.(rates)
    end
    if isa(rates_s, Matrix{Float64})
        throw(ArgumentError("A matrix of stochastic rates is ambiguous! Please provide a vector of stochastic rates.
        You may want to use diagonaljumps."))
    end

    fs_out = fs(0, rho0)
    n = length(fs_out[1]) + length(fs_out[2])
    no_rates = length(fs_out) == 3

    dmaster_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_h_dynamic(t, rho, f, rates, drho, tmp)
    dmaster_stoch(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) = if no_rates
        dmaster_stoch_dynamic_norates(t, rho, fs, rates_s, drho, tmp, index)
    else
        dmaster_stoch_dynamic_rates(t, rho, fs, rates_s, drho, tmp, index)
    end
    integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch, rho0, fout, n; kwargs...)
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


function dmaster_stochastic(rho::DenseOperator, H::Operator, rates::Void,
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    if index == length(J) + 1
        operators.gemm!(-1.0im, H, rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H, 1.0, drho)
    else
        operators.gemm!(1, J[index], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[index], 1, drho)

        operators.gemm!(-0.5, Jdagger[index], tmp, 1, drho)

        operators.gemm!(1., rho, Jdagger[index], 0, tmp)
        operators.gemm!(-0.5, tmp, J[index], 1, drho)
    end
    return drho
end
function dmaster_stochastic(rho::DenseOperator, H::Operator, rates::Vector{Float64},
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    if index == length(J) + 1
        operators.gemm!(-1.0im, H, rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H, 1.0, drho)
    else
        operators.gemm!(rates[index], J[index], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[index], 1, drho)

        operators.gemm!(-0.5, Jdagger[index], tmp, 1, drho)

        operators.gemm!(rates[index], rho, Jdagger[index], 0, tmp)
        operators.gemm!(-0.5, tmp, J[index], 1, drho)
    end
    return drho
end

function dmaster_stochastic(rho::DenseOperator, H::Vector, rates::Void,
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    if index > length(J)
        operators.gemm!(-1.0im, H[index - length(J)], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index - length(J)], 1.0, drho)
    else
        operators.gemm!(1, J[index], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[index], 1, drho)

        operators.gemm!(-0.5, Jdagger[index], tmp, 1, drho)

        operators.gemm!(1., rho, Jdagger[index], 0, tmp)
        operators.gemm!(-0.5, tmp, J[index], 1, drho)
    end
    return drho
end
function dmaster_stochastic(rho::DenseOperator, H::Vector, rates::Vector{Float64},
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    if index > length(J)
        operators.gemm!(-1.0im, H[index - length(J)], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index - length(J)], 1.0, drho)
    else
        operators.gemm!(rates[index], J[index], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[index], 1, drho)

        operators.gemm!(-0.5, Jdagger[index], tmp, 1, drho)

        operators.gemm!(rates[index], rho, Jdagger[index], 0, tmp)
        operators.gemm!(-0.5, tmp, J[index], 1, drho)
    end
    return drho
end

function dmaster_stoch_dynamic_norates(t::Float64, rho::DenseOperator, f::Function, rates::Vector{Float64},
            drho::DenseOperator, tmp::DenseOperator, index::Int)

    result = f(t, rho)
    @assert 3 == length(result)
    H, J, Jdagger = result
    if index > length(J)
        operators.gemm!(-1.0im, H[index - length(J)], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index - length(J)], 1.0, drho)
    else
        operators.gemm!(rates[index], J[index], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[index], 1, drho)

        operators.gemm!(-0.5, Jdagger[index], tmp, 1, drho)

        operators.gemm!(rates[index], rho, Jdagger[index], 0, tmp)
        operators.gemm!(-0.5, tmp, J[index], 1, drho)
    end
    return drho
end
function dmaster_stoch_dynamic_norates(t::Float64, rho::DenseOperator, f::Function, rates::Void,
            drho::DenseOperator, tmp::DenseOperator, index::Int)

    result = f(t, rho)
    @assert 3 == length(result)
    H, J, Jdagger = result
    if index > length(J)
        operators.gemm!(-1.0im, H[index - length(J)], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index - length(J)], 1.0, drho)
    else
        operators.gemm!(1, J[index], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[index], 1, drho)

        operators.gemm!(-0.5, Jdagger[index], tmp, 1, drho)

        operators.gemm!(1, rho, Jdagger[index], 0, tmp)
        operators.gemm!(-0.5, tmp, J[index], 1, drho)
    end
    return drho
end

function dmaster_stoch_rates(t::Float64, rho::DenseOperator, f::Function, rates::Void,
            drho::DenseOperator, tmp::DenseOperator, index::Int)
    result = f(t, rho)
    @assert length(result) == 4
    H, J, Jdagger, rates_ = result
    if index > length(J)
        operators.gemm!(-1.0im, H[index - length(J)], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index - length(J)], 1.0, drho)
    else
        operators.gemm!(rates_[index], J[index], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[index], 1, drho)

        operators.gemm!(-0.5, Jdagger[index], tmp, 1, drho)

        operators.gemm!(rates_[index], rho, Jdagger[index], 0, tmp)
        operators.gemm!(-0.5, tmp, J[index], 1, drho)
    end
    return drho
end



function integrate_master_stoch(tspan, df::Function, dg::Function,
                        rho0::DenseOperator, fout::Union{Void, Function},
                        n::Int;
                        kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    x0 = reshape(rho0.data, length(rho0))
    state = DenseOperator(rho0.basis_l, rho0.basis_r, rho0.data)
    dstate = DenseOperator(rho0.basis_l, rho0.basis_r, rho0.data)
    integrate_stoch(tspan_, df, dg, x0, state, dstate, fout, n; kwargs...)
end

recast!(dstate::Ket, dx::Array{Complex128, 2}) = nothing
recast!(dstate::Operator, dx::Array{Complex128, 2}) = nothing

end # module
