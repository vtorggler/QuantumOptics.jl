module stochastic_master

export master, master_dynamic

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using ...timeevolution
import ...timeevolution: integrate_stoch, recast!
import ...timeevolution.timeevolution_master: dmaster_h, dmaster_nh, dmaster_h_dynamic, check_master

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Void}

"""
    stochastic.master(tspan, rho0, H, Hs, J; <keyword arguments>)

Time-evolution according to a stochastic master equation.

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
function master(tspan, rho0::DenseOperator, H::Operator,
                J::Vector; Js::Vector=J, Hs::Union{Void, Vector}=nothing,
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

    n = length(Js) + (isa(Hs, Void) ? 0 : length(Hs))
    dmaster_stoch(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) = dmaster_stochastic(rho, Hs, rates_s, Js, Jsdagger, drho, tmp, index)

    isreducible = check_master(rho0, H, J, Jdagger, rates)
    check_master(rho0, H, Js, Jsdagger, rates_s)
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

Time-evolution according to a stochastic master equation with a
dynamic Hamiltonian and J.

There are two implementations for integrating the master equation with dynamic
operators:

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
function master_dynamic(tspan::Vector{Float64}, rho0::DenseOperator, fdeterm::Function, fstoch::Function;
                fstoch_H::Union{Function, Void}=nothing, fstoch_J::Union{Function, Void}=nothing,
                rates::DecayRates=nothing, rates_s::DecayRates=nothing,
                fout::Union{Function,Void}=nothing,
                kwargs...)

    tmp = copy(rho0)

    if isa(rates_s, Matrix{Float64})
        throw(ArgumentError("A matrix of stochastic rates is ambiguous! Please provide a vector of stochastic rates.
        You may want to use diagonaljumps."))
    end

    # TODO: Proper type inference for length
    fs_out = fstoch(0, rho0)
    n = length(fs_out[1])

    dmaster_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_h_dynamic(t, rho, fdeterm, rates, drho, tmp)
    if isa(fstoch_H, Void) && isa(fstoch_J, Void)
        dmaster_stoch_std(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) =
            dmaster_stoch_dynamic(t, rho, fstoch, rates_s, drho, tmp, index)

        integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_std, rho0, fout, n; kwargs...)
    else
        if isa(fstoch_H, Function)
            n += length(fstoch_H(0, rho0))
        end
        if isa(fstoch_J, Function)
            n += length(fstoch_J(0, rho0)[1])
        end
        dmaster_stoch_gen(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) =
            dmaster_stoch_dynamic_general(t, rho, fstoch, fstoch_H, fstoch_J,
                    rates, rates_s, drho, tmp, index)

        integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_gen, rho0, fout, n; kwargs...)
    end
end

function dmaster_stochastic(rho::DenseOperator, H::Void, rates::Void,
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    operators.gemm!(1, J[index], rho, 0, drho)
    operators.gemm!(1, rho, Jdagger[index], 1, drho)
    operators.gemm!(-expect(J[index]+Jdagger[index], rho), rho, one(J[index]), 1, drho)
    return drho
end
function dmaster_stochastic(rho::DenseOperator, H::Void, rates::Vector{Float64},
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    operators.gemm!(rates[index], J[index], rho, 0, drho)
    operators.gemm!(rates[index], rho, Jdagger[index], 1, drho)
    operators.gemm!(-rates[index]*expect(J[index]+Jdagger[index], rho), rho, one(J[index]), 1, drho)
    return drho
end

function dmaster_stochastic(rho::DenseOperator, H::Vector, rates::Void,
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    if index > length(J)
        operators.gemm!(-1.0im, H[index - length(J)], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index - length(J)], 1.0, drho)
    else
        operators.gemm!(1, J[index], rho, 0, drho)
        operators.gemm!(1, rho, Jdagger[index], 1, drho)
        operators.gemm!(-expect(J[index]+Jdagger[index], rho), rho, one(J[index]), 1, drho)
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
        operators.gemm!(rates[index], J[index], rho, 0, drho)
        operators.gemm!(rates[index], rho, Jdagger[index], 1, drho)
        operators.gemm!(-rates[index]*expect(J[index]+Jdagger[index], rho), rho, one(J[index]), 1, drho)
    end
    return drho
end

function dmaster_stoch_dynamic(t::Float64, rho::DenseOperator, f::Function, rates::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, index::Int)
    result = f(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_ = rates
    else
        J, Jdagger, rates_ = result
    end
    dmaster_stochastic(rho, nothing, rates_, J, Jdagger, drho, tmp, index)
end

function dmaster_stoch_dynamic_general(t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Function, fstoch_J::Void, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, index::Int)
    res_stoch = fstoch(t, rho)
    # TODO: Clean up by checking length of Hvec and else calling dmaster_stoch_dynamic
    H = fstoch_H(t, rho)
    @assert 2 <= length(res_stoch) <= 3
    if length(res_stoch) == 2
        J, Jdagger = res_stoch
        rates_s_ = rates_s
    else
        J, Jdagger, rates_s_ = res_stoch
    end
    dmaster_stochastic(rho, H, rates_s_, J, Jdagger, drho, tmp, index)
end
function dmaster_stoch_dynamic_general(t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Void, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, index::Int)
    res_stoch = fstoch(t, rho)
    # TODO: Clean up by checking length of J and else calling dmaster_stoch_dynamic
    if index <= length(res_stoch[1])
        @assert 2 <= length(res_stoch) <= 3
        if length(res_stoch) == 2
            J, Jdagger = res_stoch
            rates_s_ = rates_s
        else
            J, Jdagger, rates_s_ = res_stoch
        end
        dmaster_stochastic(rho, nothing, rates_s_, J, Jdagger, drho, tmp, index)
    else
        res_jumps = fstoch_J(t, rho)
        @assert 2 <= length(res_jumps) <= 3
        if length(res_jumps) == 2
            J_stoch, J_stoch_dagger = res_jumps
            rates_ = rates
        else
            J_stoch, J_stoch_dagger, rates_ = res_jumps
        end
        dlindblad(rho, rates_, J_stoch, J_stoch_dagger, drho, tmp, index-length(res_stoch[1]))
    end
end
function dmaster_stoch_dynamic_general(t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Function, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, index::Int)
    res_stoch = fstoch(t, rho)
    H = fstoch_H(t, rho)
    # TODO: Clean up by checking length of Hvec and J and else calling dmaster_stoch_dynamic
    if index <= length(res_stoch[1]) + length(H)
        @assert 2 <= length(res_stoch) <= 3
        if length(res_stoch) == 2
            J, Jdagger = res_stoch
            rates_s_ = rates_s
        else
            J, Jdagger, rates_s_ = res_stoch
        end
        dmaster_stochastic(rho, H, rates_s_, J, Jdagger, drho, tmp, index)
    else
        res_jumps = fstoch_J(t, rho)
        @assert 2 <= length(res_jumps) <= 3
        if length(res_jumps) == 2
            J_stoch, J_stoch_dagger = res_jumps
            rates_ = rates
        else
            J_stoch, J_stoch_dagger, rates_ = res_jumps
        end
        dlindblad(rho, rates_, J_stoch, J_stoch_dagger, drho, tmp, index-length(res_stoch[1])-length(H))
    end
end

function dlindblad(rho::DenseOperator, rates::Void, J::Vector, Jdagger::Vector,
    drho::DenseOperator, tmp::DenseOperator, i::Int)
    operators.gemm!(1, J[i], rho, 0, tmp)
    operators.gemm!(1, tmp, Jdagger[i], 1, drho)
    return drho
end
function dlindblad(rho::DenseOperator, rates::Vector{Float64}, J::Vector,
    Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator, i::Int)
    operators.gemm!(rates[i], J[i], rho, 0, tmp)
    operators.gemm!(1, tmp, Jdagger[i], 1, drho)
    return drho
end

function integrate_master_stoch(tspan, df::Function, dg::Function,
                        rho0::DenseOperator, fout::Union{Void, Function},
                        n::Int;
                        kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    x0 = reshape(rho0.data, length(rho0))
    state = copy(rho0)
    dstate = copy(rho0)
    integrate_stoch(tspan_, df, dg, x0, state, dstate, fout, n; kwargs...)
end

function recast!(x::SubArray{Complex128, 1}, rho::DenseOperator)
    rho.data = reshape(x, size(rho.data))
end
recast!(state::DenseOperator, x::SubArray{Complex128, 1}) = (x[:] = state.data)

end # module
