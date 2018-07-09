module stochastic_definitions

export homodyne_carmichael

using ...operators, ...states

"""
    stochastic.homodyne_carmichael(H0, C, theta)

Helper function that defines the functions needed to compute homodyne detection
trajectories according to Carmichael with `stochastic.schroedinger_dynamic`.

# Arguments
* `H0`: The deterministic, time-independent system Hamiltonian.
* `C`: Collapse operator (or vector of operators) of the detected output channel(s).
* `theta`: The phase difference between the local oscillator and the signal field.
    Defines the operator of the measured quadrature as
    ``X_\\theta = C e^{-i\\theta} + C^\\dagger e^{i\\theta}``. Needs to be a
    vector of the same length as `C` if `C` is a vector.

Returns `(fdeterm, fstoch)`, where `fdeterm(t, psi) -> H` and
`fstoch(t, psi) -> Hs` are functions returning the deterministic and stochastic
part of the Hamiltonian required for calling `stochastic.schroedinger_dynamic`.

The deterministic and stochastic parts of the Hamiltonian are constructed as

```math
H_{det} = H_0 + H_{nl},
```

where

```math
H_{nl} = iCe^{-i\\theta} \\langle X \\rangle - \\frac{i}{2} C^\\dagger C,
```

and

```math
H_s = iCe^{-i\\theta}.
```
"""
function homodyne_carmichael(H0::Operator, C::Vector{T}, theta::Vector{R}) where {T <: Operator, R <: Real}
    @assert length(C) == length(theta)
    Hs = 1.0im*C .* exp.(-1.0im .* theta)
    X = Hs .+ dagger.(Hs)
    CdagC = -0.5im .* dagger.(C) .* C

    exp_ls = zeros(Complex128, length(X))
    function H_nl(psi::StateVector)
        @inbounds for i=1:length(X)
            exp_ls[i] = expect(X[i], psi)
        end
        sum(exp_ls .* Hs .+ CdagC)
    end

    fdeterm(t::Float64, psi::StateVector) = H0 + H_nl(psi)
    fstoch(t::Float64, psi::StateVector) = Hs
    return fdeterm, fstoch
end
homodyne_carmichael(H0::Operator, C::Operator, theta::Real) =
    homodyne_carmichael(H0, [C], [theta])

end # module
