using Base.Test
using QuantumOptics

@testset "stochastic_definitions" begin

n=20
b=FockBasis(n)
psi0 = fockstate(b, 0)
a = destroy(b)
ad = dagger(a)
H0 = ad*a

fdeterm, fstoch = stochastic.homodyne_carmichael(H0, a, 0.5π)
Y = 1.0im*(ad - a)
@test fdeterm(0.0, psi0) == H0 + expect(Y, psi0)*a - 0.5im*ad*a
@test expect(fstoch(0.0, psi0)[1], psi0) == 0.0

end # testset
