using Base.Test
using QuantumOptics

mask = correlationexpansion.indices2mask(3, [1,2])
@test mask == (true, true, false)
indices = correlationexpansion.mask2indices(mask)
@test indices == [1,2]

S1 = correlationexpansion.correlationmasks(3, 1)
S2 = correlationexpansion.correlationmasks(3, 2)
S3 = correlationexpansion.correlationmasks(3, 3)
@test S2 == Set([(true, true, false), (true, false, true), (false, true, true)])
@test S3 == Set([(true, true, true)])

b1 = FockBasis(2)
b2 = SpinBasis(1//2)
b3 = NLevelBasis(4)
b = tensor(b1, b2, b3)

rho = correlationexpansion.ApproximateOperator(b, b, S2 ∪ S3)

psi1a = normalize(coherentstate(b1, 0.1))
psi1b = fockstate(b1, 1)
psi2a = normalize(spinup(b2) + spindown(b2))
psi2b = normalize(spindown(b2))
psi3 = normalize(nlevelstate(b3, 1) + nlevelstate(b3, 2))
# psi = psi1 ⊗ psi2 ⊗ psi3

rho1 = 0.1*psi1a ⊗ dagger(psi1a) + 0.9*psi1b ⊗ dagger(psi1b)
rho2 = 0.3*psi2a ⊗ dagger(psi2a) + 0.7*psi2b ⊗ dagger(psi2b)
rho3 = psi3 ⊗ dagger(psi3)
rho = rho1 ⊗ rho2 ⊗ rho3
# rho = psi ⊗ dagger(psi)

x = correlationexpansion.approximate(rho, S2 ∪ S3)
for (c, sigma) in x.correlations
    println(correlationexpansion.mask2indices(c), ": ", sum(real(sigma.data)))
    # println(sum(abs(x.correlations[(true, true, true)].data)))
end

println(tracedistance(correlationexpansion.full(x), rho))
