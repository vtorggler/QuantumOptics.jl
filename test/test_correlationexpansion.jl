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


# Test time evolution
T = [0.:0.1:0.5;]

spinbasis = SpinBasis(1//2)
I = full(identityoperator(spinbasis))
sigmax = spin.sigmax(spinbasis)
sigmay = spin.sigmay(spinbasis)
sigmaz = spin.sigmaz(spinbasis)
sigmap = spin.sigmap(spinbasis)
sigmam = spin.sigmam(spinbasis)

N = 2
b = tensor([spinbasis for i=1:N]...)

S2 = correlationexpansion.correlationmasks(N, 2)
# S3 = correlationexpansion.correlationmasks(N, 3)

psi0 = normalize(spinup(spinbasis) + 0.5*spindown(spinbasis))
rho0 = correlationexpansion.ApproximateOperator([psi0⊗dagger(psi0) for i=1:N], S2)

Ω = [1. 2. 3.;
     2. 1. 4.;
     3. 4. 1.]
γ = 1.
δ = 0.2

Γ = eye(Complex128, N, N)

H = sum([lazy(LazyTensor(b, i, sigmaz, 0.5*δ)) for i=1:N])

for i=1:N, j=1:N
    if i==j
        continue
    end
    H += lazy(LazyTensor(b, [i, j], [sigmap, sigmam], Ω[i, j]))
end

J = LazyTensor[LazyTensor(b, i, sigmam, γ) for i=1:N]
Jdagger = LazyTensor[dagger(LazyTensor(b, i, sigmam, γ)) for i=1:N]

correlationexpansion.dmaster(rho0, H, Γ, J, Jdagger)


# tout, rho_t = cumulantexpansion.master(T, rho0, H, J)
# tout, rho_t_full = timeevolution.master(T, full(rho0), full(H), map(full, J))
