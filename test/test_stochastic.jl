using Base.Test
using QuantumOptics

@testset "stochastic" begin

b_spin = SpinBasis(1//2)
sz = sigmaz(b_spin)
sm = sigmam(b_spin)
sp = sigmap(b_spin)
zero_op = 0*sz
noise_op = 0.1*sz

H = sp + sm
ψ0 = spindown(b_spin)

function fdeterm(t, psi)
    H
end
# Test equivalence to Schrödinger equation with zero noise
function fstoch_1(t, psi)
    zero_op
end
function fstoch_2(t, psi)
    zero_op, zero_op, zero_op
end

T = [0:0.1:1;]
dt = 1e-5
# Non-dynamic Schrödinger
tout, ψt4 = stochastic.schroedinger(T, ψ0, H, [zero_op, zero_op]; dt=dt)
tout, ψt3 = stochastic.schroedinger(T, ψ0, H, zero_op; dt=dt)
# Dynamic Schrödinger
tout, ψt1 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm, fstoch_1; dt=dt)
tout, ψt2 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm, fstoch_2; dt=dt)

# Test sharp equality for same algorithms
@test ψt1 == ψt3
@test ψt2 == ψt4

tout, ψt_determ = timeevolution.schroedinger_dynamic(T, ψ0, fdeterm)
# Test approximate equality for different algorithms
for i=1:length(tout)
    @test norm(ψt1[i] - ψt2[i]) < dt
    @test norm(ψt1[i] - ψt_determ[i]) < dt
end

# Test with non-zero noise
function fstoch_3(t, psi)
    noise_op, zero_op
end
tout, ψt5 = stochastic.schroedinger(T, ψ0, H, noise_op; dt=dt)
tout, ψt6 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm, fstoch_3; dt=dt)
for i=2:length(tout)
    @test norm(ψt5[i] - ψt_determ[i]) > dt
    @test norm(ψt5[i] - ψt6[i]) > dt
end

end # testset
