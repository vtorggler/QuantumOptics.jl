using Base.Test
using QuantumOptics

@testset "stochastic" begin

b_spin = SpinBasis(1//2)
sz = sigmaz(b_spin)
sm = sigmam(b_spin)
sp = sigmap(b_spin)
zero_op = 0*sz

H = sp + sm
ψ0 = spindown(b_spin)

function fdeterm_atom(t, psi)
    H
end
function fstoch_atom(t, psi)
    zero_op
end
function fstoch_2(t, psi)
    zero_op, zero_op, zero_op
end

T = [0:0.1:1;]
dt = 1e-5
tout, ψt2 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm_atom, fstoch_2; dt=dt)
tout, ψt1 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm_atom, fstoch_atom; dt=dt)
@test ψt1 == ψt2

tout, ψt_determ = timeevolution.schroedinger_dynamic(T, ψ0, fdeterm_atom)
for i=1:length(tout)
    @test norm(ψt1[i] - ψt_determ[i]) < dt
end

end # testset
