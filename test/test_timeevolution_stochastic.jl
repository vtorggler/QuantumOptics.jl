using Base.Test
using QuantumOptics

@testset "stochastic" begin

b_spin = SpinBasis(1//2)
sz = sigmaz(b_spin)
sm = sigmam(b_spin)
sp = sigmap(b_spin)

H = sp + sm
ψ0 = spindown(b_spin)

function fdeterm_atom(t, psi)
    H
end
function fstoch_atom(t, psi)
    0*sz
end

T = [0:0.1:1;]
tout, ψt = timeevolution.schroedinger_stochastic(T, ψ0, fdeterm_atom, fstoch_atom)
tout, ψt2 = timeevolution.schroedinger_dynamic(T, ψ0, fdeterm_atom)

for i=1:length(tout)
    @test norm(ψt[i] - ψt2[i]) < 1e-6
end


b_fock = FockBasis(10)
a = destroy(b_fock)
ad = create(b_fock)


end # testset
