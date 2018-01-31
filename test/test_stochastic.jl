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
tout, ψt1 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm_atom, fstoch_atom)
tout, ψt2 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm_atom, fstoch_2)
@test ψt1 == ψt2

tout, ψt_determ = timeevolution.schroedinger_dynamic(T, ψ0, fdeterm_atom)
for i=1:length(tout)
    @test norm(ψt1[i] - ψt_determ[i]) < 1e-6
end


# Homodyne detection
# b_fock = FockBasis(10)
# a = destroy(b_fock)
# ad = create(b_fock)
# κ = 1.0
#
# # for θ = 0, λ = 1
# H_determ = 1.0im*κ/2.0*(ad^2 - a^2) - 1.0im*κ*ad*a
# H_determ2 = 1.0im*2κ*a
# X = ad + a
# function fdeterm_hom(t, psi)
#     H_determ + H_determ2*expect(X, psi)
# end
# sq_a = sqrt(2κ)*a
# function fstoch_hom(t, psi)
#     sq_a
# end
#
# ψ0 = fockstate(b_fock, 0)
# T = [0:0.1:10;]
# tout, ψt = stochastic.schroedinger_dynamic(T, ψ0, fdeterm_hom, fstoch_hom)
# x = real(expect(X, ψt))
#
# using PyPlot
# plot(x)
# plot(expect(ad*a, ψt))

end # testset
