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

T = [0:0.2:1;]
dt = 1/10.0

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
    noise_op, noise_op
end
tout, ψt5 = stochastic.schroedinger(T, ψ0, H, noise_op; dt=dt)
tout, ψt6 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm, fstoch_3; dt=dt)
for i=2:length(tout)
    @test norm(ψt5[i] - ψt_determ[i]) > 1e-3
    @test norm(ψt6[i] - ψt_determ[i]) > 1e-3
end

# Test master
ρ0 = dm(ψ0)
rates = [0.1]
J = [sm]
Js = [sm]
Hs = noise_op

tout, ρt1 = stochastic.master(T, ρ0, H, J; rates=rates, dt=dt)
tout, ρt2 = stochastic.master(T, ρ0, H, J; Hs=[Hs], rates=rates, dt=dt)

tout, ρt3 = stochastic.master(T, ρ0, H, J; Js=0.*J, dt=dt)
tout, ρt_determ = timeevolution.master(T, ρ0, H, J)

for i=2:length(tout)
    @test tracedistance(ρt1[i], ρt3[i]) > 1e-3
    @test tracedistance(ρt1[i], ρt_determ[i]) > 1e-3
end
for i=1:length(tout)
    @test tracedistance(ρt3[i], ρt_determ[i]) < dt
end

@test_throws ArgumentError stochastic.master(T, ρ0, H, [sm, sm]; rates=[0.1 0.1; 0.1 0.1], dt=dt)

# Test master dynamic
Jdagger = dagger.(J)
Js .*= rates
Jsdagger = dagger.(Js)
function fdeterm_master(t, rho)
    H, J, Jdagger
end
function fstoch1_master(t, rho)
    [zero_op], [zero_op]
end
function fstoch2_master(t, rho)
    Js, Jsdagger
end
function fstoch3_master(t, rho)
    [Hs]
end
function fstoch4_master(t, rho)
    J, Jdagger, rates
end

tout, ρt4 = stochastic.master_dynamic(T, ρ0, fdeterm_master, fstoch1_master; dt=dt)
tout, ρt5 = stochastic.master_dynamic(T, ρ0, fdeterm_master, fstoch2_master; dt=dt)
tout, ρt6 = stochastic.master_dynamic(T, ρ0, fdeterm_master, fstoch2_master; fstoch_H=fstoch3_master, dt=dt)
tout, ρt7 = stochastic.master_dynamic(T, ρ0, fdeterm_master, fstoch2_master; fstoch_J=fstoch4_master, dt=dt)
tout, ρt8 = stochastic.master_dynamic(T, ρ0, fdeterm_master, fstoch2_master; fstoch_H=fstoch3_master, fstoch_J=fstoch4_master, dt=dt)

for i=1:length(tout)
    @test tracedistance(ρt4[i], ρt_determ[i]) < dt
end
for i=2:length(tout)
    @test tracedistance(ρt5[i], ρt_determ[i]) > 1e-4
    @test tracedistance(ρt5[i], ρt_determ[i]) > 1e-4
    @test tracedistance(ρt6[i], ρt_determ[i]) > 1e-4
    @test tracedistance(ρt7[i], ρt_determ[i]) > 1e-4
    @test tracedistance(ρt8[i], ρt_determ[i]) > 1e-4
end

# Test semiclassical
u0 = Complex128[0.1, 0.5]
ψ_sc = semiclassical.State(ψ0, u0)
function fquantum(t, psi, u)
  return H
end
function fclassical(t, psi, u, du)
  du[1] = 3*u[2]
  du[2] = 5*sin(u[1])*cos(u[1])
end
function fquantum_stoch(t, psi, u)
    10Hs
end
function fclassical_stoch(t, psi, u, du)
    du[2] = 2*u[2]
end
tout, ψt_determ = semiclassical.schroedinger_dynamic(T, ψ_sc, fquantum, fclassical)
tout, ψt_sc1 = stochastic.schroedinger_semiclassical(T, ψ_sc, fquantum, fclassical; fstoch_quantum=fquantum_stoch, dt=dt)
tout, ψt_sc2 = stochastic.schroedinger_semiclassical(T, ψ_sc, fquantum, fclassical; fstoch_classical=fclassical_stoch, dt=dt)
tout, ψt_sc3 = stochastic.schroedinger_semiclassical(T, ψ_sc, fquantum, fclassical; fstoch_quantum=fquantum_stoch, fstoch_classical=fclassical_stoch, dt=dt)

# using PyPlot
# subplot(121)
# title("Classical")
# plot(tout, [p.classical[1] for p=ψt_determ], label="det")
# plot(tout, [p.classical[1] for p=ψt_sc1], label="q-noise")
# plot(tout, [p.classical[1] for p=ψt_sc2], label="c-noise")
# plot(tout, [p.classical[1] for p=ψt_sc3], label="both")
# legend()
# subplot(122)
# title("Quantum")
# plot(tout, expect(sp*sm, ψt_determ))
# plot(tout, expect(sp*sm, ψt_sc1))
# plot(tout, expect(sp*sm, ψt_sc2))
# plot(tout, expect(sp*sm, ψt_sc3))

end # testset
