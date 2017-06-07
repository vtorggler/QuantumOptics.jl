using BenchmarkTools
using QuantumOptics

N = 100
xmin = -5
xmax = 5
x0 = 0.3
p0 = -0.2
sigma0 = 1
bx = PositionBasis(xmin, xmax, N)
x = position(bx)
p = momentum(bx)
H = p^2 + 2*x^2
psi0 = gaussianstate(bx, x0, p0, sigma0)

T = [0:1.:10;]
exp_x = Float64[]
fout(t, psi) = push!(exp_x, real(expect(x, psi)))
timeevolution.schroedinger(T, psi0, H; fout=fout, reltol=1e-6, abstol=1e-8)

function f1(tspan, psi0, H)
    timeevolution.schroedinger(tspan, psi0, H)
end

function f2(tspan, psi0, H)
    timeevolution.schroedinger(tspan, psi0, H)
end

@time f1(T, psi0, H)
@time f1(T, psi0, H)
psi0_ = dagger(psi0)
@time f2(T, psi0_, H)
@time f2(T, psi0_, H)


# r1 = @benchmark f1($T, $psi0, $H)
# r2 = @benchmark f2($T, $psi0, $H)

# println(r1)
# println(r2)