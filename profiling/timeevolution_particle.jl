using QuantumOptics

N = 100
xmin = -10
xmax = 10
x0 = 2
p0 = 1
sigma0 = 1
bx = PositionBasis(xmin, xmax, N)
x = position(bx)
p = momentum(bx)
H = p^2 + full(2*x^2)
psi0 = gaussianstate(bx, x0, p0, sigma0)
T = [0:1.:10;]

function f1(N, psi0, H, x)
    for n=1:N
        exp_x = Float64[]
        fout(t, psi) = push!(exp_x, real(expect(x, psi)))
        timeevolution.schroedinger(T, psi0, H; fout=fout)
    end
end

@time f1(1, psi0, H, x)
@time f1(1, psi0, H, x)

Profile.clear()
@profile f1(20, psi0, H, x)

