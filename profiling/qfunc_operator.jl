using BenchmarkTools
using QuantumOptics

function _qfunc_operator(rho::Operator, alpha::Complex128, tmp1::Ket, tmp2::Ket)
    coherentstate(basis(rho), alpha, tmp1)
    operators.gemv!(complex(1.), rho, tmp1, complex(0.), tmp2)
    a = dot(tmp1.data, tmp2.data)
    return a/pi
end

function qfunc1(rho::Operator, xvec::Vector{Float64}, yvec::Vector{Float64})
    b = basis(rho)
    @assert isa(b, FockBasis)
    Nx = length(xvec)
    Ny = length(yvec)
    tmp1 = Ket(b)
    tmp2 = Ket(b)
    result = Matrix{Complex128}(Nx, Ny)
    @inbounds for j=1:Ny, i=1:Nx
        result[i, j] = _qfunc_operator(rho, complex(xvec[i], yvec[j])/sqrt(2), tmp1, tmp2)
    end
    result
end

function qfunc2(rho::Operator, xvec::Vector{Float64}, yvec::Vector{Float64})
    b = basis(rho)
    @assert isa(b, FockBasis)
    Nx = length(xvec)
    Ny = length(yvec)
    points = Nx*Ny
    N = b.N::Int
    f_ = Vector{Float64}(N+1)
    f_[1] = 1.
    for n=1:N
        f_[n+1] = f_[n]/sqrt(n)
    end
    _alpha = [complex(x, y)/sqrt(2) for x=xvec, y=yvec]
    _conj_alpha = conj(_alpha)
    q = zeros(_alpha)
    qtmp = similar(q)
    @inbounds for n=N+1:-1:1
        fill!(qtmp, rho.data[end, n]*f_[end])
        for m=N:-1:1
            fm_ = f_[m]
            rho_mn = rho.data[m,n]*fm_
            for i=1:points
                qtmp[i] = rho_mn + qtmp[i]*_conj_alpha[i]
            end
        end
        fn_ = f_[n]
        for i=1:points
            q[i] = qtmp[i]*fn_ + q[i]*_alpha[i]
        end
    end
    result = similar(q, Float64)
    @inbounds for i=1:points
        result[i] = real(q[i])*exp(-abs2(_conj_alpha[i]))/pi
    end
    result
end


N = 100
b = FockBasis(N)
psi = coherentstate(b, 0.7+0.1im)
rho = dm(psi)
X = collect(linspace(-1, 1, 100))
Y = collect(linspace(-2, 1, 100))

# @code_warntype qfunc2(rho, X, Y)

q = qfunc1(rho, X, Y)
println(q[1])
q = qfunc2(rho, X, Y)
println(q[1])
# println(qfunc(psi, -1, -2))

function timings()
    @time qfunc1(rho, X, Y)
    @time qfunc1(rho, X, Y)
    @time qfunc2(rho, X, Y)
    @time qfunc2(rho, X, Y)
end
timings()

# function run_qfunc1(N, rho, X, Y)
#     for i in 1:N
#         qfunc1(rho, X, Y)
#     end
# end

# function run_qfunc2(N, rho, X, Y)
#     for i in 1:N
#         qfunc2(rho, X, Y)
#     end
# end

# Profile.clear()
# @profile run_qfunc2(50, rho, X, Y)

# tmp1=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1]))
# tmp2=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1]))


# r1 = @benchmark qfunc1($rho, $X, $Y)
# r2 = @benchmark qfunc2($rho, $X, $Y)

# println(r1)
# println(r2)
