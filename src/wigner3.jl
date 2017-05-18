using QuantumOptics


function clenshaw_laguerre{T<:Number}(α::Int, x::Float64, a::Vector{T})
    n = length(a)-1
    ϕ1 = 1 + α - x
    if n==0
        return a[1]
    elseif n==1
        return a[1] + a[2]*ϕ1
    end
    b2 = 0.
    b1 = 0.
    b0 = 0.
    @inbounds for k=n:-1:1
        b2 = b1
        b1 = b0
        A = (2*k + ϕ1)/(k+1)
        B = -(k+1+α)/(k+2)
        b0 = a[k+1] + A*b1 + B*b2
    end
    B1 = -(1 + α)/2
    return a[1] + ϕ1*b0 + B1*b1
end

function clenshaw_wigner{T<:Number}(α::Int, x::Float64, a::Vector{T})
    n = length(a)-1
    ϕ1 = -(α+1-x)/sqrt(α+1)
    if n==0
        return a[1]
    elseif n==1
        return a[1] + a[2]*ϕ1
    end
    b2 = 0.
    b1 = 0.
    b0 = 0.
    @inbounds for k=n:-1:1
        b2 = b1
        b1 = b0
        A = -(2*k + 1 + α - x)/sqrt((k+α+1)*(k+1))
        B = -sqrt((k+1+α)*(k+1)/((k+2)*(α+k+2)))
        b0 = a[k+1] + A*b1 + B*b2
    end
    B1 = -sqrt((α+1)/(2*(α+2)))
    return a[1] + ϕ1*b0 + B1*b1
end

α = 1
x = 0.3
# n = 3
# a = zeros(Float64, n+1)
# a[end] = 1
# a = [0.4, 0.3, 0.1, 0.7]
# println(clenshaw_laguerre(α, x, a))
# L0(α, x) = 1.
# L1(α, x) = -x + α + 1.
# L2(α, x) = x^2/2 - (α+2)*x + (α+2)*(α+1)/2
# L3(α, x) = -x^3/6 + (α+3)*x^2/2 - (α+2)*(α+3)*x/2 + (α+1)*(α+2)*(α+3)/6
# # println(L3(α, x))
# println(a[1]*L0(α, x)+ a[2]*L1(α, x) + a[3]*L2(α, x) + a[4]*L3(α, x))

# a_wigner = [0, 0, 1]
# a_laguerre = [(-1)^n*sqrt(factorial(α)*factorial(n)/factorial(α+n))*a_wigner[n+1] for n=0:length(a_wigner)-1]
# r1 = clenshaw_wigner(α, x, a_wigner)
# r2 = clenshaw_laguerre(α, x, a_laguerre)

# println(r1)
# println(r2)


function wigner(rho, x, y)
    b = basis(rho)
    @assert isa(b, FockBasis)
    α = complex(x, y)/sqrt(2)
    w = complex(0.)
    coefficients = zeros(Complex128, b.N+1)
    for L=b.N:-1:0
        D = diag(rho.data, L)
        for n=0:b.N-L
            D[n+1] *= (-1)^n*sqrt(factorial(n)/factorial(L+n))
        end
        if L==0
            coefficients[L+1] = clenshaw_laguerre(L, abs2(2*α), D)
        else
            coefficients[L+1] = 2*clenshaw_laguerre(L, abs2(2*α), D)
        end
    end
    println("coeffs: ", coefficients)
    exp(-2*abs2(α))/pi*real(QuantumOptics.polynomials.horner(coefficients, 2*α))
end

function wigner2(rho, x, y)
    b = basis(rho)
    @assert isa(b, FockBasis)
    α = complex(x, y)/sqrt(2)
    w = complex(0.)
    coefficients = zeros(Complex128, b.N+1)
    @inbounds for L=b.N:-1:0
        D = diag(rho.data, L)
        fac = 1.
        for n=0:b.N-L
            D[n+1] *= fac
            fac *= -sqrt((n+1)/(L+n+1))
        end
        if L==0
            coefficient = clenshaw_laguerre(L, abs2(2*α), D)
        else
            coefficient = 2*clenshaw_laguerre(L, abs2(2*α), D)
        end
        w = coefficient + w*(2*α)/sqrt(L+1)
    end
    exp(-2*abs2(α))/pi*real(w)
end

function wigner3(rho, x, y)
    b = basis(rho)
    @assert isa(b, FockBasis)
    α = complex(x, y)/sqrt(2)
    w = complex(0.)
    @inbounds for L=b.N:-1:0
        D = diag(rho.data, L)
        if L==0
            coefficient = clenshaw_wigner(L, abs2(2*α), D)
        else
            coefficient = 2*clenshaw_wigner(L, abs2(2*α), D)
        end
        w = coefficient + w*(2*α)/sqrt(L+1)
    end
    exp(-2*abs2(α))/pi*real(w)
end

function clenshaw_wigner2(N::Int, α::Int, x::Float64, a::Matrix{Complex128})
    # n = length(a)-1
    n = N-α
    ϕ1 = -(α+1-x)/sqrt(α+1)
    if n==0
        return a[1, α+1]
    elseif n==1
        return a[1, α+1] + a[2, α+2]*ϕ1
    end
    f0 = sqrt(float((n+α)*(n)))
    f1 = 1.
    f0_ = 1/f0
    f1_ = 1.
    b2 = complex(0.)
    b1 = complex(0.)
    b0 = a[n+1, α+n+1]
    @inbounds for k=n-1:-1:1
        b2 = b1
        b1 = b0
        A = -(2*k + 1 + α - x)*f0_
        B = -f0*f1_
        f1 = f0
        f1_ = f0_
        f0 = sqrt((k+α)*k)
        f0_ = 1/f0
        b0 = a[k+1, α+k+1] + A*b1 + B*b2
    end
    B1 = -sqrt((α+1)/(2*(α+2)))
    return a[1, α+1] + ϕ1*b0 + B1*b1
end

function clenshaw_wigner3(N::Int, α::Int, x::Float64, a::Matrix{Complex128})
    # n = length(a)-1
    n = N-α
    ϕ1 = -(α+1-x)/sqrt(α+1)
    if n==0
        return a[1, α+1]
    elseif n==1
        return a[1, α+1] + a[2, α+2]*ϕ1
    end
    f0 = sqrt(float((n+α-1)*(n-1)))
    f1 = sqrt(float((n+α)*n))
    f0_ = 1/f0
    f1_ = 1/f1
    b2 = complex(0.)
    b1 = a[n+1, α+n+1]
    A = -(2*n-1+α-x)/f1
    b0 = a[n, α+n] + A*b1
    @inbounds for k=n-2:-1:1
        b2 = b1
        b1 = b0
        A = -(2*k + 1 + α - x)*f0_
        B = -f0*f1_
        b0 = a[k+1, α+k+1] + A*b1 + B*b2
        # println(b0)
        f1, f1_ = f0, f0_
        f0 = sqrt((k+α)*k)
        f0_ = 1/f0
    end
    B1 = -sqrt((α+1)/(2*(α+2)))
    return a[1, α+1] + ϕ1*b0 + B1*b1
end

function wigner4(rho, x, y)
    b = basis(rho)
    @assert isa(b, FockBasis)
    N = b.N::Int
    α = complex(x, y)/sqrt(2)
    abs2α = abs2(2*α)
    w = complex(0.)
    coefficient = complex(0.)
    @inbounds for L=N:-1:1
        # L = 2
        coefficient = 2*clenshaw_wigner3(N, L, abs2α, rho.data)
        w = coefficient + w*(2*α)/sqrt(L+1)
        # println("L: ", L, "  w: ", w)
    end
    coefficient = clenshaw_wigner3(N, 0, abs2α, rho.data)
    w = coefficient + w*(2*α)
    # println("w: ", w)
    exp(-2*abs2(α))/pi*real(w)
end


x = 1
y = 1
b = FockBasis(100)
# rho = DenseOperator(b)
# rho.data[2,4] = 1im
# rho.data[1,2] = 1
rho = dm(coherentstate(b, 2))

# @code_warntype wigner4(rho, x, y)
# @code_warntype clenshaw_wigner2(3, 2, 1., rho.data)

# w = wigner2(rho, x, y)
# println("wigner=", w)
w = wigner3(rho, x, y)
println("wigner=", w)
w = wigner4(rho, x, y)
println("wigner=", w)
# @time w = wigner2(rho, x, y)
# @time w = wigner2(rho, x, y)
@time w = wigner3(rho, x, y)
@time w = wigner3(rho, x, y)
@time w = wigner4(rho, x, y)
@time w = wigner4(rho, x, y)
# println("wigner=", w)

# function f1(N, rho, x, y)
#     for i=1:N
#         wigner4(rho, x, y)
#     end
# end

# @time f1(10000, rho, 1., 1.)
# @time f1(10000, rho, 1., 1.)

# Profile.clear()
# @profile f1(10000, rho, 1., 1.)
