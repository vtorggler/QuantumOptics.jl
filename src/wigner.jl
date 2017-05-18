using QuantumOptics

typealias V Vector{Complex128}

function clenshaw_matrix(N::Int, L::Int, ρ::Matrix{Complex128},
                abs2_2α::Vector{Float64}, _2α::V,
                w::V, b0::V, b1::V, b2::V, scale::Int)
    n = N-L
    if n==0
        w .= scale*ρ[1, L+1] .+ w .* _2α ./ sqrt(L+1)
    elseif n==1
        # ϕ1 = -(L+1-x)/sqrt(L+1)
        w .= scale.*(ρ[1, L+1] .- ρ[2, L+2].*(L+1.-abs2_2α)./sqrt(L+1)) .+ w.*_2α./sqrt(L+1)
    else
        f0 = sqrt(float((n+L-1)*(n-1)))
        f1 = sqrt(float((n+L)*n))
        f0_ = 1/f0
        f1_ = 1/f1
        # b2 = complex(0.)
        b1 .= ρ[n+1, L+n+1]
        # A = -(2*n-1+L-x)/f1
        b0 .= ρ[n, L+n].-(2*n-1+L.-abs2_2α).*f1_.*b1
        @inbounds for k=n-2:-1:1
            b1, b2, b0 = b0, b1, b2
            # A = -(2*k + 1 + L - x)*f0_
            B = -f0*f1_
            b0 .= ρ[k+1, L+k+1] .-(2*k+1+L.-abs2_2α).*f0_.*b1 .+ B.*b2
            f1 = f0
            f1_ = f0_
            f0 = sqrt((k+L)*k)
            f0_ = 1/f0
        end
        B1 = -sqrt((L+1)/(2*(L+2)))
        w .= scale.*(ρ[1, L+1] -(L+1-abs2_2α)./sqrt(L+1).*b0 .+ B1.*b1) .+ w.*_2α./sqrt(L+1)
    end
end

function clenshaw_matrix2(N::Int, L::Int, ρ::Matrix{Complex128},
                abs2_2α::Vector{Float64}, _2α::V,
                w::V, b0::V, b1::V, b2::V, scale::Int)
    n = N-L
    points = length(w)
    if n==0
        f = scale*ρ[1, L+1]
        @inbounds for i=1:points
            w[i] = f + w[i]*_2α[i]/sqrt(L+1)
        end
    elseif n==1
        f1 = 1/sqrt(L+1)
        @inbounds for i=1:points
            w[i] = scale*(ρ[1, L+1] - ρ[2, L+2]*(L+1-abs2_2α[i])*f1) + w[i]*_2α[i]*f1
        end
    else
        f0 = sqrt(float((n+L-1)*(n-1)))
        f1 = sqrt(float((n+L)*n))
        f0_ = 1/f0
        f1_ = 1/f1
        fill!(b1, ρ[n+1, L+n+1])
        @inbounds for i=1:points
            b0[i] = ρ[n, L+n] - (2*n-1+L-abs2_2α[i])*f1_*b1[i]
        end
        @inbounds for k=n-2:-1:1
            b1, b2, b0 = b0, b1, b2
            @inbounds for i=1:points
                b0[i] = ρ[k+1, L+k+1] - (2*k+1+L-abs2_2α[i])*f0_*b1[i] - f0*f1_*b2[i]
            end
            f1 , f1_ = f0, f0_
            f0 = sqrt((k+L)*k)
            f0_ = 1/f0
        end
        @inbounds for i=1:points
            w[i] = scale*(ρ[1, L+1] - (L+1-abs2_2α[i])*f0_*b0[i] - f0*f1_*b1[i]) + w[i]*_2α[i]*f0_
        end
    end
end

function wigner_matrix(rho::DenseOperator, xvec::Vector{Float64}, yvec::Vector{Float64})
    b = basis(rho)
    @assert isa(b, FockBasis)
    N = b.N::Int
    points = length(xvec)*length(yvec)
    _2α = reshape(complex.(xvec, transpose(yvec))*sqrt(2), points)
    abs2_2α = abs2(_2α)
    w = zeros(Complex128, points)
    # println(abs2_2α)
    b0 = V(points)
    b1 = V(points)
    b2 = V(points)
    @inbounds for L=N:-1:1
        # L = 2
        clenshaw_matrix(N, L, rho.data, abs2_2α, _2α, w, b0, b1, b2, 2)
        # println("L: ", L, "  w: ", w)
    end
    clenshaw_matrix(N, 0, rho.data, abs2_2α, _2α, w, b0, b1, b2, 1)
    # println("w: ", w)
    exp(-abs2_2α./2)./pi.*real(w)
end

function wigner_matrix2(rho::DenseOperator, xvec::Vector{Float64}, yvec::Vector{Float64})
    b = basis(rho)
    @assert isa(b, FockBasis)
    N = b.N::Int
    points = length(xvec)*length(yvec)
    _2α = Vector{Complex128}(points)
    i = 0
    @inbounds for x=xvec
        for y=yvec
            i += 1
            _2α[i] = complex(x, y)*sqrt(2)
        end
    end

    # _2α = reshape(complex.(xvec, transpose(yvec))*sqrt(2), points)
    abs2_2α = abs2(_2α)
    w = zeros(Complex128, points)
    # # println(abs2_2α)
    b0 = V(points)
    b1 = V(points)
    b2 = V(points)
    @inbounds for L=N:-1:1
        # L = 2
        clenshaw_matrix2(N, L, rho.data, abs2_2α, _2α, w, b0, b1, b2, 2)
        # println("L: ", L, "  w: ", w)
    end
    clenshaw_matrix2(N, 0, rho.data, abs2_2α, _2α, w, b0, b1, b2, 1)
    # println("w: ", w)
    @inbounds for i=1:points
        abs2_2α[i] = exp(-abs2_2α[i]/2)/pi.*real(w[i])
    end
    reshape(abs2_2α, length(xvec), length(yvec))
end

x = ones(Float64, 2)
y = ones(Float64, 2)
b = FockBasis(100)
# rho = DenseOperator(b)
# rho.data[2,4] = 1im
# rho.data[1,2] = 1
rho = dm(coherentstate(b, 2))

w = wigner_matrix(rho, x, y)
println(w[1])
w = wigner_matrix2(rho, x, y)
println(w[1])

@time wigner_matrix(rho, x, y)
@time wigner_matrix(rho, x, y)

@time wigner_matrix2(rho, x, y)
@time wigner_matrix2(rho, x, y)

# using BenchmarkTools

# r = @benchmark wigner_matrix2($rho, $x, $y)
# println(r)

# function f1(N, rho, x, y)
#     for i=1:N
#         wigner_matrix2(rho, x, y)
#     end
# end

# Profile.clear()
# @profile f1(5, rho, x, y)