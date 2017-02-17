using Base.Test
using QuantumOptics

srand(0)

randdo(b) = (x = normalize(Ket(b, rand(Complex128, length(b)))); x ⊗ dagger(x))
randop(b) = DenseOperator(b, rand(Complex128, length(b), length(b)))

function test_op_equal(op1, op2, eps=1e-10)
    @test_approx_eq_eps 0. tracedistance_general(full(op1), full(op2)) eps
end

b1 = FockBasis(2)
b2 = SpinBasis(1//2)
b3 = NLevelBasis(5)
b4 = NLevelBasis(7)
b = tensor(b1, b2, b3, b4)

S2 = correlationexpansion.correlationmasks(4, 2)
S3 = correlationexpansion.correlationmasks(4, 3)
S4 = correlationexpansion.correlationmasks(4, 4)

rho = randdo(b1) ⊗ randdo(b2) ⊗ randdo(b3) ⊗ randdo(b4)

# rho_ = correlationexpansion.approximate(rho, S2 ∪ S3 ∪ S4)
# rho_2 = correlationexpansion2.approximate(rho, S2 ∪ S3 ∪ S4)
# rho_3 = correlationexpansion3.approximate(rho, S2 ∪ S3 ∪ S4)

# rho_ = correlationexpansion.approximate(rho, S2 ∪ S3)
# rho_2 = correlationexpansion2.approximate(rho, S2 ∪ S3)
# rho_3 = correlationexpansion3.approximate(rho, S2 ∪ S3)

rho_ = correlationexpansion.approximate(rho, S2)
rho_2 = correlationexpansion2.approximate(rho, S2)
rho_3 = correlationexpansion3.approximate(rho, S2)

# rho_ = correlationexpansion.approximate(rho)
# rho_2 = correlationexpansion2.approximate(rho)
# rho_3 = correlationexpansion3.approximate(rho)


# Example 1:
j1 = LazyTensor(b, [1,2], [randop(b1), randop(b2)])
j2 = LazyTensor(b, [1,3], [randop(b1), randop(b3)])
j3 = LazyTensor(b, [1,4], [randop(b1), randop(b4)])
j4 = LazyTensor(b, [2,3], [randop(b2), randop(b3)])
j5 = LazyTensor(b, [2,4], [randop(b2), randop(b4)])
j6 = LazyTensor(b, [3,4], [randop(b3), randop(b4)])

# J = LazyTensor[j1, j2, j3, j4, j5, j6]
J = LazyTensor[]
v = rand(Float64, length(J))
Γ = v * transpose(v)

H = LazyTensor[]
for i=1:4
    for j=i+1:4
        h = LazyTensor(b, [i, j], [randop(b.bases[i]), randop(b.bases[j])])
        push!(H, h)
        push!(H, dagger(h))
    end
end
H = LazySum(H...)


T = [0.:0.001:0.01;]
println("Start time evolution:")
tout_, rho_t_ = correlationexpansion.master(T, rho_, H, J; Gamma=Γ)
tout2_, rho_t2_ = correlationexpansion2.master(T, rho_2, H, J; Gamma=Γ)
tout3_, rho_t3_ = correlationexpansion3.master(T, rho_3, H, J; Gamma=Γ)


@time tout_, rho_t_ = correlationexpansion.master(T, rho_, H, J; Gamma=Γ)
@time tout2_, rho_t2_ = correlationexpansion2.master(T, rho_2, H, J; Gamma=Γ)
@time tout3_, rho_t3_ = correlationexpansion3.master(T, rho_3, H, J; Gamma=Γ)

Profile.clear()
Profile.init(10^8, 0.001)

@profile tout_, rho_t_ = correlationexpansion.master(T, rho_, H, J; Gamma=Γ)
rho_full = full(rho)
H_full = full(H)
J_full = [full(j) for j in J]
tout, rho_t = timeevolution.master_h(T, rho_full, H_full, J_full; Gamma=Γ)
@time tout, rho_t = timeevolution.master_h(T, rho_full, H_full, J_full; Gamma=Γ)

for i=1:length(rho_t)
    println("Check: ", i)
    test_op_equal(rho_t3_[i], rho_t_[i], 1e-5)
    test_op_equal(rho_t3_[i], rho_t2_[i], 1e-5)
end
