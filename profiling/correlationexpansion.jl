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
b3 = NLevelBasis(3)
b4 = NLevelBasis(2)
b = tensor(b1, b2, b3, b4)

S2 = correlationexpansion.correlationmasks(4, 2)
S3 = correlationexpansion.correlationmasks(4, 3)
S4 = correlationexpansion.correlationmasks(4, 4)

rho = randdo(b1) ⊗ randdo(b2) ⊗ randdo(b3) ⊗ randdo(b4)
rho_ = correlationexpansion.approximate(rho, S2 ∪ S3 ∪ S4)

# Example 1:
j1 = LazyTensor(b, [1,2], [randop(b1), randop(b2)])
j2 = LazyTensor(b, [1,3], [randop(b1), randop(b3)])
j3 = LazyTensor(b, [1,4], [randop(b1), randop(b4)])
j4 = LazyTensor(b, [2,3], [randop(b2), randop(b3)])
j5 = LazyTensor(b, [2,4], [randop(b2), randop(b4)])
j6 = LazyTensor(b, [3,4], [randop(b3), randop(b4)])

J = LazyTensor[j1, j2, j3, j4, j5, j6]
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


T = [0.:0.000001:0.00001;]
println("Start time evolution:")
tout_, rho_t_ = correlationexpansion.master(T, rho_, H, J; Gamma=Γ)

Profile.clear()
Profile.init(10^8, 0.001)

@profile tout_, rho_t_ = correlationexpansion.master(T, rho_, H, J; Gamma=Γ)
tout, rho_t = timeevolution.master_h(T, full(rho_), full(H), [full(j) for j in J]; Gamma=Γ)
for i=1:length(rho_t)
    test_op_equal(rho_t[i], rho_t_[i], 1e-5)
end
