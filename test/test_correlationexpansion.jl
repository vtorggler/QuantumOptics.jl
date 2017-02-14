using Base.Test
using QuantumOptics

srand(0)

b1 = FockBasis(2)
b2 = SpinBasis(1//2)
b3 = NLevelBasis(4)
b4 = NLevelBasis(2)
b = tensor(b1, b2, b3, b4)

randdo(b) = (x = normalize(Ket(b, rand(Complex128, length(b)))); x ⊗ dagger(x))
randop(b) = DenseOperator(b, rand(Complex128, length(b), length(b)))

function test_op_equal(op1, op2, eps=1e-10)
    @test_approx_eq_eps 0. tracedistance_general(full(op1), full(op2)) eps
end

# Test Masks
mask = correlationexpansion.indices2mask(3, [1,2])
@test mask == (true, true, false)
indices = correlationexpansion.mask2indices(mask)
@test indices == [1,2]

S2 = correlationexpansion.correlationmasks(4, 2)
S3 = correlationexpansion.correlationmasks(4, 3)
S4 = correlationexpansion.correlationmasks(4, 4)
@test S2 == Set([(true, true, false, false), (true, false, true, false), (true, false, false, true),
                 (false, true, true, false), (false, true, false, true), (false, false, true, true)])
@test S3 == Set([(false, true, true, true), (true, false, true, true), (true, true, false, true), (true, true, true, false)])
@test S4 == Set([(true, true, true, true)])

# Test creation of ApproximateOperator
op1 = DenseOperator(b, rand(Complex128, length(b), length(b)))
op1_ = correlationexpansion.approximate(op1, S2 ∪ S3 ∪ S4)
test_op_equal(op1, op1_)

# Test multiplication
h = 0.5*lazy(randop(b1)) ⊗ lazy(randop(b2)) ⊗ lazy(randop(b3)) ⊗ lazy(randop(b4))

test_op_equal(full(h)*0.3*full(op1_), h*(0.3*op1_))
test_op_equal(full(op1_)*0.3*full(h), (op1_*0.3)*h)
test_op_equal(full(h)*full(op1_)*0.3*full(h), h*(op1_*0.3)*h)

# Test ptrace
test_op_equal(ptrace(full(h)*op1, 1), ptrace(h*op1_, 1))
test_op_equal(ptrace(full(h)*op1, 2), ptrace(h*op1_, 2))
test_op_equal(ptrace(full(h)*op1, 3), ptrace(h*op1_, 3))
test_op_equal(ptrace(full(h)*op1, 4), ptrace(h*op1_, 4))
test_op_equal(ptrace(full(h)*op1, [1,2]), ptrace(h*op1_, [1,2]))
test_op_equal(ptrace(full(h)*op1, [1,3]), ptrace(h*op1_, [1,3]))
test_op_equal(ptrace(full(h)*op1, [1,4]), ptrace(h*op1_, [1,4]))
test_op_equal(ptrace(full(h)*op1, [2,3]), ptrace(h*op1_, [2,3]))
test_op_equal(ptrace(full(h)*op1, [2,4]), ptrace(h*op1_, [2,4]))
test_op_equal(ptrace(full(h)*op1, [3,4]), ptrace(h*op1_, [3,4]))
test_op_equal(ptrace(full(h)*op1, [1,2,3]), ptrace(h*op1_, [1,2,3]))
test_op_equal(ptrace(full(h)*op1, [1,2,4]), ptrace(h*op1_, [1,2,4]))
test_op_equal(ptrace(full(h)*op1, [1,3,4]), ptrace(h*op1_, [1,3,4]))
test_op_equal(ptrace(full(h)*op1, [2,3,4]), ptrace(h*op1_, [2,3,4]))

# Compare to standard master time evolution
rho = randdo(b1) ⊗ randdo(b2) ⊗ randdo(b3) ⊗ randdo(b4)
rho_ = correlationexpansion.approximate(rho, S2 ∪ S3 ∪ S4)

j1 = lazy(randop(b1)) ⊗ lazy(randop(b2)) ⊗ lazy(randop(b3)) ⊗ lazy(randop(b4))
j2 = lazy(randop(b1)) ⊗ lazy(randop(b2)) ⊗ lazy(randop(b3)) ⊗ lazy(randop(b4))
J = LazyTensor[j1, j2]
v = rand(Float64, length(J))
Γ = v * transpose(v)

H = LazySum(h, dagger(h))
T = [0.:0.001:0.01;]
tout_, rho_t_ = correlationexpansion.master(T, rho_, H, J; Gamma=Γ)
tout, rho_t = timeevolution.master_h(T, full(rho_), full(H), [full(j) for j in J]; Gamma=Γ)
for i=1:length(rho_t)
    test_op_equal(rho_t[i], rho_t_[i], 1e-5)
end
