using Base.Test
using QuantumOptics

srand(0)

b1 = NLevelBasis(4)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

b = b1⊗b2⊗b3


function test_op_equal(op1, op2)
    @test_approx_eq_eps 0. tracedistance_general(full(op1), full(op2)) 1e-11
end

op1a = DenseOperator(b1, b1, rand(Complex128, length(b1), length(b1)))
op1b = DenseOperator(b1, b1, rand(Complex128, length(b1), length(b1)))
op2a = DenseOperator(b2, b2, rand(Complex128, length(b2), length(b2)))
op2b = DenseOperator(b2, b2, rand(Complex128, length(b2), length(b2)))
op3a = DenseOperator(b3, b3, rand(Complex128, length(b3), length(b3)))
op3b = DenseOperator(b3, b3, rand(Complex128, length(b3), length(b3)))
I1 = full(identityoperator(b1))
I2 = full(identityoperator(b2))
I3 = full(identityoperator(b3))

op = LazyTensor(b, Dict(1=>op1a, 2=>op2a, 3=>op3a))
test_op_equal(op1a⊗op2a⊗op3a, full(op))
test_op_equal(op1a⊗op2a⊗op3a, sparse(op))

op = LazyTensor(b, Dict(1=>sparse(op1a), 2=>op2a, 3=>sparse(op3a)))
test_op_equal(op1a⊗op2a⊗op3a, full(op))
test_op_equal(op1a⊗op2a⊗op3a, sparse(op))

op = LazyTensor(b, Dict(1=>sparse(op1a), 3=>op3a))
test_op_equal(op1a⊗I2⊗op3a, full(op))
test_op_equal(op1a⊗I2⊗op3a, sparse(op))

op = LazyTensor(b, Dict([1,3]=>sparse(op1a⊗op3a)))
test_op_equal(op1a⊗I2⊗op3a, full(op))
test_op_equal(op1a⊗I2⊗op3a, sparse(op))

op = LazyTensor(b, Dict(1=>op1a))
test_op_equal(op1a⊗I2⊗I3, full(op))
test_op_equal(op1a⊗I2⊗I3, sparse(op))

op = LazyTensor(b, Dict(2=>op2a))
test_op_equal(I1⊗op2a⊗I3, full(op))
test_op_equal(I1⊗op2a⊗I3, sparse(op))

op = LazyTensor(b, Dict(3=>op3a))
test_op_equal(I1⊗I2⊗op3a, full(op))
test_op_equal(I1⊗I2⊗op3a, sparse(op))