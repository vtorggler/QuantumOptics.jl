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

op1 = DenseOperator(b, b, rand(Complex128, length(b), length(b)))
op2 = DenseOperator(b, b, rand(Complex128, length(b), length(b)))
op3 = DenseOperator(b, b, rand(Complex128, length(b), length(b)))


# Arithmetic
op_l = (-0.2*lazy(op1)) - 0.5*lazy(op2) + lazy(sparse(op3))/3
op = -0.2*op1 - 0.5 * op2 + op3/3

test_op_equal(op, full(op_l))
test_op_equal(op, sparse(op_l))

op_l = (2*(lazy(sparse(op1)) - lazy(op2)) + lazy(op3))/3
op = (2*(op1 - op2) + op3)/3

test_op_equal(op, full(op_l))
test_op_equal(op, sparse(op_l))
