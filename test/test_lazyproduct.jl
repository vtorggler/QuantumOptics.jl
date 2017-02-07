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

op1 = DenseOperator(b1, b2, rand(Complex128, length(b1), length(b2)))
op2 = DenseOperator(b2, b3, rand(Complex128, length(b2), length(b3)))
op3 = DenseOperator(b3, b3, rand(Complex128, length(b3), length(b3)))


# LazyProduct
op_l = (0.4*lazy(op1))*(-lazy(op2)*lazy(op3)/3)
op = -0.4*op1*op2*op3/3

test_op_equal(op, full(op_l))
test_op_equal(op, sparse(op_l))
