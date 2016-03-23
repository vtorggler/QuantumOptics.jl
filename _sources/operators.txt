.. _section-operators:

Operators
=========

Operators can be defined as linear mappings from one Hilbert space to another. However, equivalently to states, operators in **Quantumoptics.jl** are interpreted as coefficients of an abstract operator in respect to one (or more generally two, possibly distinct) :ref:`basis <section-bases>`. For a certain choice of bases :math:`\{|u_i\rangle\}_i` and :math:`\{|v_j\rangle\}_j` an abstract operator :math:`A` has the coefficients :math:`A_{ij}` which are connected by the relation

.. math::

    A =  \sum_{ij} A_{ij} | u_i \rangle \langle v_j |

For performance reasons there are several different implementation of operators in **Quantumoptics.jl** all inheriting from the abstract :jl:abstract:`Operator` type:

* :ref:`subsection-denseoperators`
* :ref:`subsection-sparseoperators`
* :ref:`subsection-lazyoperators`

All of them have the same interface and can in most cases be used interchangeably. E.g. all of them implement the expected arithmetic functions
\*, /, +, -::

    b = SpinBasis(1//2)
    sx = sigmax(b)
    sy = sigmay(b)
    sx + sy
    sx * sy

and additionally implement the following functions:

* Hermitian conjugate:
    :jl:func:`dagger(::DenseOperator)`

* Normalization:
    :jl:func:`trace(::DenseOperator)`
    :jl:func:`norm(::DenseOperator, )`
    :jl:func:`normalize(::DenseOperator)`
    :jl:func:`normalize!(::DenseOperator)`

* Expectation values:
    :jl:func:`expect(::Operator, ::DenseOperator)`


* Tensor product:
    :jl:func:`tensor(::DenseOperator, ::DenseOperator)`
    :jl:func:`tensor(::Ket, ::Bra)`
    :jl:func:`projector(::Ket, ::Bra)`

* Partial trace:
    :jl:func:`ptrace(::DenseOperator, indices)`
    :jl:func:`ptrace(::DenseOperator, index)`
    :jl:func:`ptrace(::Ket, indices)`
    :jl:func:`ptrace(::Bra, indices)`

For creating operators of the type :math:`A = I \otimes I \otimes ... a_i ... \otimes I` the very useful embed function can be used:

* :jl:func:`embed(basis, indices, operators)`
* :jl:func:`embed(basis, index, op)`


.. _subsection-denseoperators:

Dense operators
^^^^^^^^^^^^^^^

:jl:type:`DenseOperator` is implemented as::

    type DenseOperator <: Operator
        basis_l::Basis
        basis_r::Basis
        data::Matrix{Complex128}
    end

The data is stored as complex (dense) matrix in the *data* field.

The :jl:func:`full(Operator)` can be used to convert other types of operators to dense operators.


.. _subsection-sparseoperators:

Sparse operators
^^^^^^^^^^^^^^^^

:jl:type:`SparseOperator` is the default type used in **Quantumoptics.jl**. The reason is that in many quantum systems the Hamiltonians in respect to the commonly used bases are sparse. They are implemented as::

    type SparseOperator <: Operator
        basis_l::Basis
        basis_r::Basis
        data::SparseMatrixCSC{Complex128}
    end

To convert other operators to sparse operators the :jl:func:`sparse(::Operator)` can be used.


.. _subsection-lazyoperators:

Lazy operators
^^^^^^^^^^^^^^

Lazy operators allow delayed evaluation of certain operations. This is for useful when combining two operators is numerically expensive but separate application on states is cheap. This is for example the case for the :jl:type:`FFTOperator` operator implemented for particles. It allows using FFT transformation to convert a state from real space to momentum space applying a diagonal operator and converting it back. This is only fast if the the order of operations is :math:`\mathrm{IFFT}*(D*(\mathrm{FFT}*\psi))`. To create a Hamiltonian that uses this calculation order, lazy evaluation is needed::

    xmin = -5
    xmax = 5
    Npoints = 100
    b_position = PositionBasis(xmin, xmax, Npoints)
    b_momentum = MomentumBasis(b_position)

    p = momentumoperator(b_momentum)
    x = positionoperator(b_position)

    fft = particle.FFTOperator(b_momentum, b_position);

    H = LazyProduct(dagger(fft), p^2/2, fft) + x^2

In this case the Hamiltonian :math:`H` is a lazy sum of the kinetic term :math:`p^2/2` and the potential term :math:`x^2` where the kinetic term is the lazy product mentioned before. In the end this results in a speed up from :math:`O(N^2)` to :math:`O(N \log N)`.

All lazy operators inherit from the abstract :jl:abstract:`LazyOperator`. There are currently three different concrete implementations:

* :jl:type:`LazyTensor`
* :jl:type:`LazySum`
* :jl:type:`LazyProduct`
