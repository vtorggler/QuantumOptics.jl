.. _section-operators:

Operators
=========

Operators can be defined as linear mappings from one Hilbert space to another. However, equivalently to states, operators in **Quantumoptics.jl** are interpreted as coefficients of an abstract operator in respect to one (or more generally two, possibly distinct) :ref:`basis <section-bases>`. For a certain choice of bases :math:`\{|u_i\rangle\}_i` and :math:`\{|v_j\rangle\}_j` an abstract operator :math:`A` has the coefficients :math:`A_{ij}` which are connected by the relation

.. math::

    A =  \sum_{ij} A_{ij} | u_i \rangle \langle v_j |

For performance reasons there are several different implementation of operators in **Quantumoptics.jl**:

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

and also :jl:func:`dagger(::DenseOperator)`, :jl:func:`normalize(::DenseOperator)` and :jl:func:`normalize!(::DenseOperator)` are available.

* Expectation values can be calculated with:

.. epigraph::

    .. jl:autofunction:: operators.jl expect

* Composite operators can be created with:

.. epigraph::

    .. jl:autofunction:: operators.jl tensor(::DenseOperator, ::DenseOperator)

    .. jl:autofunction:: operators.jl tensor(::Ket, ::Bra)

* The inverse operation - taking a partial trace is done with:

.. epigraph::

    .. jl:autofunction:: operators.jl ptrace(::DenseOperator, indices)

* For creating operators of the type :math:`A = I \otimes I \otimes ... a_i ... \otimes I` the very useful embed function can be used:

.. epigraph::

    .. jl:autofunction:: operators.jl embed


.. _subsection-denseoperators:

Dense operators
^^^^^^^^^^^^^^^


.. _subsection-sparseoperators:

Sparse operators
^^^^^^^^^^^^^^^^


.. _subsection-lazyoperators:

Lazy operators
^^^^^^^^^^^^^^
