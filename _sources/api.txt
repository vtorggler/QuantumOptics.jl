.. section-api::

API
===

Bases
-----

.. jl:autoabstract:: bases.jl Basis

.. jl:autotype:: bases.jl GenericBasis

.. jl:autotype:: bases.jl CompositeBasis

.. jl:autofunction:: bases.jl tensor

.. jl:autotype:: subspace.jl SubspaceBasis

.. jl:autofunction:: subspace.jl orthonormalize

.. jl:autofunction:: subspace.jl projector

Systems
-------

Fock
^^^^

.. jl:autotype:: fock.jl FockBasis

.. jl:autofunction:: fock.jl number

.. jl:autofunction:: fock.jl destroy

.. jl:autofunction:: fock.jl create

.. jl:autofunction:: fock.jl fockstate

.. jl:autofunction:: fock.jl coherentstate


Spin
^^^^

.. jl:autotype:: spin.jl SpinBasis

.. jl:autofunction:: spin.jl sigmax

.. jl:autofunction:: spin.jl sigmay

.. jl:autofunction:: spin.jl sigmaz

.. jl:autofunction:: spin.jl sigmap

.. jl:autofunction:: spin.jl sigmam

.. jl:autofunction:: spin.jl spinup

.. jl:autofunction:: spin.jl spindown


Particle
^^^^^^^^

.. jl:autotype:: particle.jl PositionBasis

.. jl:autotype:: particle.jl MomentumBasis

.. jl:autofunction:: particle.jl positionoperator

.. jl:autofunction:: particle.jl momentumoperator

.. jl:autofunction:: particle.jl laplace_x

.. jl:autofunction:: particle.jl laplace_p

.. jl:autotype:: particle.jl FFTOperator


N-Particles
^^^^^^^^^^^

.. jl:autoabstract:: nparticles.jl NParticleBasis

.. jl:autotype:: nparticles.jl BosonicNParticleBasis

.. jl:autotype:: nparticles.jl FermionicNParticleBasis

.. jl:autofunction:: nparticles.jl nparticleoperator_1

.. jl:autofunction:: nparticles.jl nparticleoperator_2


Superoperators
--------------