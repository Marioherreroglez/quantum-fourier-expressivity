import functools

import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation

# from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ


def _can_replace(x, y):
    """
    Convenience function that returns true if x is close to y and if
    x does not require grad
    """
    return (
        not qml.math.is_abstract(x)
        and not qml.math.requires_grad(x)
        and qml.math.allclose(x, y)
    )


stack_last = functools.partial(qml.math.stack, axis=-1)


class RX(Operation):
    r"""
    The single qubit X rotation

    .. math:: R_x(\phi) = e^{-i\phi\sigma_x/2} = \begin{bmatrix}
                \cos(\phi/2) & -i\sin(\phi/2) \\
                -i\sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_x(\phi)) = \frac{1}{2}\left[f(R_x(\phi+\pi/2)) - f(R_x(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_x(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "X"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def generator(self):
        return -0.5 * self.prefactor * qml.PauliX(wires=self.wires)

    def __init__(self, phi, wires, id=None, prefactor=1):
        super().__init__(phi, wires=wires, id=id)
        self.prefactor = prefactor

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.RX.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.RX.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689+0.0000j, 0.0000-0.2474j],
                [0.0000-0.2474j, 0.9689+0.0000j]])
        """
        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)

        if qml.math.get_interface(theta) == "tensorflow":
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)

        # The following avoids casting an imaginary quantity to reals when backpropagating
        c = (1 + 0j) * c
        js = -1j * s
        return qml.math.stack([stack_last([c, js]), stack_last([js, c])], axis=-2)

    def adjoint(self):
        return RX(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [RX(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.CRX(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return RX(theta, wires=self.wires)

    def single_qubit_rot_angles(self):
        # RX(\theta) = RZ(-\pi/2) RY(\theta) RZ(\pi/2)
        pi_half = qml.math.ones_like(self.data[0]) * (np.pi / 2)
        return [pi_half, self.data[0], -pi_half]


class DiagonalRotationUnitary(Operation):
    r"""DiagonalRotationUnitary(D, wires)
    Apply the rotation of an arbitrary diagonal unitary matrix with a dimension that is a power of two.

    .. math:: R_U(\phi) = e^{-i\phi U}

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (1,)
    * Gradient recipe: None

    Args:
        phi (float): rotation angle :math:`\phi`
        D (array[complex]): diagonal of unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str or None): String representing the operation (optional)
    """

    num_wires = AnyWires
    num_params = 1
    ndim_params = (0,)

    grad_method = None

    def __init__(self, phi, D, wires, id=None, prefactor=1):
        super().__init__(phi, wires=wires, id=id)
        self.prefactor = prefactor
        self.D = D

    # def generator(self):
    #    return  qml. (D,wires=self.wires)

    #@classmethod
    def compute_matrix(self,phi):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Args:
            phi (tensor_like or float): rotation angle
            D (tensor_like): diagonal of unitary matrix

        Returns:
            tensor_like: canonical matrix
        """
        return qml.math.diag(qml.math.exp(-1j * phi * self.D))

    # def adjoint(self):
    #    return DiagonalRotationUnitary(
    #        qml.math.conj(self.parameters[0]), wires=self.wires
    #    )

    # def pow(self, z):
    #    cast_data = qml.math.cast(self.data[0], np.complex128)
    #    return [DiagonalRotationUnitary(cast_data**z, wires=self.wires)]

    # def _controlled(self, control):
    #    return DiagonalRotationUnitary(
    #        qml.math.hstack([np.ones_like(self.parameters[0]), self.parameters[0]]),
    #        wires=control + self.wires,
    #    )

    # def simplify(self):
    #    theta = self.data[0] % (4 * np.pi)

    #   if _can_replace(theta, 0):
    #        return qml.Identity(wires=self.wires)

    #   return DiagonalRotationUnitary(theta, wires=self.wires)
