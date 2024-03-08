r"""
Contains the SimplifiedTwoDesign template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from ansatze.strongly_entangling import StronglyEntanglingLayers


class LocalTwoDesign(Operation):
    r"""
    Layers consisting of a local 2-design architecture of general rotations and controlled-NOT entanglers

    A 2-design is an ensemble of unitaries whose statistical properties are the same as sampling random unitaries
    with respect to the Haar measure up to the first 2 moments. For local 2-designs, the unitaries are
    products of local unitaries, where each local unitary acts on a small number of qubits. If there are sufficient 

    :math:`l` layers are applied, the basic building block of the layers are stongly entangling acting on :math:`m`qubits
    After  control-NOT rotation gates (one for each wire).
    Each layer consists of an "even" part whose entanglers start with the first qubit,
    and an "odd" part that starts with the second qubit.


    ``weights`` contains the group of rotation angles of the respective layers. Each layer takes
    3*N*d, where :math:`N` is the number of wires and `d` is the depth of the strongly entangling layer.
    The number of layers :math:`L` is derived from the first dimension of ``weights``.

    Args:
        weights (tensor_like): tensor of rotation angles for the layers, shape ``(l, M-1, 2)``
        wires (Iterable): wires that the template acts on



        **Parameter shapes**

        A list of shapes for the two weights arguments can be computed with the static method
        :meth:`~.qml.LocalTwoDesign.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shapes = qml.LocalTwoDesign.shape(n_periodic_layers=2, n_wires=2)
            weights = [np.random.random(size=shape) for shape in shapes]

    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self,weights, wires, m_subgroups,sub_l, id=None):
        shape = qml.math.shape(weights)[-5:]
        if len(wires) % m_subgroups != 0:
            raise ValueError(
                f"Number of wires {len(wires)} must be divisible by the depth of the strongly entangling layer {m_subgroups}"
            )
        if len(shape) > 1:
            if shape[1] != sub_l:
                raise ValueError(
                    f"Weights tensor must have second dimension (not counting batch dimension) of length sub_l={sub_l}; got {shape[1]}"
                )
            if shape[3] != m_subgroups:
                raise ValueError(
                    f"Weights tensor must have third dimension (not counting batch dimension) of length m_subgroups={m_subgroups}; got {shape[3]}"
                )
            if shape[2]*shape[3] != len(wires):
                raise ValueError(
                    f"""Weights tensor must have third dimension (n_subgroups) times fourth dimension (m_subgroups) (not counting batch dimension)
                    of length wires={len(wires)}; got {shape[2]}x{shape[3]}= {shape[2]*shape[3]}"""
                )

            if shape[4] != 3:
                raise ValueError(
                    f"Weights tensor must have fourth dimension (not counting batch dimension) of length 3; got {shape[4]}"
                )

        self.n_periodic_layers = shape[0]

        super().__init__(weights, wires=wires,id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
         weights, wires,
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            weights (tensor_like): tensor of rotation angles for the layers
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> todo: add example
        """

        n_periodic_layers = qml.math.shape(weights)[-5]
        m_subgroups = qml.math.shape(weights)[-2]
        m_wires = qml.math.shape(weights)[-3]
        op_list = []
        for layer in range(n_periodic_layers):
            for subgroup in range(m_subgroups):
                op_list.append(StronglyEntanglingLayers(weights[...,layer,:,:,subgroup,:],wires=wires[subgroup*m_wires:(subgroup+1)*m_wires]))
            
        return op_list

    @staticmethod
    def shape(n_periodic_layers,m_wires,m_subgroups,sub_l):
        r"""Returns a list of shapes for the 2 parameter tensors.

        Args:
            n_periodic_layers (int): number of layers
            m_wires (int): proportionality factor between m_subgroups and n_wires
            m (int): size of the local subgroup. will be useful in the future
            sub_l (int): number of layers in the subgroup
            
        Returns:
            list[tuple[int]]: list of shapes
        """

        return n_periodic_layers, sub_l,m_wires,m_subgroups, 3
