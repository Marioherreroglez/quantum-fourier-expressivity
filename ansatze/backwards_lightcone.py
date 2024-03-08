r"""
Contains the BackwardsLightcone template.
"""

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from .strongly_entangling import StronglyEntanglingLayers


class BackwardsLightCone(Operation):
    r"""
    Left and right layers for the backwards lightcone architecture consisting of one unique encoding layer
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, encoding_wires, ansatz_section, id=None):
        if ansatz_section not in ["preEncoding", "postEncoding"]:
            raise ValueError(
                "Invalid ansatz section. Must be 'preEncoding' or 'postEncoding'."
            )
        # calculate the number of layers for the left and right part
        n_left_layers = int(1 + (len(wires) - encoding_wires) / 2)
        n_right_layers = int(encoding_wires / 2)
        if ansatz_section == "preEncoding":
            n_layers = n_left_layers
            if len(weights) != n_layers:
                raise ValueError(
                    f"Expected the number of layers of weights to be {n_left_layers} for the pre-encoding ansatz; got {len(weights)}"
                )
        else:
            n_layers = n_right_layers
            if len(weights) != n_right_layers:
                raise ValueError(
                    f"Expected the number of layers of weights to be {n_right_layers} for the post-encoding ansatz; got {len(weights)}"
                )

        # Check shapes for every element of the list
        shapes_list = [qml.math.shape(weight)[-3:] for weight in weights]
        expected_wires_per_layer_preEnc = [len(wires) - 2 * i for i in range(n_layers)]
        expected_wires_per_layer_postEnc = [
            encoding_wires - 2 * i for i in range(n_layers)
        ]
        if ansatz_section == "preEncoding":
            expected_wires_per_layer = expected_wires_per_layer_preEnc
        else:
            expected_wires_per_layer = expected_wires_per_layer_postEnc
        for i, shape in enumerate(shapes_list):
            if shape[1] != expected_wires_per_layer[i]:
                raise ValueError(
                    f"Weights tensor for layer {i+1} must have second dimension (not counting batch dimension) of length {expected_wires_per_layer[i]}; got {shape[1]}"
                )
            if shape[2] != 3:
                raise ValueError(
                    f"Weights tensor for layer {i+1} must have third dimension (not counting batch dimension) of length 3; got {shape[2]}"
                )

        self.n_sublayers_list = [shape[-3] for shape in shapes_list]
        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights,
        wires,
    ):
        r"""Representation of the operator as a product of other operators.

        Args:
            weights (tensor_like): weights tensor
            wires (Wires): wires that the operator acts on
        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []
        m_wires = 2
        if weights[0].shape[-2] != len(wires):
            initial_wire = (len(wires) - weights[0].shape[-2]) // 2
        else:
            initial_wire = 0
        for layer in range(len(weights)):
            wires_per_layer = weights[layer].shape[-2]
            for weight_idx, qubit_pair in enumerate(
                range(initial_wire, wires_per_layer + initial_wire, m_wires)
            ):
                op_list.append(
                    StronglyEntanglingLayers(
                        weights[layer][..., weight_idx : weight_idx + m_wires, :],
                        wires=wires[qubit_pair : qubit_pair + m_wires],
                    )
                )
            initial_wire += 1
        return op_list

    @staticmethod
    def shape(n_sublayers, wires, n_encoding_wires):
        r"""
        Returns a list with the weights for each layer
        """
        # Check if wires is even
        if wires % 2 != 0:
            raise ValueError(f"Number of wires {wires} must be even")
        # Check if encoding wires is even
        if n_encoding_wires % 2 != 0:
            raise ValueError(
                f"Number of encoding wires {n_encoding_wires} must be even"
            )
        # Check if encoding wires is smaller than wires
        if n_encoding_wires > wires:
            raise ValueError(
                f"Number of encoding wires {n_encoding_wires} must be smaller than number of wires {wires}"
            )

        step = -2
        left_wires_per_layer = range(wires, n_encoding_wires + step, step)
        right_wires_per_layer = range(n_encoding_wires, 2 + step, step)

        return [
            (n_sublayers, wires_layer, 3) for wires_layer in left_wires_per_layer
        ], [(n_sublayers, wires_layer, 3) for wires_layer in right_wires_per_layer]
