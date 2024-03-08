"""
Description: This module provides implementation for the Data Reuploading model using PennyLane.
Author: Mario Herrero-Gonzalez
Created: 2024-01-30
"""

from math import pi
from typing import Optional, Union

import numpy as np
import pennylane as qml
import torch
from tqdm import tqdm

from ansatze import (
    BackwardsLightCone,
    LocalTwoDesign,
    OnlyRotationLayers,
    SimplifiedTwoDesign,
    StronglyEntanglingLayers,
)
from encoding_gates import RX, DiagonalRotationUnitary
from redundancies import cumulative_redundancies


class ReuploadingModel:
    def __init__(
        self,
        dev: str,
        encoding: str,
        diff_generator_per_layer: bool,
        ansatz: str,
        cost: str,
        n_qubits: int,
        n_samples: int,
        n_circuit_layers: int = 1,
        n_periodic_layers: Optional[int] = None,
        n_encoding_qubits: Optional[int] = None,
        sub_l: Optional[int] = None,
        m_wires: Optional[int] = None,
        m_subgroups: Optional[int] = None,
        scaling: Optional[float] = 1,
        qubit_measured: Optional[int] = None,
    ):
        # Initialize your model here
        self.encoding = encoding
        self.diff_generator_per_layer = diff_generator_per_layer
        self.ansatz = ansatz
        self.cost = cost
        self.n_qubits = n_qubits
        self.d = 2**n_qubits
        self.n_samples = n_samples
        self.n_circuit_layers = n_circuit_layers
        self.n_periodic_layers = n_periodic_layers
        self.n_encoding_qubits = n_encoding_qubits
        if ansatz == "BackwardsLightCone":
            assert self.n_encoding_qubits is not None
            self.d = 2 ** int(self.n_encoding_qubits)
        self.sub_l = sub_l
        self.m_wires = m_wires
        self.m_subgroups = m_subgroups
        self.scaling = scaling
        self.qubit_measured = qubit_measured
        self.dev = qml.device(dev, wires=self.n_qubits)
        layers_dict = {
            "BasicEntangling": qml.BasicEntanglerLayers,
            "SimplifiedTwoDesign": SimplifiedTwoDesign,
            "StronglyEntangling": StronglyEntanglingLayers,
            "OnlyRotations": OnlyRotationLayers,
            "LocalTwoDesign": LocalTwoDesign,
            "BackwardsLightCone": BackwardsLightCone,
        }
        self.layers = layers_dict[self.ansatz]
        self.weights = self.get_weights()
        if ansatz == "SimplifiedTwoDesign":
            self.init_weights, self.weights = self.weights
        self.circuit = qml.QNode(self._circuit, self.dev)
        self.freqs, self.redundancies = self.get_redundancies()
        self.max_freq = int(self.freqs[-1])
        self.steps = 2 * self.max_freq + 1
        self.sampling_datapoints = torch.linspace(
            0, 2 * torch.pi * (self.steps - 1) / self.steps, steps=self.steps
        )
        self.n_coeffs = len(self.freqs)
        self._coefficients = None

    def _circuit(self, input):
        if self.ansatz == "BackwardsLightCone":
            self.layers(
                self.weights[0],
                wires=range(self.n_qubits),
                encoding_wires=self.n_encoding_qubits,
                ansatz_section="preEncoding",
            )
            assert self.n_encoding_qubits is not None
            starting_wire = (self.n_qubits - self.n_encoding_qubits) // 2
            idx_encoding_wires = list(
                range(starting_wire, starting_wire + self.n_encoding_qubits)
            )
            if self.encoding == "pauli_encoding":
                for idx in idx_encoding_wires:
                    RX(self.scaling * input, wires=idx, id="s_" + str(0))
            elif self.encoding == "enhanced_pauli_encoding":
                for idx in idx_encoding_wires:
                    prefactor = 3**idx
                    RX(
                        input * prefactor,
                        wires=idx,
                        id="s_" + str(0),
                        prefactor=prefactor,
                    )
            elif self.encoding == "golomb_encoding":
                U = self._get_golomb_ruler()
                DiagonalRotationUnitary(
                    input, U, wires=idx_encoding_wires, id="s_" + str(0)
                )
        else:
            for i in range(self.n_circuit_layers):
                if self.ansatz == "SimplifiedTwoDesign":
                    assert self.init_weights is not None
                    self.layers(
                        self.init_weights[i],
                        self.weights[i],
                        wires=range(self.n_qubits),
                    )
                elif self.ansatz == "LocalTwoDesign":
                    self.layers(
                        self.weights[i],
                        wires=range(self.n_qubits),
                        m_subgroups=self.m_subgroups,
                        sub_l=self.sub_l,
                    )
                else:
                    self.layers(self.weights[i], wires=range(self.n_qubits))
                if self.encoding == "pauli_encoding":
                    for j in range(self.n_qubits):
                        RX(input, wires=j, id="s_" + str(i))
                elif self.encoding == "enhanced_pauli_encoding":
                    if self.diff_generator_per_layer:
                        for j in range(self.n_qubits):
                            prefactor = 3 ** (i * self.n_qubits + j)
                            RX(
                                input * prefactor,
                                wires=j,
                                id="s_" + str(i),
                                prefactor=prefactor,
                            )
                    else:
                        for j in range(self.n_qubits):
                            prefactor = 3**j
                            RX(
                                input * prefactor,
                                wires=j,
                                id="s_" + str(i),
                                prefactor=prefactor,
                            )
                elif self.encoding == "golomb_encoding":
                    U = self._get_golomb_ruler()
                    DiagonalRotationUnitary(
                        input, U, wires=range(self.n_qubits), id="s_" + str(i)
                    )
                else:
                    raise ValueError(
                        "Encoding must be either enhanced_pauli_encoding, golomb_encoding or pauli_encoding"
                    )

        if self.ansatz == "SimplifiedTwoDesign":
            assert self.init_weights is not None
            self.layers(
                self.init_weights[-1], self.weights[-1], wires=range(self.n_qubits)
            )
        elif self.ansatz == "LocalTwoDesign":
            self.layers(
                self.weights[-1],
                wires=range(self.n_qubits),
                m_subgroups=self.m_subgroups,
                sub_l=self.sub_l,
            )
        elif self.ansatz == "BackwardsLightCone":
            if len(self.weights[1]) == 1:
                weights_pos = torch.stack(list(self.weights[1]))
            else:
                weights_pos = self.weights[1]
            self.layers(
                weights_pos,
                wires=range(self.n_qubits),
                encoding_wires=self.n_encoding_qubits,
                ansatz_section="postEncoding",
            )
        else:
            self.layers(self.weights[-1], wires=range(self.n_qubits))
        if self.ansatz == "BackwardsLightCone":
            middle_qubits = [self.n_qubits // 2 - 1, self.n_qubits // 2]
            if self.cost == "one_qubit":
                return qml.expval(qml.PauliZ(wires=self.qubit_measured))
            if self.cost == "global":
                return qml.probs(wires=middle_qubits)  # good global
            if self.cost == "local":
                return [qml.probs(wires=i) for i in middle_qubits]
        else:
            if self.cost == "one_qubit":
                return qml.expval(qml.PauliZ(wires=self.qubit_measured))
            if self.cost == "global":
                return qml.probs(wires=range(self.n_qubits))  # good global
            if self.cost == "local":
                return [qml.probs(wires=i) for i in range(self.n_qubits)]
            if self.cost == "global_test":
                global_projector = qml.Projector(
                    np.zeros(2**self.n_qubits), wires=range(self.n_qubits)
                )
                return qml.expval(global_projector)

    def get_weights_shape(self) -> Union[list, tuple]:
        """
        Obtains the shapes of the weights for a given ansatz and number of qubits.

        Args:
            ansatz (str): The ansatz used in the quantum circuit ('BasicEntangling', 'SimplifiedTwoDesign', 'StronglyEntangling',
            'Only_Rotation', 'BackwardsLightCone', 'Local_Two_Design')
            n_qubits (int): Number of qubits in the circuit.
            n_circuit_layers (int): Number of circuit layers in the data reuploading circuit
            n_periodic_layers (int): Number of periodic layers within each circuit layer
            n_samples (int): Number of samples to draw the statistics from.
            n_encoding_qubits (int, optional): When using 'BackwardsLightCone', the number of encoding qubits. Defaults to None.
            sub_l (int, optional): When using 'BackwardsLightCone' or 'Local_Two_Design', the number of sublayers within a subgroup. Defaults to None.
            m_wires (int, optional): When using 'Local_Two_Design', the number of wires in each subgroup. Defaults to None.
            m_subgroups (int, optional): When using 'Local_Two_Design', the number of subgroups. Defaults to None.

        Returns:
            Union[list,tuple]: The shapes of the weights for the given ansatz and number of qubits. list in the case of 'BackwardsLightCone' or
                            'SimplifiedTwoDesign' and tuple otherwise.
        """

        if self.ansatz == "LocalTwoDesign":
            shapes = self.layers.shape(
                m_wires=self.m_wires,
                m_subgroups=self.m_subgroups,
                n_periodic_layers=self.n_periodic_layers,
                sub_l=self.sub_l,
            )
            weights_shape = (
                self.n_circuit_layers + 1,
                self.n_samples,
            ) + shapes
        elif self.ansatz == "BackwardsLightCone":
            shape_preEnc, shape_postEnc = self.layers.shape(
                n_sublayers=self.sub_l,
                wires=self.n_qubits,
                n_encoding_wires=self.n_encoding_qubits,
            )
            for i in range(len(shape_preEnc)):
                shape_preEnc[i] = (self.n_samples,) + shape_preEnc[i]
            for i in range(len(shape_postEnc)):
                shape_postEnc[i] = (self.n_samples,) + shape_postEnc[i]
            weights_shape = [shape_preEnc, shape_postEnc]
        else:
            shapes = self.layers.shape(self.n_periodic_layers, self.n_qubits)
            if self.ansatz == "SimplifiedTwoDesign":
                init_shapes, shapes = shapes
                weights_shape = [
                    (self.n_circuit_layers + 1, self.n_samples)
                    + init_shapes,  # + (self.n_samples,),
                    (self.n_circuit_layers + 1, self.n_samples) + shapes,
                ]
            else:
                weights_shape = (
                    self.n_circuit_layers + 1,
                    self.n_samples,
                ) + shapes
        return weights_shape

    def get_weights(self):
        # TODO: Add None as initialization option
        weights_shape = self.get_weights_shape()
        if self.ansatz == "SimplifiedTwoDesign":
            init_weights_shape, weights_shape = weights_shape
            init_weights = 2 * pi * torch.randn(init_weights_shape)
            weights = 2 * pi * torch.randn(weights_shape)
            return init_weights, weights
        if self.ansatz == "BackwardsLightCone":
            weights = [
                [2 * pi * torch.randn(w_shape) for w_shape in weights_shape[0]],
                [2 * pi * torch.randn(w_shape) for w_shape in weights_shape[1]],
            ]
        else:
            weights = 2 * pi * torch.randn(weights_shape)
        return weights

    def get_redundancies(self):
        redundancies = cumulative_redundancies(self.circuit)(0.1)
        freqs = list(redundancies[list(redundancies.keys())[-1]].keys())
        freqs = freqs[len(freqs) // 2 :]
        return freqs, redundancies

    def measure_circuit(self, input):
        if self.cost == "local":
            mean_probs = []
            outcome = torch.stack(list(self.circuit(input)))
            mean_prob = torch.sum(outcome[:, :, 0], dim=0) / self.n_qubits
            mean_probs.append(mean_prob)
            return torch.stack(mean_probs)

        elif self.cost == "one_qubit" and self.ansatz == "BackwardsLightcone":
            middle_qubits = [self.n_qubits // 2 - 1, self.n_qubits // 2]
            if self.qubit_measured not in middle_qubits:
                raise ValueError(
                    "The measured qubit must be in the middle qubits, i.e. qubit_measured = {}".format(
                        middle_qubits
                    )
                )
            return torch.stack([self.circuit(x_) for x_ in input])
        else:
            return self.circuit(input)

    def circuit_diagram(self):
        print(
            qml.draw(
                self.circuit,
                expansion_strategy="device",
                show_matrices=False,
                max_length=250,
            )(0.1)
        )

    @property
    def coefficients(self):
        """
        Computes the Fourier coefficients if it hasn't been done before.
        """
        if self._coefficients is None:
            # Perform the computation
            self._coefficients = self._perform_fourier_coefficients()

        return self._coefficients

    def perform_measurement(self):
        # Generate measurements and ensure each is a tensor
        t = self.sampling_datapoints
        measure = []
        with tqdm(
            t, desc="Calculating Coefficients", leave=False, bar_format="{l_bar}{bar}|"
        ) as pbar:
            for _t in pbar:
                result = self.measure_circuit(_t)
                if not isinstance(result, torch.Tensor):
                    raise ValueError("measure_circuit must return a torch.Tensor")
                measure.append(result)
        measure = torch.stack(measure)
        measure = measure.squeeze()
        assert measure is not None
        if self.cost == "global":
            measure = measure[:, :, 0]
        return measure

    def _perform_fourier_coefficients(self):
        measure = self.perform_measurement()
        y = torch.fft.rfft(measure.T) / self.steps
        return y.squeeze()

    def _get_golomb_ruler(self):
        """
        Returns an optimal Golomb ruler for a given order `d`. An optimal Golomb ruler
        has the smallest maximal difference between its elements, with each element
        difference being unique. For `d > 4`, perfect Golomb rulers, with consecutive
        integer spacing, do not exist.

        Parameters:
        - self.d: Order of the Golomb ruler.

        Returns:
        - np.array: Optimal Golomb ruler.
        """
        if self.d == 4:
            # return torch.diag(torch.tensor([0, 1, 4, 6]))
            return torch.tensor([0, 1, 4, 6])
        elif self.d == 8:
            # return torch.diag(torch.tensor([0, 1, 4, 9, 15, 22, 32, 34]))
            return torch.tensor([0, 1, 4, 9, 15, 22, 32, 34])
        elif self.d == 16:
            # return torch.diag(torch.tensor([0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150, 163, 168, 177]))
            return torch.tensor(
                [0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150, 163, 168, 177]
            )
        else:
            raise NotImplementedError("Golomb ruler not implemented for n > 4")
