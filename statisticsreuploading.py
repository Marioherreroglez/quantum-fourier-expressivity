import itertools
import warnings
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

# from statistics import SimulationStatisticsCoefficients, TheoryStatisticsCoefficients
import torch
from tqdm import tqdm

from redundancies import cumulative_redundancies
from reuploadingmodel import ReuploadingModel

# TODO: Abstract method for bounds?
# TODO: Separate upper bound with and without sq term


class SimulationStatisticsCoefficients:
    def __init__(self, model):
        self.circuit = model.circuit
        self.coefficients = model.coefficients
        self.average_coefficients = torch.mean(self.coefficients, dim=0)
        self.abs_coefficients = torch.abs(self.coefficients)
        self.average_abs_coefficients = torch.mean(self.abs_coefficients, dim=0)
        self.variance_coefficients = (
            torch.mean(self.abs_coefficients**2, dim=0)
            - torch.abs(torch.mean(self.coefficients, dim=0)) ** 2
        )


class TheoreticalStatisticsCoefficients:
    def __init__(
        self,
        model,
        bool_calculate_epsilon=False,
        show_epsilon_matrices=False,
        device_epsilon="cpu",
        epsilon_precalculated: Optional[Union[int, float]] = None,
    ):
        self.n_qubits = model.n_qubits
        self.circuit = model.circuit
        self.d = model.d
        self.freqs = model.freqs
        self.indices = self.get_indices_1layer()
        self.redundancies = model.redundancies
        self.ansatz = model.ansatz
        self._upper_var = None
        self._upper_var_square_cardinality = None
        self._upper_var_local = None
        self._upper_bound_multiple_circuit_layers_epsilon_approximate = None
        self.n_circuit_layers = model.n_circuit_layers
        self.ansatz = model.ansatz
        self.encoding = model.encoding
        self.n_periodic_layers = model.n_periodic_layers
        self.m_wires = model.m_wires
        self.sub_l = model.sub_l
        self.weights = model.weights
        self.cost = model.cost
        self.redundancies = model.redundancies
        self.layers_names = self._get_layers_names()
        if self.ansatz == "SimplifiedTwoDesign":
            self.init_weights = model.init_weights
        self.show_epsilon_matrices = show_epsilon_matrices
        self.device_epsilon = device_epsilon
        self.epsilon_precalculated = epsilon_precalculated
        self.d = 2**self.n_qubits
        self.n_encoding_qubits = model.n_encoding_qubits
        self.circuit = model.circuit
        self.circuit_no_encoding = self.create_filtered_circuit()
        if self.ansatz == "SimplifiedTwoDesign":
            self.Ws = qml.matrix(self.circuit_no_encoding)(
                None, self.weights, self.init_weights
            )

        else:
            self.Ws = torch.stack(
                qml.matrix(self.circuit_no_encoding)(None, self.weights)
            ).squeeze()
        self.bool_calculate_epsilon = bool_calculate_epsilon

        if self.bool_calculate_epsilon:
            self.bool_calculate_epsilon = (
                self._check_possibility_of_epsilon_calculation()
            )
        if self.epsilon_precalculated is not None:
            self.epsilon = self.epsilon_precalculated
        elif self.bool_calculate_epsilon:
            self.epsilon = self.get_epsilon()
        else:
            warnings.warn(
                "Epsilon not calculated. Using maximum value of 2^(2n)", RuntimeWarning
            )
            if self.ansatz != "BackwardsLightCone":
                self.epsilon = 2 ** (2 * self.n_qubits)
            else:
                self.epsilon = 2 ** (2 * self.n_encoding_qubits)
        self.__init__constants()
        self.var_coeffs_theory = self.variance_haar_random()

    def __init__constants(self):
        if self.cost == "global":
            self.norm_2_sq = 1
            self.Tr_O = 1
            self.sum_els_op2 = self.Tr_O**2
            self.scale_red = 1 / (self.d * (self.d + 1))
        elif self.cost == "local" or self.cost == "one_qubit":
            self.Tr_O = (
                2 ** (self.n_qubits - 1)
                + (self.n_qubits - 1) * 2 ** (self.n_qubits - 2)
            ) / self.n_qubits
            self.norm_2_sq = (
                self.d / 2 + (self.n_qubits - 1) * self.d / 4
            ) / self.n_qubits
            self.scale_red = (self.d * self.norm_2_sq - self.d**2 / 4) / (
                self.d * (self.d**2 - 1)
            )
            self.sum_els_op2 = self.Tr_O**2
        else:
            raise ValueError(
                "Theoretical value of variance not implemented for "
                + self.cost
                + " cost."
            )

    def _get_layers_names(self):
        return list(self.redundancies.keys())

    def _check_possibility_of_epsilon_calculation(self):
        if self.n_circuit_layers != 1:
            warnings.warn(
                "Epsilon calculation is only possible for 1 layer circuits.",
                RuntimeWarning,
            )
            return False
        if self.ansatz == "BackwardsLightCone":
            warnings.warn(
                "Epsilon calculation is not possible for BackwardsLightCone.",
                RuntimeWarning,
            )
            return False
        return True

    def get_epsilon(self):
        n_exp = self.Ws.clone().detach().shape[0]
        N = self.Ws.clone().detach().shape[1]
        outer1_einsum = torch.einsum("bij,bkl->bikjl", self.Ws, self.Ws).reshape(
            n_exp, N**2, N**2
        )
        outer2_einsum = torch.transpose(outer1_einsum, 1, 2).conj()

        BATCH_SIZE = 64
        num_batches = (N**2 + BATCH_SIZE - 1) // BATCH_SIZE
        distance = []
        with tqdm(
            range(num_batches),
            desc="Calculating Epsilon",
            leave=False,
            bar_format="{l_bar}{bar}|",
        ) as pbar:
            for batch_idx in pbar:
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, N**2)

                for j in range(start_idx, end_idx):
                    j1, j1p = j // N, j % N
                    jp = j1p * N + j1
                    if jp <= j:
                        matrix_ket = outer1_einsum[:, j]
                        bra_matrix = outer2_einsum[:, :, jp]
                        integral_theta = (
                            torch.einsum("bl,bm->lm", matrix_ket, bra_matrix) / n_exp
                        )
                        integral_haar = self._haar_matrix(j1, j1p, j1p, j1, N)

                        max_distance = torch.max(
                            torch.abs(integral_haar - integral_theta)
                        )
                        distance.append(max_distance.detach().cpu().numpy())
                        if self.show_epsilon_matrices:
                            self._plot_element_comparison(
                                j, j1, j1p, jp, j1p, j1, integral_haar, integral_theta
                            )
        distance = np.array(distance)
        return np.max(distance) * N**2

    def _haar_matrix(self, j1, i1p, j1p, i1, N):
        """
        computes the haar matrix corresponding to E[W|j1><j1'|W* otimes W|i1'><i1|W*]
        """

        def _delta(i, j):
            return i == j

        haar_matrix = torch.zeros((N, N, N, N), device=self.device_epsilon)
        for k1 in range(N):
            for l1 in range(N):
                for k2 in range(N):
                    for l2 in range(N):
                        haar_matrix[l1, l2, k1, k2] = (
                            _delta(l1, k1)
                            * _delta(l2, k2)
                            * _delta(j1, j1p)
                            * _delta(i1, i1p)
                            + _delta(l1, k2)
                            * _delta(k1, l2)
                            * _delta(j1, i1)
                            * _delta(i1p, j1p)
                        ) / (N**2 - 1) - (
                            _delta(l1, k1)
                            * _delta(l2, k2)
                            * _delta(j1, i1)
                            * _delta(i1p, j1p)
                            + _delta(l1, k2)
                            * _delta(k1, l2)
                            * _delta(j1, j1p)
                            * _delta(i1, i1p)
                        ) / (
                            N * (N**2 - 1)
                        )

        haar_matrix = haar_matrix.reshape((N**2, N**2))

        return haar_matrix

    def _plot_element_comparison(
        self,
        i,
        i1,
        i1p,
        j,
        j1,
        j1p,
        integral_haar,
        exp_term_k,
    ):
        plt.figure(figsize=(8, 2))

        plt.suptitle(
            f"Comparison of Components for i={i} ({i1}, {i1p}) and j={j} ({j1}, {j1p})",
            fontsize=14,
            y=1.1,
        )

        plt.subplot(1, 3, 1)
        plt.imshow(integral_haar, cmap="viridis")
        plt.title("Haar Integral", fontsize=12)
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(torch.real(exp_term_k), cmap="viridis")
        plt.colorbar()
        plt.title("Real Part", fontsize=12)

        plt.subplot(1, 3, 3)
        plt.imshow(torch.imag(exp_term_k), cmap="cividis")
        plt.title("Imaginary Part", fontsize=12)
        plt.colorbar()

        plt.subplots_adjust(
            left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3
        )

        plt.show()

    def create_filtered_circuit(self):
        non_encoding_ops = [op for op in self.circuit.qtape.operations if op.id is None]
        measurements = self.circuit.qtape.measurements

        def new_quantum_function(*qnode_args, **qnode_kwargs):
            # Reapply the non-encoding operations
            for op in non_encoding_ops:
                qml.apply(op)

            # Reapply the measurements
            return [qml.apply(m) for m in measurements]

        # Create a new QNode with the same device as the original one
        new_circuit = qml.QNode(new_quantum_function, self.circuit.device)
        return new_circuit

    def get_evals(self):
        """
        Computes eigenvalues of the global hamiltonian (one Pauli rotation on each qubit)
        """
        # Group eigenvalues by id of the encoding gate
        encoding_ops = [op for op in self.circuit.qtape.operations if op.id is not None]
        id_encoding_ops = [op.id for op in encoding_ops]
        evals = []
        for op in encoding_ops:
            if op.name == "DiagonalRotationUnitary":
                evals = op.D.tolist()
                return evals # This doesn't allow to combine with more than one layer!
            else:
                matrix = qml.matrix(qml.generator(op, format="observable"))
                evals.append(np.linalg.eigvalsh(matrix).tolist())
                if len(evals) > 1:
                    evals = list(itertools.product(*evals))
                    evals = [[combination[0] - combination[1] for combination in evals]]
        return evals[0]

    def get_indices_1layer(self):
        """
        Computes R(w) from the eigenvalues of the global hamiltonian
        """
        evals = self.get_evals()
        spectrum = []
        index_pairs = {}
        indexed_evals = list(enumerate(evals))
        for combination in itertools.product(indexed_evals, indexed_evals):
            # Extract indeces and values
            indeces = (combination[0][0], combination[1][0])
            values = (combination[0][1], combination[1][1])

            # Calculate the difference and append to the spectrum list
            difference = values[1] - values[0]
            spectrum.append(difference)

            # Keep track of the index pairs used for each difference
            if difference not in index_pairs:
                index_pairs[difference] = [indeces]
            else:
                index_pairs[difference].append(indeces)
        index_pairs = {k: v for k, v in index_pairs.items() if k >= 0}
        index_pairs = {key: index_pairs[key] for key in sorted(index_pairs.keys())}
        return index_pairs

    def variance_haar_random(self):
        # Theoretical var
        g = {}
        if self.ansatz == "BackwardsLightCone":
            m = 2
            L_1 = (self.n_qubits - self.n_encoding_qubits) / 2 + 1
            L_2 = self.n_encoding_qubits / 2
            for freq in self.redundancies[self.layers_names[-1]]:
                R_L = self.redundancies[self.layers_names[-1]][freq]
                g[freq] = (
                    (2 ** (m * L_2))
                    / ((2 ** (m * L_1) + 1) * (2 ** (2 * m * L_2) - 1))
                    * R_L
                    * (1 - 1 / (2**m))
                )
        elif True:  # Edinburgh homemade Fixing L layer
            partial_redundancies = [
                cumulative_redundancies(self.circuit, starting_layer=layer)(0.1)
                for layer in range(self.n_circuit_layers)
            ]
            partial_redundancies = [
                partial_redundancy[self.layers_names[-1]]
                for partial_redundancy in partial_redundancies
            ]
            for freq in self.redundancies[self.layers_names[-1]]:
                R_L = self.redundancies[self.layers_names[-1]][freq]
                # calculate numerator for R^L and R^L_1
                R_oneL = 0
                if self.n_circuit_layers > 1 and freq in partial_redundancies[1]:
                    R_oneL = partial_redundancies[1][freq]

                g[freq] = self.scale_red * (
                    (R_L - R_oneL)
                    / (
                        self.d
                        * (self.d + 1)
                        * (self.d**2 - 1) ** (self.n_circuit_layers - 1)
                    )
                )

                if self.n_circuit_layers > 2:
                    for layer in range(2, self.n_circuit_layers):
                        if freq in partial_redundancies[layer]:
                            print(
                                layer,
                                freq,
                                R_L,
                                R_oneL,
                                partial_redundancies[layer][freq],
                            )
                            g[freq] += self.scale_red * (
                                partial_redundancies[layer][freq]
                                / (
                                    self.d
                                    * (self.d**2 - 1)
                                    ** (self.n_circuit_layers - layer + 2)
                                )
                            )
        elif False:  # Previous LIP6 HOMEMADE Fixing first layer
            for freq in self.redundancies[self.layers_names[-1]]:
                R_L = self.redundancies[self.layers_names[-1]][freq]
                # calculate numerator for R^L and R^L_1
                R_oneL = 0
                if (
                    self.n_circuit_layers > 1
                    and freq in self.redundancies[self.layers_names[-2]]
                ):
                    R_oneL = self.redundancies[self.layers_names[-2]][freq]

                g[freq] = self.scale_red * (
                    (R_L - R_oneL)
                    / (
                        self.d
                        * (self.d + 1)
                        * (self.d**2 - 1) ** (self.n_circuit_layers - 1)
                    )
                )

                if self.n_circuit_layers > 2:
                    for layer in range(2, self.n_circuit_layers):
                        print(
                            layer,
                            freq,
                            R_L,
                            R_oneL,
                            self.redundancies[self.layers_names[-1 - layer]][freq],
                        )
                        if freq in self.redundancies[self.layers_names[-1 - layer]]:
                            g[freq] += self.scale_red * (
                                self.redundancies[self.layers_names[-1 - layer]][freq]
                                / (
                                    self.d
                                    * (self.d**2 - 1)
                                    ** (self.n_circuit_layers - layer + 2)
                                )
                            )
        else:
            for freq in self.redundancies[self.layers_names[-1]]:
                R_L = self.redundancies[self.layers_names[-1]][freq]
                g[freq] = self.scale_red * (
                    R_L
                    / (
                        self.d
                        * (self.d + 1)
                        * (self.d**2 - 1) ** (self.n_circuit_layers - 1)
                    )
                )
                # if self.n_circuit_layers >1:
                # g[freq] -=
                if self.n_circuit_layers > 1:
                    if freq in self.redundancies[self.layers_names[-2]]:
                        g[freq] -= self.scale_red * (
                            self.redundancies[self.layers_names[-2]][freq]
                            / (
                                self.d
                                * (self.d + 1)
                                * (self.d**2 - 1) ** (self.n_circuit_layers - 1)
                            )
                        )
        # g[0.0] -= scale_red/d #THERE IS A MISTAKE HERE
        var_theory = list(g.values())
        var_theory = var_theory[len(var_theory) // 2 :]
        return var_theory

    @property
    def upper_var_local(self):
        """
        Computes the upper bound for the variance without the quadratic term.
        """
        if self._upper_var_local is not None:
            return self._upper_var_local
        if (
            self._upper_var_local is None
            and self.n_circuit_layers == 1
            # and self.encoding == "pauli_encoding"
            and (self.ansatz == "BackwardsLightCone" or self.ansatz == "LocalTwoDesign")
        ):
            # Perform the computation
            self._upper_var_local = self._perform_upper_local_bound_var_epsilon_approx()
            return self._upper_var_local
        else:
            if self.ansatz != "BackwardsLightCone" and self.ansatz != "LocalTwoDesign":
                raise ValueError(
                    "Local Upper bound is not implemented for other ansatz than BackwardsLightCone or LocalTwoDesign"
                )

    @property
    def upper_var(self):
        """
        Computes the upper bound for the variance without the quadratic term.
        """
        if self._upper_var is not None:
            return self._upper_var
        elif (
            self._upper_var is None
            and self.ansatz != "BackwardsLightCone"
            and self.n_circuit_layers == 1
            # and self.encoding == "pauli_encoding"
        ):
            # Perform the computation
            self._upper_var = self._perform_upper_bound_var_epsilon_approx()
            return self._upper_var
        else:
            # raise error depending on the case
            if self.n_circuit_layers != 1:
                raise ValueError(
                    "Upper bound is not implemented for more than one circuit layer (L)"
                )
            elif self.encoding != "pauli_encoding":
                raise ValueError(
                    "Upper bound is not implemented for other exponential that is not normal Pauli encoding"
                )
            elif self.ansatz == "BackwardsLightCone":
                raise ValueError(
                    "Upper bound is not implemented for BackwardsLightCone"
                )

    @property
    def upper_var_square_cardinality(self):
        """
        Computes the upper bound for the variance with the quadratic term.
        """
        if self._upper_var_square_cardinality is not None:
            return self._upper_var_square_cardinality
        elif (
            self._upper_var_square_cardinality is None
            and self.ansatz != "BackwardsLightCone"
            and self.n_circuit_layers == 1
            # and self.encoding == "pauli_encoding"
        ):
            upper_var = self.upper_var
            # Perform the computation
            self._upper_var_square_cardinality = (
                self._perform_upper_bound_var_squared_cardinality_epsilon_approx()
            )
            return self._upper_var_square_cardinality
        else:
            # raise error depending on the case
            if self.n_circuit_layers != 1:
                raise ValueError(
                    "Upper bound is not implemented for more than one circuit layer (L)"
                )
            # elif self.encoding != "pauli_encoding":
            #    raise ValueError(
            #        "Upper bound is not implemented for other exponential that is not normal Pauli encoding"
            #    )
            elif self.ansatz == "BackwardsLightCone":
                raise ValueError(
                    "Upper bound is not implemented for BackwardsLightCone"
                )

    @property
    def upper_bound_multiple_circuit_layers_epsilon_approximate(self):
        if self._upper_bound_multiple_circuit_layers_epsilon_approximate is not None:
            return self._upper_bound_multiple_circuit_layers_epsilon_approximate
        elif self.ansatz != "BackwardsLightCone":
            self._upper_bound_multiple_circuit_layers_epsilon_approximate = (
                self._perform_upper_bound_multiple_circuit_layers_epsilon_approximate()
            )
            return self._upper_bound_multiple_circuit_layers_epsilon_approximate
        else:
            raise ValueError("Upper bound is not implemented for BackwardsLightCone")

    def _perform_upper_bound_var_epsilon_approx(self):
        self._upper_var = {}
        var_haar = {}
        C_1 = (self.d * self.norm_2_sq - self.Tr_O**2) / (self.d * (self.d**2 - 1))
        C_2 = self.sum_els_op2 / (self.d**2)
        K_1 = (self.d * self.Tr_O**2 - self.norm_2_sq) / (self.d * (self.d**2 - 1))
        K_2 = C_1

        for freq in self.indices.keys():
            var_haar[freq] = 0
            cardinality = self.redundancies[self.layers_names[-1]][freq]

            self._upper_var[freq] = cardinality * (
                C_1 * self.epsilon / self.d**2
                + C_2 * self.epsilon / (self.d * (self.d + 1))
            )

            for i_1, ip_1 in self.indices[freq]:
                for j_1, jp_1 in self.indices[freq]:
                    if i_1 == j_1 and ip_1 == jp_1:
                        var_haar[freq] += K_2
                    if i_1 == ip_1 and j_1 == jp_1:
                        var_haar[freq] += K_1
                        if i_1 == j_1:
                            var_haar[freq] += K_1 + K_2
            self._upper_var[freq] += var_haar[freq] / (self.d * (self.d + 1))
        return torch.tensor(list(self._upper_var.values()))

    def _perform_upper_bound_var_squared_cardinality_epsilon_approx(self):
        self._upper_var_square_cardinality = {}
        C_2 = self.sum_els_op2 / (self.d**2)
        for freq in self.indices.keys():
            cardinality = self.redundancies[self.layers_names[-1]][freq]
            self._upper_var_square_cardinality[freq] = (
                cardinality * C_2 * (self.epsilon**2 / self.d**2)
            )
        return self.upper_var + torch.tensor(
            list(self._upper_var_square_cardinality.values())
        )

    def _perform_upper_local_bound_var_epsilon_approx(self):
        # Only works for one layer as specified in the property
        if self.ansatz == "LocalTwoDesign":
            m_block_qubits = self.m_wires
            # TODO: calculate spectrum for lightcone depending on qubit measured
            # Is it necessary to refilter again the circuit?
            # There is no epsilon involved
        else:
            m_block_qubits = 2

        self._upper_var_local = {}
        for freq in self.indices.keys():
            cardinality = self.redundancies[self.layers_names[-1]][freq]
            # TODO: implement also the rank of the projector
            # non projector case
            # upper_var_local[freq] = (cardinality**2)*((2**(m_block_qubits+1))/(2**(2*m_block_qubits)-1))**(n_qubits/2+1)
            # projector case
            self._upper_var_local[freq] = (
                (cardinality**2)
                / (2 ** (2 * m_block_qubits))
                * ((2 ** (m_block_qubits + 1)) / (2 ** (2 * m_block_qubits) - 1))
                ** (self.n_encoding_qubits)
            )
        return torch.tensor(list(self._upper_var_local.values()))[
            : len(self.indices.keys())
        ]

    def _perform_upper_bound_multiple_circuit_layers_epsilon_approximate(self):
        self._upper_var_multiple_layers = {}
        for freq in self.indices.keys():
            cardinality = self.redundancies[self.layers_names[-1]][freq]
            self._upper_var_multiple_layers[freq] = (
                cardinality**2
                * (2 / self.d + self.epsilon) ** self.n_circuit_layers
                * (
                    (self.norm_2_sq + self.Tr_O**2) / (self.d * (self.d + 1))
                    + self.epsilon * self.Tr_O**2
                )
            )
        return torch.tensor(list(self._upper_var_multiple_layers.values()))
