import os
import pickle
from typing import Optional, Union

import numpy as np


class ResultsSaver:
    def __init__(self, model, simulation_stats, theoretical_stats):
        self.variables_to_save = self.get_variables_to_save(
            model, simulation_stats, theoretical_stats
        )
        self.model = model
        self.path = self.build_logs_path()

    def build_logs_path(self) -> str:
        """
        Constructs a file path string for logging and data storage depending on the parameters and conditions of the
        quantum circuit.

        This function dynamically builds a file path based on various parameters related to the quantum circuit,
        including its encoding, ansatz, cost function, and other specifics such as the number of qubits,
        the number of encoding qubits, and the number of periodic layers. It supports additional parameters for
        ansatz types like 'BackwardsLightCone' and 'LocalTwoDesign' through keyword arguments.

        Args:
        - encoding (str): The type of encoding used in the quantum circuit ('pauli_encoding' 'enhanced_pauli_encoding').
        - diff_generator_per_layer (bool): Whether or not to use a different generator for each layer in the circuit.
        - ansatz (str): The ansatz used in the quantum circuit ('BasicEntangling', 'SimplifiedTwoDesign', 'StronglyEntangling',
        'Only_Rotation', 'BackwardsLightCone', 'Local_Two_Design')
        - cost (str): The type of cost function used ('global', 'local', 'one_qubit').

        Returns:
        - str: A string representing the constructed file path based on the provided parameters and conditions.

        Example:
        file_path = build_file_path("pauli_encoding", False, 5, "BackwardsLightCone", "one_qubit")
        """
        base_path = f"logs/{self.model.encoding}/"
        if (
            self.model.n_circuit_layers > 1
            and self.model.diff_generator_per_layer
            and self.model.encoding == "enhanced_pauli_encoding"
        ):
            base_path += "diff_gen/"
        if self.model.ansatz == "BackwardsLightCone":
            ansatz_details = f"{self.model.ansatz}_{self.model.cost}_qubits_{self.model.n_qubits:02}_encoding_qubits_{self.model.n_encoding_qubits:02}_subperiodic_layers_{self.model.sub_l:02}"
        else:
            ansatz_details = f"{self.model.ansatz}_{self.model.cost}_qubits_{self.model.n_qubits:02}_circ_layers_{self.model.n_circuit_layers:02}_periodic_layers_{self.model.n_periodic_layers:02}"
        if self.model.cost == "one_qubit":
            ansatz_details = ansatz_details.replace(
                f"{self.model.cost}",
                f"{self.model.cost}_qubit_measured_{self.model.qubit_measured:02}",
            )
        if self.model.ansatz == "LocalTwoDesign":
            ansatz_details = ansatz_details.replace(
                f"{self.model.n_qubits:02}",
                f"{self.model.n_qubits:02}_m_{self.model.m_wires:02}_subgroups_{self.model.m_subgroups:02}",
            )
            ansatz_details = (
                ansatz_details + f"_subperiodic_layers_{self.model.sub_l:02}"
            )
        return base_path + ansatz_details + f"/{self.model.n_samples:05}_samples/"

    def get_variables_to_save(self, model, simulation_stats, theoretical_stats):
        _variables_to_save = {
            "freqs.npy": model.freqs,
            "coeffs.npy": model.coefficients.numpy(),
            "redundancies.pkl": model.redundancies,
            "abs_coeffs_avg.npy": simulation_stats.average_abs_coefficients,
            "abs_coeffs_var.npy": simulation_stats.variance_coefficients.numpy(),
            "abs_coeffs_theory.npy": theoretical_stats.var_coeffs_theory,
        }

        # Add epsilon if it was possible to calculate it,
        # add epsilon_max if it was not possible to calculate epsilon
        if theoretical_stats.bool_calculate_epsilon:
            _variables_to_save["epsilon.npy"] = theoretical_stats.epsilon
        else:
            _variables_to_save["epsilon_max.npy"] = theoretical_stats.epsilon

        # Add the bounds depending on ansatz
        if model.ansatz == "BackwardsLightCone":
            _variables_to_save["abs_coeffs_upper_bound_local.npy"] = (
                theoretical_stats.upper_var_local
            )
        elif model.n_circuit_layers == 1 :
            if model.ansatz == "LocalTwoDesign":
                _variables_to_save["abs_coeffs_upper_bound_local.npy"] = (
                    theoretical_stats.upper_var_local
                )
            _variables_to_save["abs_coeffs_upper_bound.npy"] = (
                theoretical_stats.upper_var
            )
            _variables_to_save["abs_coeffs_upper_bound_squared.npy"] = (
                theoretical_stats.upper_var_square_cardinality
            ) 
        # If any of the values already exists in the path, remove it from the dictionary
        # for key, value in _variables_to_save.copy().items():
        #    if os.path.exists(os.path.join(self.path, key)):
        #        del _variables_to_save[key]

        return _variables_to_save

    def save_variables(self):
        # Create the directory if it does not exist
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for key, value in self.variables_to_save.items():
            if key[-3:] == "npy":
                np.save(os.path.join(self.path, key), value)
            elif key[-3:] == "pkl":
                with open(os.path.join(self.path, key), "wb") as f:
                    pickle.dump(value, f)
