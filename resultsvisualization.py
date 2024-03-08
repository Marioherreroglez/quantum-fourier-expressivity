from typing import Optional, Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.markers import MarkerStyle


class ResultsVisualization:
    def __init__(self, model, simulation_stats, theoretical_stats):
        self.model = model
        self.simulation_stats = simulation_stats
        self.theoretical_stats = theoretical_stats

    def plot_pannel_samples_coeffs(self):
        coeffs_real = np.real(self.model.coefficients)
        coeffs_imag = np.imag(self.model.coefficients)
        bone_cmap = cm.get_cmap("bone_r")

        # Define the start point (0 is the start, 1 is the end of the colormap)
        start = 0.3  # Adjust this value to control the starting shade

        # Create a new colormap starting from 'start' point of the original 'bone' colormap
        darker_bone_cmap = mcolors.LinearSegmentedColormap.from_list(
            "truncated_bone", bone_cmap(np.linspace(start, 1, 256))
        )

        fig, ax = plt.subplots(1, self.model.n_coeffs, figsize=(16, 2))
        for idx, ax_ in enumerate(ax):
            max_abs_val = (
                torch.max(self.simulation_stats.abs_coefficients[:, idx]) * 1.1
            )  # Adding a 10% margin
            ax_.set_title(r"c_{}".format(idx))

            # Creating a hexbin plot for density
            hb = ax_.hexbin(
                coeffs_real[:, idx],
                coeffs_imag[:, idx],
                gridsize=50,
                cmap=darker_bone_cmap,
                mincnt=1,
            )

            # Extract bin counts and locations
            counts = hb.get_array()
            verts = hb.get_offsets()
            x, y = verts[:, 0], verts[:, 1]
            # Adding a color bar per subplot
            cb = plt.colorbar(hb, ax=ax_)
            cb.set_label("counts")

            # ax_.set_xlim(-1,1)
            # ax_.set_ylim(-1,1)
            ax_.set_xlim(-max_abs_val, max_abs_val)
            ax_.set_ylim(-max_abs_val, max_abs_val)
            avg_real = torch.real(self.simulation_stats.average_coefficients)[
                idx
            ].item()
            avg_imag = torch.imag(self.simulation_stats.average_coefficients)[
                idx
            ].item()
            ax_.scatter(avg_real, avg_imag, color="red", s=3)
            # If max abs value is 0, set the limits to (-1, 1)
            # take into account that 0 could be 1e-16 or something like that
            # plot grid
            ax_.grid()
            if max_abs_val < 5e-16:
                ax_.set_xlim(-1, 1)
                ax_.set_ylim(-1, 1)
            plt.tight_layout()

    def plot_abs_coeffs_bounds(self):
        plt.figure()

        if self.model.ansatz == "LocalTwoDesign":
            if self.model.cost == "one_qubit":
                plt.title(
                    self.model.encoding
                    + " "
                    + self.model.ansatz
                    + " epsilon "
                    + str(np.round(self.theoretical_stats.epsilon, 4))
                    + "\n"
                    + "N Qubits "
                    + str(self.model.n_qubits)
                    + " N Circuit Layers: "
                    + str(self.model.n_circuit_layers)
                    + " N Ansatz Layers "
                    + str(self.model.n_periodic_layers)
                    + " N Wires: "
                    + str(self.model.m_wires)
                    + " N Subgroups: "
                    + str(self.model.m_subgroups)
                    + "\n"
                    + " N Subperiodic Layers: "
                    + str(self.model.sub_l)
                    + " Cost: "
                    + self.model.cost
                    + " Q Measured:"
                    + str(self.model.qubit_measured)
                    + " N samples: "
                    + str(self.model.n_samples),
                    fontsize=10,
                )
            else:
                plt.title(
                    self.model.encoding
                    + " "
                    + self.model.ansatz
                    + " epsilon "
                    + str(np.round(self.theoretical_stats.epsilon, 4))
                    + "\n"
                    + "N Qubits "
                    + str(self.model.n_qubits)
                    + " N Circuit Layers: "
                    + str(self.model.n_circuit_layers)
                    + " N Ansatz Layers "
                    + str(self.model.n_periodic_layers)
                    + " N Wires: "
                    + str(self.model.m_wires)
                    + " N Subgroups: "
                    + str(self.model.m_subgroups)
                    + "\n"
                    + " N Subperiodic Layers: "
                    + str(self.model.sub_l)
                    + " Cost: "
                    + self.model.cost
                    + " N samples: "
                    + str(self.model.n_samples),
                    fontsize=10,
                )
        elif self.model.ansatz == "BackwardsLightCone":
            if self.model.cost == "one_qubit":
                plt.title(
                    self.model.encoding
                    + " "
                    + self.model.ansatz
                    + " epsilon "
                    + str(np.round(self.theoretical_stats.epsilon, 4))
                    + "\n"
                    + "N Qubits "
                    + str(self.model.n_qubits)
                    + " N Encoding Qubits: "
                    + str(self.model.n_encoding_qubits)
                    + "\n"
                    + " N Subperiodic Layers: "
                    + str(self.model.sub_l)
                    + " Cost: "
                    + self.model.cost
                    + " Q Measured:"
                    + str(self.model.qubit_measured)
                    + " N samples: "
                    + str(self.model.n_samples),
                    fontsize=10,
                )
            else:
                plt.title(
                    self.model.encoding
                    + " "
                    + self.model.ansatz
                    + " epsilon "
                    + str(np.round(self.theoretical_stats.epsilon, 4))
                    + "\n"
                    + "N Qubits "
                    + str(self.model.n_qubits)
                    + " N Encoding Qubits: "
                    + str(self.model.n_encoding_qubits)
                    + "\n"
                    + " N Subperiodic Layers: "
                    + str(self.model.sub_l)
                    + " Cost: "
                    + self.model.cost
                    + " N samples: "
                    + str(self.model.n_samples),
                    fontsize=10,
                )

        else:
            if self.model.cost == "one_qubit":
                plt.title(
                    self.model.encoding
                    + " "
                    + self.model.ansatz
                    + " epsilon "
                    + str(np.round(self.theoretical_stats.epsilon, 4))
                    + "\n"
                    + "N Qubits "
                    + str(self.model.n_qubits)
                    + " N Circuit Layers: "
                    + str(self.model.n_circuit_layers)
                    + " N Ansatz Layers "
                    + str(self.model.n_periodic_layers)
                    + " Cost: "
                    + self.model.cost
                    + " Q Measured:"
                    + str(self.model.qubit_measured)
                    + " N samples: "
                    + str(self.model.n_samples),
                    fontsize=10,
                )
            else:
                plt.title(
                    self.model.encoding
                    + " "
                    + self.model.ansatz
                    + " epsilon "
                    + str(np.round(self.theoretical_stats.epsilon, 4))
                    + "\n"
                    + "N Qubits "
                    + str(self.model.n_qubits)
                    + " N Circuit Layers: "
                    + str(self.model.n_circuit_layers)
                    + " N Ansatz Layers "
                    + str(self.model.n_periodic_layers)
                    + " Cost: "
                    + self.model.cost
                    + " N samples: "
                    + str(self.model.n_samples),
                    fontsize=10,
                )
        plt.scatter(
            self.model.freqs, torch.abs(self.simulation_stats.average_coefficients)
        )

        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.grid()

    def plot_var_abs_coeffs_bounds(
        self,
        var_log_scale=False,
        plot_square_term=True,
        plot_upper_bound=True,
        plot_upper_local_bound=True,
        plot_simulation_variance=True,
        plot_haar_random_variance=True,
    ):
        plt.figure()

        if self.model.cost == "one_qubit":
            plt.title(
                self.model.encoding
                + " "
                + self.model.ansatz
                + " epsilon "
                + str(np.round(self.theoretical_stats.epsilon, 4))
                + "\n"
                + "N Qubits "
                + str(self.model.n_qubits)
                + " N Circuit Layers: "
                + str(self.model.n_circuit_layers)
                + " N Ansatz Layers "
                + str(self.model.n_periodic_layers)
                + " N Wires: "
                + str(self.model.m_wires)
                + " N Subgroups: "
                + str(self.model.m_subgroups)
                + "\n"
                + " N Subperiodic Layers: "
                + str(self.model.sub_l)
                + " Cost: "
                + self.model.cost
                + " Q Measured:"
                + str(self.model.qubit_measured)
                + " N samples: "
                + str(self.model.n_samples),
                fontsize=10,
            )
        else:
            plt.title(
                self.model.encoding
                + " "
                + self.model.ansatz
                + " epsilon "
                + str(np.round(self.theoretical_stats.epsilon, 4))
                + "\n"
                + "N Qubits "
                + str(self.model.n_qubits)
                + " N Circuit Layers: "
                + str(self.model.n_circuit_layers)
                + " N Ansatz Layers "
                + str(self.model.n_periodic_layers)
                + " N Wires: "
                + str(self.model.m_wires)
                + " N Subgroups: "
                + str(self.model.m_subgroups)
                + "\n"
                + " N Subperiodic Layers: "
                + str(self.model.sub_l)
                + " Cost: "
                + self.model.cost
                + " N samples: "
                + str(self.model.n_samples),
                fontsize=10,
            )

        # Plotting the data in the desired order for the legend
        if (
            self.model.ansatz == "LocalTwoDesign"
            or self.model.ansatz == "BackwardsLightCone"
        ):
            if plot_upper_local_bound:
                plt.plot(
                    self.model.freqs,
                    self.theoretical_stats.upper_var_local,
                    marker=MarkerStyle("o"),
                    color="#91322f",
                    label="Upper Local Bound",
                    alpha=0.8,
                )
        if self.model.n_circuit_layers == 1 and self.model.ansatz != "BackwardsLightCone":
            if plot_square_term:
                plt.plot(
                    self.model.freqs,
                    self.theoretical_stats.upper_var_square_cardinality,
                    color="#91322f",
                    label="Upper bound w sq term",
                    alpha=0.7,
                )

            if plot_upper_bound:
                plt.plot(
                    self.model.freqs,
                    self.theoretical_stats.upper_var,
                    marker=MarkerStyle("x"),
                    color="#91322f",
                    label="Upper Bound",
                    alpha=0.8,
                )
        if plot_simulation_variance:
            plt.plot(
                self.model.freqs,
                self.simulation_stats.variance_coefficients[self.model.freqs],
                marker=MarkerStyle("o"),
                color="#ed802d",
                label="Sampling",
                alpha=0.7,
                zorder=2,
            )
        if plot_haar_random_variance:
            plt.plot(
                self.model.freqs,
                self.theoretical_stats.var_coeffs_theory,
                marker=MarkerStyle("s"),
                color="#91322f",
                label="Theory 2-design",
                zorder=1,
            )
        # if self.model.ansatz != "BackwardsLightCone":
        #    plt.plot(
        #        self.model.freqs,
        #        self.theoretical_stats.upper_bound_multiple_circuit_layers_epsilon_approximate,
        #        marker=MarkerStyle("x"),
        #        color="blue",
        #        label="Upper Bound Multiple C Layers",
        #        alpha=0.8,
        #    )
        if var_log_scale:
            plt.yscale("log")

        plt.xlabel("Frequency")
        plt.ylabel("Variance")
        plt.legend()
        plt.grid()
