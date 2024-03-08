from collections import Counter
from functools import wraps
from itertools import combinations, product

import numpy as np
import pennylane as qml


def get_spectrum_redundancy(op, decimals):
    r"""Extract the frequencies contributed by an input-encoding gate to the
    overall Fourier representation of a quantum circuit.

    If :math:`G` is the generator of the input-encoding gate :math:`\exp(-i x G)`,
    the frequencies are the differences between any two of :math:`G`'s eigenvalues.
    We only compute non-negative frequencies in this subroutine.

    Args:
        op (~pennylane.operation.Operation): Operation to extract
            the frequencies for
        decimals (int): Number of decimal places to round the frequencies to

    Returns:
        set[float]: non-negative frequencies contributed by this input-encoding gate
    """
    if op.name == "DiagonalRotationUnitary":
        evals = op.D
    else:
        matrix = qml.matrix(qml.generator(op, format="observable"))
        # todo: use qml.math.linalg once it is tested properly
        evals = np.linalg.eigvalsh(matrix)
    zeros = np.zeros_like(evals)
    # compute all unique positive differences of eigenvalues, then add 0
    # note that evals are sorted already
    _spectrum = np.round(
        [x[1] - x[0] for x in combinations(evals, 2)], decimals=decimals
    )
    _spectrum_neg = -_spectrum
    _spectrum = np.append(zeros, _spectrum)
    _spectrum = np.append(_spectrum_neg, _spectrum)

    return _spectrum


def join_spectra_redundancy(spec1, spec2):
    if not isinstance(spec1, np.ndarray):
        spec1 = np.array([key for key, value in spec1.items() for _ in range(value)])
    if not isinstance(spec2, np.ndarray):
        spec2 = np.array([key for key, value in spec2.items() for _ in range(value)])

    r"""Join two sets of frequencies that belong to the same input.

    Since :math:`\exp(i a x)\exp(i b x) = \exp(i (a+b) x)`, the spectra of two gates
    encoding the same :math:`x` are joined by computing the set of sums and absolute
    values of differences of their elements.
    We only compute non-negative frequencies in this subroutine and assume the inputs
    to be non-negative frequencies as well.

        spec2 (set[float]): second spectrum
    Returns:
        set[float]: joined spectrum
    """
    # We assume that every qubit has an encoding gate per encoding layer
    # Otherwise the redundancies in this code will be lower
    # With identity each frequency will have additional redundancy of *2^(# non-affected qubits)
    if (spec1 == np.array([0])).all():
        return spec2
    if (spec2 == np.array([0])).all():
        return spec1

    sums = []
    for s1 in spec1:
        for s2 in spec2:
            sums.append(s1 + s2)

    sums = np.array(sums)
    redundancy_dict = Counter(sums)
    redundancy_dict = {k: v for k, v in sorted(redundancy_dict.items())}

    return sums, redundancy_dict


def cumulative_redundancies(qnode, encoding_gates=None, decimals=8, starting_layer=0):
    @wraps(qnode)
    def wrapper(*args, **kwargs):
        qnode.construct(args, kwargs)
        tape = qnode.qtape
        freqs = {}
        qubits_layers = {}
        redundancy_gate_list = {}
        # obtain index of first encoding gate

        for op in tape.operations:
            id = op.id
            # if the operator has no specific ID, move to the next
            if id is None:
                continue

            # if user has not specified encoding_gate id's,
            # consider any id
            is_encoding_gate = encoding_gates is None or id in encoding_gates

            if is_encoding_gate:
                if len(op.parameters) != 1:
                    raise ValueError(
                        "Can only consider one-parameter gates as "
                        f"data-encoding gates; got {op.name}."
                    )
                spec = get_spectrum_redundancy(op, decimals=decimals)
                
                if id not in freqs:
                    qubits_layers[id] = 1
                    redundancy_gate_list[id] = [
                        {k: v for k, v in sorted(Counter(spec).items())}
                    ]
                else:
                    qubits_layers[id] += 1
                # if id has been seen before, join this spectrum to another one
                if id in freqs:
                    spec, redundancy_intermediate_dict = join_spectra_redundancy(
                        freqs[id], spec
                    )
                    redundancy_gate_list[id].append(redundancy_intermediate_dict)
                freqs[id] = spec

        # Turn spectra into sorted lists and include negative frequencies
        for id, spec in freqs.items():
            spec = sorted(spec)
            freqs[id] = [-f for f in spec[:0:-1]] + spec

        # Add trivial spectrum for requested gate ids that are not in the circuit
        if encoding_gates is not None:
            for id in set(encoding_gates).difference(freqs):
                freqs[id] = []

        # return (freqs, redundancy_gate_list)
        # We assume that one layer has n gates, 1 per qubit
        # dict specifying qubits used for each encoding layer
        # Select the last cumulative redundancy values of each layer
        redundancy_layer_cumulative = {}
        first_layer = list(redundancy_gate_list.keys())[starting_layer]
        redundancy_layer_cumulative[first_layer] = redundancy_gate_list[first_layer][-1]
        gate_list = list(redundancy_gate_list.keys())
        gate_list = gate_list[starting_layer:]
        # print(gate_list)

        for l_idx, l in enumerate(gate_list[:-1]):
            (
                redundancy_layer_cumulative[gate_list[l_idx + 1]],
                _,
            ) = join_spectra_redundancy(
                redundancy_gate_list[gate_list[l_idx]][-1],
                redundancy_layer_cumulative[l],
            )
            redundancy_layer_cumulative[l] = Counter(redundancy_layer_cumulative[l])
        redundancy_layer_cumulative[gate_list[-1]] = Counter(
            redundancy_layer_cumulative[gate_list[-1]]
        )

        return redundancy_layer_cumulative

    return wrapper
