import numpy as np
from copy import deepcopy
from math import factorial
from scipy import spatial
from quantum_algorithms import *
from qiskit.circuit.library import CHGate
from qiskit.circuit.library import MCMT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit_aer.noise import pauli_error
from qiskit.quantum_info import state_fidelity, DensityMatrix, partial_trace
from qiskit import transpile

def convert(input_table):
    # input table is a list where each element is a list with 2 elements, converting it to a state vector, e.g. [[1, 2], [3, 4]]
    normalize_factor = 1 / np.sqrt(2 ** len(input_table))
    state_vector = np.zeros(2 ** (len(input_table) * 2), dtype=np.float64)
    non_zero_items = []
    for i in range(2 ** (len(input_table))):
        # get binary representation of i
        i_bin = bin(i)[2:].zfill(len(input_table))
        tmp_string = [-1 for _ in range(len(input_table) * 2)]
        for j in range(len(i_bin)):
            if i_bin[j] == '0':
                index1 = input_table[j][0] - 1
                index2 = input_table[j][1] - 1
                tmp_string[index1] = 0
                tmp_string[index2] = 1
            else:
                index1 = input_table[j][0] - 1
                index2 = input_table[j][1] - 1
                tmp_string[index1] = 1
                tmp_string[index2] = 0
        # get the value of tmp_string
        tmp_string = ''.join([str(i) for i in tmp_string])
        tmp_value = int(tmp_string, 2)
        # count the number of 1s in tmp_string
        count = i_bin.count('1')
        if count % 2 == 0:
            state_vector[tmp_value] = normalize_factor
            non_zero_items.append(tmp_string)
        else:
            state_vector[tmp_value] = -normalize_factor
            non_zero_items.append('-' + tmp_string)
    return state_vector, non_zero_items

def calculate_rank(input_tables):
    table_matrix = np.zeros((len(input_tables), 2 ** (len(input_tables[0]) * 2)), dtype=np.float64)
    for i in range(len(input_tables)):
        table_matrix[i], _ = convert(input_tables[i])
    rank = np.linalg.matrix_rank(table_matrix)
    return rank

def generate_tables(N, current_table):
    if current_table is None:
        current_table = []
    if (2 * len(current_table)) == N:
        vaild_tables.append(current_table)
        return
    remaining_numbers = [i for i in range(1, N + 1)]
    for item in current_table:
        remaining_numbers.remove(item[0])
        remaining_numbers.remove(item[1])
    next_item = [-1, -1]
    next_item[0] = min(remaining_numbers)
    remaining_numbers.remove(next_item[0])
    for number in remaining_numbers:
        next_item[1] = number
        tmp = deepcopy(current_table)
        tmp.append(next_item)
        generate_tables(N, tmp)

def vector_number_in_complete_table(N):
    return int(factorial(N) / (factorial(int(N/2)) * factorial(int(N/2) + 1)))

def generate_complete_set(N):
    # generate complete set table of DFS states for N qubits
    if N % 2 != 0:
        raise ValueError('N must be even')
    global vaild_tables
    vaild_tables = []
    generate_tables(N, [])
    complete_set = []
    current_rank = 0
    target_rank = vector_number_in_complete_table(N)
    for table in vaild_tables:
        complete_set.append(table)
        calculated_rank = calculate_rank(complete_set)
        if calculated_rank > current_rank:
            current_rank = calculated_rank
            if current_rank == target_rank:
                return complete_set
        else:
            complete_set.pop()
    return None

def state_vector_to_string(state_vector):
    # convert state vector to string representation
    state_vector = np.round(state_vector, 10)
    # find the minimum abs non-zero value
    min_value = 1
    for i in range(len(state_vector)):
        if state_vector[i] != 0 and abs(state_vector[i]) < min_value:
            min_value = abs(state_vector[i])
    non_zero_items = []
    for i in range(len(state_vector)):
        if state_vector[i] != 0:
            non_zero_items.append((bin(i)[2:].zfill(int(np.log2(len(state_vector)))), state_vector[i]/min_value))
    return non_zero_items

def show_correct_answer_strings(N):
    # for N qubits, show the correct answer in string representation
    _, basis_string = correct_answer(N)
    for i in range(len(basis_string)):
        print(f'|u_{i}> = ', end='')
        for item in basis_string[i]:
            if item[1] == 1:
                print(f'+|{item[0]}>', end='')
            elif item[1] == -1:
                print(f'-|{item[0]}>', end='')
            else:
                if item[1] > 0:
                    print(f'+{item[1]:.2f}|{item[0]}>', end='')
                else:
                    print(f'{item[1]:.2f}|{item[0]}>', end='')
        print()

def QGS_supersinglets(input_tables, circuit_return=True, error_rate=1e-4):
    # orthogonalize the input supersinglet states using quantum gram schmidt, a trivial method
    input_matrix = np.zeros((len(input_tables), 2 ** (len(input_tables[0]) * 2)), dtype=np.float64)
    for i in range(len(input_tables)):
        print(input_tables[i])
        input_matrix[i], _ = convert(input_tables[i])
    circuit_set, basis, length = quantum_gram_schmidt(input_matrix, circuit_return=circuit_return, error_rate=error_rate)
    return circuit_set, basis, length

def CGS_supersinglets(input_tables):
    # orthogonalize the input supersinglet states using classical gram schmidt, a trivial method
    input_matrix = np.zeros((len(input_tables), 2 ** (len(input_tables[0]) * 2)), dtype=np.float64)
    for i in range(len(input_tables)):
        input_matrix[i], _ = convert(input_tables[i])
    # use QR decomposition to get the orthogonal matrix
    Q, R = np.linalg.qr(input_matrix.T)
    basis = Q.T
    # basis[i] is |u_i>
    return basis

def correct_answer(N):
    # for N qubits, return the correct answer basis and its string representation
    if N % 2 != 0:
        raise ValueError('N must be even')
    # generate the complete set
    complete_set = generate_complete_set(N)
    basis = CGS_supersinglets(complete_set)
    # convert the basis to a string
    basis_string = []
    for i in range(len(basis)):
        basis_string.append(state_vector_to_string(basis[i]))

    return basis, basis_string

def trival_construction(N):
    # suppose we have state tomography technique, we perform orthogonalization immediately
    table = generate_complete_set(N)
    # print(table)
    circuit_set, basis, length = QGS_supersinglets(table)
    print('the number of generated basis is ', length)
    print('the number of a complete set is ', vector_number_in_complete_table(N))

    print('for super singlets:')

    print(state_vector_to_string(basis[0]))
    print(state_vector_to_string(basis[1]))
    print(state_vector_to_string(basis[2]))
    print(state_vector_to_string(basis[3]))

#--------------------------------- now realize the quantum DFS preaparation circuit ---------------------------------#
def from_m_to_m_plus_one(qc, input_state, k):
    """
    construct a_{k+1}^(m+1) from a_{k+1}^(m)
    """
    total_qubit_number = qc.num_qubits
    running_times = 0
    # reverse the order of input_state
    while True:
        tmp_qc = QuantumCircuit(total_qubit_number)
        tmp_qc.initialize(input_state, range(k, total_qubit_number))
        tmp_qc = tmp_qc.compose(qc, range(total_qubit_number), inplace=False)
        simulator = Aer.get_backend('statevector_simulator')
        job = execute(tmp_qc, simulator, shots=1)
        result = job.result()
        running_times += 1
        target_result = '0' * k
        if target_result in result.get_counts().keys():
            #print(result.get_counts())
            statevector = np.asarray(result.get_statevector())
            # print(state_vector_to_string(statevector))
            reduced_state_vector = np.zeros(len(input_state), dtype=complex)
            for i in range(len(input_state)):
                reduced_state_vector[i] = statevector[i* 2 ** (k)]
            return reduced_state_vector, running_times
        
def construct_oracle_for_table(current_table, ancilla_number, control_qubit=-1):
    """
    construct the oracle for the table, the oracle is a quantum circuit
    current_table: the current table, a list of tuples, e.g. [(1, 2), (3, 4)]
    ancilla_number: the number of ancilla qubits
    return: the oracle circuit
    """
    # get the number of qubits in the circuit
    if control_qubit < 0:
        qubit_number = len(current_table) * 2 + ancilla_number
    else:
        qubit_number = len(current_table) * 2 + ancilla_number
    #print('the number of qubits in the circuit is ', qubit_number)
    oracle = QuantumCircuit(qubit_number)
    system_qubit_number = 2 * len(current_table)
    # apply CNOT gates to all pairs in the table
    if control_qubit == -1:
        for i in range(len(current_table)):
            oracle.h(system_qubit_number - current_table[i][0] + ancilla_number)
            oracle.x(system_qubit_number - current_table[i][1] + ancilla_number)
            oracle.cx(system_qubit_number - current_table[i][0]+ancilla_number, system_qubit_number - current_table[i][1]+ancilla_number)
            oracle.z(system_qubit_number - current_table[i][1] + ancilla_number)
        return oracle
    else:
        for i in range(len(current_table)):
            oracle.append(CHGate(), [control_qubit, system_qubit_number - current_table[i][0]+ancilla_number])
            oracle.cx(control_qubit, system_qubit_number - current_table[i][1] + ancilla_number)
            oracle.ccx(control_qubit, system_qubit_number - current_table[i][0]+ancilla_number, system_qubit_number - current_table[i][1]+ancilla_number)
            oracle.cz(control_qubit, system_qubit_number - current_table[i][1] + ancilla_number)
            oracle.barrier()
        return oracle

def construct_u_k_plus_one(previous_bases, a_k_plus_one, system_size, error_rate=1e-10, fix_iter=10, choose=1):
    """previous_bases[i] and a_k_plus_one are all like [[1,2],[3,4]]"""
    k = len(previous_bases)
    #print(f'constructing U_{k+1} from previous bases and a_{k+1}')
    #print(f'the k+1 th table is {a_k_plus_one}')
    correct_states, _ = correct_answer(system_size)
    qc = QuantumCircuit(k+system_size, k)
    initial_input_state, _ = convert(a_k_plus_one)
    
    for i in range(0,k):
        qc.h(i)
        tmp_circuit = construct_oracle_for_table(previous_bases[i], k, i)
        qc = qc.compose(tmp_circuit.inverse(), qubits=range(k + system_size))

        qc.barrier()
        for j in range(k, k+ system_size):
            qc.x(j)

    
        mcz = MCMT('cz', system_size, 1)
        applied_qubits = [i] + [j for j in range(k, k + system_size)]
        qc = qc.compose(mcz, qubits=applied_qubits)

        for j in range(k, k + system_size):
            qc.x(j)

        qc.barrier()

        qc = qc.compose(tmp_circuit, qubits=range(k + system_size))
        qc.h(i)
        qc.barrier()

    for i in range(0, k):
        qc.measure(i, i)

    current_input_state = initial_input_state
    total_running_number = 0 # how many times the qc needed to be run?
    total_running_times = 0  # how many m needed?
    infidelities = [1 - np.abs(np.dot(current_input_state, correct_states[k]))**2]
    if choose == 1:
        while True:
            current_output_state, running_number = from_m_to_m_plus_one(qc, current_input_state, k)
            total_running_number = running_number * (total_running_number + 1)
            current_infidelity = 1 - np.abs(np.dot(current_output_state, correct_states[k]))**2
            infidelities.append(current_infidelity)
            total_running_times += 1
            if current_infidelity < error_rate:
                break
            else:
                #print('infidelity is ', current_infidelity)
                #print(f'current state is {state_vector_to_string(current_output_state)}')
                #print(f'correct state is {state_vector_to_string(correct_states[k])}')
                current_input_state = current_output_state
    elif choose == 2:
        for i in range(fix_iter):
            current_output_state, running_number = from_m_to_m_plus_one(qc, current_input_state, k)
            total_running_number = running_number * (total_running_number + 1)
            current_infidelity = 1 - np.abs(np.dot(current_output_state, correct_states[k]))**2
            infidelities.append(current_infidelity)
            total_running_times += 1
            #print(f'iteration {i+1}, infidelity is ', current_infidelity)
            #print(f'current state is {state_vector_to_string(current_output_state)}')
            #print(f'correct state is {state_vector_to_string(correct_states[k])}')
            current_input_state = current_output_state
    else:
        raise ValueError('choose must be 1 or 2')
    info_output = (k+1, error_rate, total_running_times, total_running_number, infidelities)
    return current_output_state, qc, info_output        
        
def nontrival_construction_1(N, error_rate=1e-10):
    # iterate till convergence
    table = generate_complete_set(N)
    results = []
    qcs = []
    infos = []
    first_state, _ = convert(table[0])
    results.append(first_state)
    infos.append((1,error_rate, 1, 1))
    qcs.append(construct_oracle_for_table(table[0], 0))
    for i in range(1, len(table)):
        current_state, qc, current_info = construct_u_k_plus_one(table[:i], table[i], system_size=N, error_rate=error_rate, choose=1)
        results.append(current_state)
        qcs.append(qc)
        infos.append(current_info)
    
    return qcs, results, infos

def nontrival_construction_2(N):
    # fix iteration number
    table = generate_complete_set(N)
    results = []
    qcs = []
    infos = []
    first_state, _ = convert(table[0])
    results.append(first_state)
    infos.append((1,1e-10, 1, 1))
    qcs.append(construct_oracle_for_table(table[0], 0))
    for i in range(1, len(table)):
        current_state, qc, current_info = construct_u_k_plus_one(table[:i], table[i], system_size=N, error_rate=1e-10, fix_iter=10, choose=2)
        results.append(current_state)
        qcs.append(qc)
        infos.append(current_info)
    
    return qcs, results, infos

#--------------------------------- now consider the noisy case --------------------------------------------#
def pauli_channel_scenario(single_qubit_error_rate=0.001, two_qubit_error_rate=0.01, measurement_error_rate=0.01):
    # define a noise model using Pauli channels
    """
    Use Pauli channel when:
    - You have asymmetric error rates (common in real hardware)
    - You want to model specific dominant error types
    - You're simulating specific hardware characteristics
    """
    noise_model = NoiseModel()
    
    # Real hardware often has different rates for X, Y, Z errors
    single_qubit_pauli = pauli_error([
        ('X', single_qubit_error_rate * 0.5),  # Most common in some architectures
        ('Z', single_qubit_error_rate * 0.5),  # Common due to dephasing
        ('I', 1 - single_qubit_error_rate)   # No error
    ])
    
    # Two-qubit gates might have correlated errors
    two_qubit_pauli = pauli_error([
        ('IX', two_qubit_error_rate * 0.2), ('XI', two_qubit_error_rate * 0.2), ('XX', two_qubit_error_rate * 0.1),
        ('IZ', two_qubit_error_rate * 0.2), ('ZI', two_qubit_error_rate * 0.2), ('ZZ', two_qubit_error_rate * 0.1),
        ('II', 1 - two_qubit_error_rate)  # No error
    ])

    meas_error = pauli_error([
        ('X', measurement_error_rate),  # Simple symmetric measurement error
        ('I', 1 - measurement_error_rate)
    ])
    
    noise_model.add_all_qubit_quantum_error(single_qubit_pauli, ['h', 'x', 'y', 'z'])
    noise_model.add_all_qubit_quantum_error(two_qubit_pauli, ['cx'])
    noise_model.add_all_qubit_quantum_error(meas_error, ['measure'])
    
    return noise_model

def calculate_noisy_final_state(noise_model, qc, ancilla_number):
    # check the fidelity of the output state under noise model
    while True:
        simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
        qc.save_density_matrix()
        result = simulator.run(qc, shots=1).result()
        measurement = result.data(0)['counts'].keys()
        if '0x0' in measurement:
            noisy_state = result.data(0)['density_matrix']
            traced_qubits = list(range(ancilla_number)) + [qc.num_qubits-2, qc.num_qubits - 1]
            output_state = partial_trace(noisy_state, traced_qubits).data
            return output_state


def construct_u_k_plus_one_noisy(qc, a_k_plus_one, state_index, correct_state=None, iter_time=1, single_qubit_error_rate=0.001, two_qubit_error_rate=0.01, measurement_error_rate=0.01):
    """previous_bases[i] and a_k_plus_one are all like [[1,2],[3,4]]"""
    '''
    qc: the quantum circuit to be used, already constructed, removed the measurement part
    a_k_plus_one: the table for a_{k+1}
    correct_state: the correct state vector for u_{k+1}
    state_index: the index k
    '''
    ancilla_number = state_index
    # density matrix
    tmp_qc = QuantumCircuit(qc.num_qubits, ancilla_number * iter_time)
    system_qubit_number = len(a_k_plus_one) * 2
    for i in range(len(a_k_plus_one)):
        tmp_qc.h(system_qubit_number - a_k_plus_one[i][0] + ancilla_number)
        tmp_qc.x(system_qubit_number - a_k_plus_one[i][1] + ancilla_number)
        tmp_qc.cx(system_qubit_number - a_k_plus_one[i][0]+ancilla_number, system_qubit_number - a_k_plus_one[i][1]+ancilla_number)
        tmp_qc.z(system_qubit_number - a_k_plus_one[i][1] + ancilla_number)
    for i in range(iter_time):
        tmp_qc = tmp_qc.compose(qc, range(qc.num_qubits))
        for j in range(ancilla_number):
            tmp_qc.measure(j, j + i * ancilla_number)
    tmp_qc.barrier()
    noisy_model = pauli_channel_scenario(single_qubit_error_rate=single_qubit_error_rate, two_qubit_error_rate=two_qubit_error_rate, measurement_error_rate=measurement_error_rate)
    noisy_final_state = calculate_noisy_final_state(noisy_model, tmp_qc, ancilla_number)
    ideal_final_state = np.outer(correct_state, correct_state.conj())
    fidelity = state_fidelity(DensityMatrix(ideal_final_state), DensityMatrix(noisy_final_state))
    return fidelity.real

def calculate_noisy_final_state_for_6(noise_model, qc, ancilla_number):
    # check the fidelity of the output state under noise model
    while True:
        simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
        qc.save_density_matrix()
        result = simulator.run(qc, shots=1).result()
        measurement = result.data(0)['counts'].keys()
        if '0x0' in measurement:
            noisy_state = result.data(0)['density_matrix']
            traced_qubits = list(range(ancilla_number))
            output_state = partial_trace(noisy_state, traced_qubits).data
            return output_state


def construct_u_k_plus_one_noisy_for_6(qc, a_k_plus_one, state_index, correct_state=None, iter_time=1, single_qubit_error_rate=0.001, two_qubit_error_rate=0.01, measurement_error_rate=0.01):
    """previous_bases[i] and a_k_plus_one are all like [[1,2],[3,4]]"""
    '''
    qc: the quantum circuit to be used, already constructed, removed the measurement part
    a_k_plus_one: the table for a_{k+1}
    correct_state: the correct state vector for u_{k+1}
    state_index: the index k
    '''
    ancilla_number = state_index
    # density matrix
    tmp_qc = QuantumCircuit(qc.num_qubits, ancilla_number * iter_time)
    system_qubit_number = len(a_k_plus_one) * 2
    for i in range(len(a_k_plus_one)):
        tmp_qc.h(system_qubit_number - a_k_plus_one[i][0] + ancilla_number)
        tmp_qc.x(system_qubit_number - a_k_plus_one[i][1] + ancilla_number)
        tmp_qc.cx(system_qubit_number - a_k_plus_one[i][0]+ancilla_number, system_qubit_number - a_k_plus_one[i][1]+ancilla_number)
        tmp_qc.z(system_qubit_number - a_k_plus_one[i][1] + ancilla_number)
    for i in range(iter_time):
        tmp_qc = tmp_qc.compose(qc, range(qc.num_qubits))
        for j in range(ancilla_number):
            tmp_qc.measure(j, j + i * ancilla_number)
    tmp_qc.barrier()
    noisy_model = pauli_channel_scenario(single_qubit_error_rate=single_qubit_error_rate, two_qubit_error_rate=two_qubit_error_rate, measurement_error_rate=measurement_error_rate)
    noisy_final_state = calculate_noisy_final_state_for_6(noisy_model, tmp_qc, ancilla_number)
    ideal_final_state = np.outer(correct_state, correct_state.conj())
    fidelity = state_fidelity(DensityMatrix(ideal_final_state), DensityMatrix(noisy_final_state))
    return fidelity.real
