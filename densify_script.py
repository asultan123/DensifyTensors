import gurobipy as gp
from gurobipy import GRB
import numpy as np 
import torch
import scipy
import scipy.sparse
from math import prod


def generate_01_sparse_matrix(size, density):
    f, c, ky, kx = size
    x = scipy.sparse.rand(f, c*ky*kx, density, format='csr')
    x.data[:] = 1
    x = np.array(x.todense()).reshape((f, c, ky, kx))
    return x


def get_non_zero_count(matrix):
    return prod(matrix[np.where(matrix > 0)].shape)


def calc_density(matrix):
    ones = get_non_zero_count(matrix)
    return ones/prod(matrix.shape)


def density_of_remaining_weights(original_matrix, selection_bitmap):
    bitmap = np.zeros(original_matrix.shape[0:2])
    for i, j in selection_bitmap:
        bitmap[i, j] = 1
    not_selected = 1-bitmap
    total_unselected = np.sum(not_selected)
    ignored_weights = np.multiply(not_selected, np.sum(original_matrix, axis=(3, 2)))
    unselected_ones = get_non_zero_count(ignored_weights)
    return  unselected_ones/ total_unselected 



def find_densest_subtensor_in_weight_tensor(
    tensor,
    min_filters=None,
    min_channels=None,
    initialize=False,
    timeout=None,
    avoid_bitmap=None,
):
    f_size, c_size, ky_size, kx_size = tensor.shape
    if min_filters is not None and min_filters > f_size:
        raise ValueError("filter lowerbound must be lower than max filters")
    if min_channels is not None and min_channels > c_size:
        raise ValueError("channel lowerbound must be lower than max filters")

    tensor_cpy = np.copy(tensor)
    # tensor_cpy[np.where(tensor_cpy == 0)] = -1
    m = gp.Model("densify")
    if timeout is not None:
        m.setParam(GRB.Param.TimeLimit, timeout)
    F = m.addVars(f_size, vtype=GRB.BINARY, name="F")
    C = m.addVars(c_size, vtype=GRB.BINARY, name="C")

    if initialize:
        est_filter_density = [
            (i, s) for s, i in zip(np.sum(tensor, axis=(2, 1)), range(tensor.shape[0]))
        ]
        est_filter_density.sort(key=lambda x: x[1], reverse=True)
        est_channel_density = [
            (i, s) for s, i in zip(np.sum(tensor, axis=(2, 0)), range(tensor.shape[1]))
        ]
        est_channel_density.sort(key=lambda x: x[1], reverse=True)
        initial_filters = [i for i, _ in est_filter_density[: min_filters + 1]]
        initial_channels = [j for j, _ in est_channel_density[: min_channels + 1]]

        for i in initial_filters:
            F[i].start = 1
        for j in initial_channels:
            C[j].start = 1

    Z = m.addVars(f_size, c_size, vtype=GRB.BINARY, name="Z")

    if avoid_bitmap is not None:
        m.addConstrs(
            (Z[i, j] == 0 for i, j in avoid_bitmap),
            name="avoid_constraints",
        )

    if min_filters is not None:
        m.addConstr(min_filters == gp.quicksum([F[i] for i in range(len(F))]), name='filter_constraints')
    if min_channels is not None:
        m.addConstr(min_channels == gp.quicksum([C[j] for j in range(len(C))]), name='channel_constraints')
        
    m.addConstrs(
        (Z[i, j] == gp.and_(F[i], C[j]) for i in range(len(F)) for j in range(len(C))),
        name="and_constraints",
    )
    m.setObjective(
        gp.quicksum(
            Z[i, j] * tensor_cpy[i, j, ky, kx]
            for i in range(len(F))
            for j in range(len(C))
            for ky in range(ky_size)
            for kx in range(kx_size)
        ),
        GRB.MAXIMIZE,
    )
    m.optimize()
    if m.status == GRB.INFEASIBLE: # TODO: Shouldn't happen
        m.remove(m.getConstrByName('filter_constraints'))
        m.optimize()
    if m.status == GRB.INFEASIBLE:
        m.remove(m.getConstrByName('channel_constraints'))
        m.optimize()
    if m.status == GRB.INFEASIBLE:
        raise Exception('')

    dense_filter_indicies = [i for i, f in F.items() if f.X > 0.5]
    dense_channel_indicies = [j for j, c in C.items() if c.X > 0.5]
    selection_bitmap = [
        (i, j) for i in range(len(F)) for j in range(len(C)) if Z[i, j].X > 0.5
    ]
    dense_tensor = tensor[dense_filter_indicies, :, :][:, dense_channel_indicies, :]
    return dense_tensor, selection_bitmap, dense_filter_indicies, dense_channel_indicies

def print_selection_bitmap(size, selection_bitmap):
    bitmap = np.zeros(size)
    for i, j in selection_bitmap:
        bitmap[i, j] = 1
    print(bitmap)
    return bitmap

weight_tensor = generate_01_sparse_matrix(size = (16, 16, 3, 3), density = 0.60)
input_tensor = weight_tensor
size = prod(input_tensor.shape[0:2])
density_tracker = []
target_size = 2
selection_bitmap = None
for i in range(size//target_size**2):
    initial_size = input_tensor.shape[0]
    print(f'Reducing from tensor of size {initial_size} to tensors of size {target_size}')
    (
        output_tensor,
        new_selection_bitmap,
        dense_filter_indicies,
        dense_channel_indicies,
    ) = find_densest_subtensor_in_weight_tensor(input_tensor, target_size, None, timeout=25, avoid_bitmap=selection_bitmap)
    if(prod(output_tensor.shape) == 0):
        break
    input_density = calc_density(input_tensor)
    output_density = calc_density(output_tensor)
    density_tracker.append(output_density)
    try:
        selection_bitmap.extend(new_selection_bitmap)
    except AttributeError:
        selection_bitmap = new_selection_bitmap
        
    remaining_density = density_of_remaining_weights(input_tensor, selection_bitmap)
    
    print(f"density of input tensor: {input_density}")
    print(f"density of output tensor: {output_density}")
    print(f"selected filters: {dense_filter_indicies}")
    print(f"selected channels: {dense_channel_indicies}")
    print_selection_bitmap(input_tensor.shape[0:2], selection_bitmap)

# TODO: Fix bug in density tracker with random extra 0s at the end when density is too low
print(density_tracker)
print('')