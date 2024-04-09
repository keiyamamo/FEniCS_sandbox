
import argparse
from pathlib import Path
import numpy as np

from vasp.simulations.simulation_common import load_mesh_and_data
from dolfin import Mesh, MPI, Measure, Constant, assemble


def parse_arguments():
    """Read arguments from commandline"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file. If not given (None), " +
                             "it will assume that mesh is located <folder_path>/Mesh/mesh.h5)")
    args = parser.parse_args()

    return args

def print_mesh_summary(mesh: Mesh) -> None:
    """
    Print a summary of geometric information about the volumetric mesh.

    Args:
        mesh (dolfin.Mesh): Volumetric mesh object.
    """
    # Check if the input mesh is of the correct type
    if not isinstance(mesh, Mesh):
        raise ValueError("Invalid mesh object provided.")

    # Extract local x, y, and z coordinates from the mesh
    local_x_coords = mesh.coordinates()[:, 0]
    local_y_coords = mesh.coordinates()[:, 1]
    local_z_coords = mesh.coordinates()[:, 2]

    # Create a dictionary to store local geometric information
    local_info = {
        "x_min": local_x_coords.min(),
        "x_max": local_x_coords.max(),
        "y_min": local_y_coords.min(),
        "y_max": local_y_coords.max(),
        "z_min": local_z_coords.min(),
        "z_max": local_z_coords.max(),
        "num_cells": mesh.num_cells(),
        "num_edges": mesh.num_edges(),
        "num_faces": mesh.num_faces(),
        "num_facets": mesh.num_facets(),
        "num_vertices": mesh.num_vertices()
    }

    # Gather local information from all processors to processor 0
    comm = mesh.mpi_comm()
    gathered_info = comm.gather(local_info, 0)
    num_cells_per_processor = comm.gather(local_info["num_cells"], 0)

    # Compute the volume of the mesh
    dx = Measure("dx", domain=mesh)
    volume = assemble(Constant(1) * dx)

    # Print the mesh information summary only on processor 0
    if MPI.rank(comm) == 0:
        # Combine gathered information to get global information
        combined_info = {key: sum(info[key] for info in gathered_info) for key in gathered_info[0]}
        

        # Print various mesh statistics
        print("=== Mesh Information Summary ===")
        print(f"X range: {combined_info['x_min']} to {combined_info['x_max']} "
              f"(delta: {combined_info['x_max'] - combined_info['x_min']:.4f})")
        print(f"Y range: {combined_info['y_min']} to {combined_info['y_max']} "
              f"(delta: {combined_info['y_max'] - combined_info['y_min']:.4f})")
        print(f"Z range: {combined_info['z_min']} to {combined_info['z_max']} "
              f"(delta: {combined_info['z_max'] - combined_info['z_min']:.4f})")
        print(f"Number of cells: {combined_info['num_cells']}")
        print(f"Number of cells per processor: {int(np.mean(num_cells_per_processor))}")
        print(f"Number of edges: {combined_info['num_edges']}")
        print(f"Number of faces: {combined_info['num_faces']}")
        print(f"Number of facets: {combined_info['num_facets']}")
        print(f"Number of vertices: {combined_info['num_vertices']}")
        print(f"Volume: {volume}")
        print(f"Number of cells per volume: {combined_info['num_cells'] / volume}\n")



def main() -> None:

    mesh_path = parse_arguments().mesh_path
    mesh, _, _ = load_mesh_and_data(mesh_path)
    print_mesh_summary(mesh)

if __name__ == "__main__":
    main()

