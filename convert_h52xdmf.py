from pathlib import Path
import argparse

from fenics import Mesh, MPI, HDF5File, VectorFunctionSpace, Function, XDMFFile

def read_command_line():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        description="Convert h5 to xdmf.") 
    parser.add_argument('--i', '-input', type=Path, help="Path to input folder.")

    return parser.parse_args()

def convert_h52xdmf(input_folder):
    """
    Convert h5 to xdmf.
    """
    u_mean_path = input_folder / "Uavg.h5"
    mesh_path = input_folder / "Mesh.h5"

    # read mesh from h5
    mesh = Mesh()
    h5 = HDF5File(MPI.comm_world, str(mesh_path), "r")
    h5.read(mesh, "Mesh", False)
    h5.close()
    
    # Define functionspace and function for mean velocity
    Vv = VectorFunctionSpace(mesh, "CG", 1)
    u_mean = Function(Vv)

    # create write for mean velocity
    u_mean_xdmf_path = input_folder / "u_mean_viz.xdmf"
    u_mean_xdmf = XDMFFile(MPI.comm_world, str(u_mean_xdmf_path))
    
    # read mean velocity from h5
    u_mean_h5 = HDF5File(MPI.comm_world, str(u_mean_path), "r")
    vec_name = "Uavg"
    u_mean_h5.read(u_mean, vec_name)

    u_mean_xdmf.write(u_mean, 0.0)



if __name__ == "__main__":
    args = read_command_line()
    convert_h52xdmf(args.i)