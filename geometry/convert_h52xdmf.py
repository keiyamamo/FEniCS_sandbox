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
    # u_mean_path = input_folder / "Uavg.h5"
    u_mean_path = input_folder / "u_mean.h5"
    # tke_path = input_folder / "turbulent_kinetic_energy.h5"
    mesh_path = input_folder / "mesh.h5"

    # read mesh from h5
    mesh = Mesh()
    h5 = HDF5File(MPI.comm_world, str(mesh_path), "r")
    h5.read(mesh, "mesh", False)
    h5.close()
    
    # Define functionspace and function for mean velocity
    Vv = VectorFunctionSpace(mesh, "CG", 2)
    u_mean = Function(Vv)

    # create write for mean velocity
    # u_mean_xdmf_path = input_folder / "tke_viz.xdmf"
    u_mean_xdmf_path = input_folder / "u_mean_checkpoint.xdmf"
    u_mean_xdmf = XDMFFile(MPI.comm_world, str(u_mean_xdmf_path))
    
    # read mean velocity from h5
    u_mean_h5 = HDF5File(MPI.comm_world, str(u_mean_path), "r")
    # tke_h5 = HDF5File(MPI.comm_world, str(tke_path), "r")
    vec_name = "turbulent_kinetic_energy/turbulent_kinetic_energy_0/vector"
    vec_name = "u_mean/vector_0"
    u_mean_h5.read(u_mean, vec_name)
    # tke_h5.read(u_mean, vec_name)

    u_mean_xdmf.write_checkpoint(u_mean, "u_mean", 0, XDMFFile.Encoding.HDF5, append=False)



if __name__ == "__main__":
    args = read_command_line()
    convert_h52xdmf(args.i)