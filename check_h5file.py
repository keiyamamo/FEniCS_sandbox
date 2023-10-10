import os
import h5py
import numpy as np
from argparse import ArgumentParser

"""
Author: Kei Yamamoto <keiya@simula.no>
Last updated: 2023/08/31
When restarting a simulation in turtleFSI, the visualization files are not always correct due to different mesh partitioning.
This scripts fixes the visualization files by checking mesh in h5 files and swapping the node numbering, tpoology, and vector values.
After running this script, you can use the combine_xdmf.py script to merge the visualization files if xdmf files are not yet merged.
TODO: 1. reanme wrong / correct to run / run_1
      2. add support for more than 2 runs
      3. remove --correct and --wrong arguments and find the correct and wrong files automatically

Example of usage: python check_h5file.py --folder /cluster/work/users/keiya/case16_mrmodel/Visualization/ --correct pressure.h5 --wrong pressure_run_1.h5
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--folder", help="Path to the folder containing the visualization files")
    parser.add_argument("--correct", help="Path to the correct visualization file")
    parser.add_argument("--wrong", help="Path to the wrong visualization file")
    args = parser.parse_args()
    return args

def main(folder, correct, wrong):
    """
    Args:
        folder (str): Path to the folder containing the visualization files
        correct (str): Path to the correct visualization file, usually velocity/displacement/pressure.h5 
        wrong (str): Path to the wrong visualization file, usually velocity/displacement/pressure_run_{i}.h5, i = 1, 2, 3, ...

    Returns:
        None
    """
    # Here we find the path to the visualization files 
    wrongNumberVizPath = os.path.join(folder, wrong)
    correctNumberVizPath = os.path.join(folder, correct)

    # Open the files using h5py, r+ means read and write
    with h5py.File(wrongNumberVizPath, 'r+') as wrongNumberViz, \
        h5py.File(correctNumberVizPath, 'r+') as correctNumberViz:
    
        # Get the mesh coordinates from the mesh
        wrongNumberNodes = wrongNumberViz['Mesh/0/mesh/geometry'][:]
        correctNumberNodes = correctNumberViz['Mesh/0/mesh/geometry'][:]
        #Here, we simply copy toplogy from the correct file to the wrong file if they are not the same
        if (correctNumberViz['Mesh/0/mesh/topology'][:] != wrongNumberViz['Mesh/0/mesh/topology'][:]).all():
            print('Topology is not the same')
        else:
            print('Topology is the same')

        # Check if the node numbering is correct
        if (correctNumberNodes == wrongNumberNodes).all():
            print('Node numbering is correct')
            print('...exiting')
            wrongNumberViz.close()
            correctNumberViz.close()
            exit()
        else:
            print('Node numbering is incorrect')
        
        # add index to the node coordinates
        indexed_correctNumberNodes = np.hstack((np.arange(len(correctNumberNodes), dtype=int).reshape(-1, 1), correctNumberNodes))
        indexed_wrongNumberNodes = np.hstack((np.arange(len(wrongNumberNodes), dtype=int).reshape(-1, 1), wrongNumberNodes))

        # sort the node coordinates based on the x, y, and z coordinates (here we assumme 3D mesh)
        # after sorting, the index is the mapping from the unsorted node numbering to the sorted node numbering
        if indexed_correctNumberNodes[:,1].size == np.unique(indexed_correctNumberNodes[:, 1]).size:
            print('x coordinate is unique and sort based on x coordinate only')
            sorted_correctNumberNodes = np.argsort(indexed_correctNumberNodes[:, 0])
            sorted_wrongNumberNodes = np.argsort(indexed_wrongNumberNodes[:, 0])
            print("Done sorting")
        else:
            print('x coordinate is not unique and sort based on x, y, and z coordinates')
            if  correctNumberNodes.shape[1] == 2:
                sorted_correctNumberNodes = np.lexsort((indexed_correctNumberNodes[:, 1], indexed_correctNumberNodes[:, 2]))
                sorted_wrongNumberNodes = np.lexsort((indexed_wrongNumberNodes[:, 1], indexed_wrongNumberNodes[:, 2]))
                print("Done sorting")
            elif correctNumberNodes.shape[1] == 3:
                sorted_correctNumberNodes = np.lexsort((indexed_correctNumberNodes[:, 1], indexed_correctNumberNodes[:, 2], indexed_correctNumberNodes[:, 3]))
                sorted_wrongNumberNodes = np.lexsort((indexed_wrongNumberNodes[:, 1], indexed_wrongNumberNodes[:, 2], indexed_wrongNumberNodes[:, 3]))
                print("Done sorting")
        
        # sort the node coordinates based on the index
        # First, extract the index from the sorted node coordinates
        print("Extracting the index from the sorted node coordinates")
        map_index_correctNumberNodes = indexed_correctNumberNodes[sorted_correctNumberNodes][:, 0].astype(int)
        map_index_wrongNumberNodes = indexed_wrongNumberNodes[sorted_wrongNumberNodes][:, 0].astype(int)
        
        # Then, sort the node coordinates based on the index
        print("Sorting the node coordinates based on the index")
        correctNumberNodes = correctNumberNodes[map_index_correctNumberNodes]

        # Overwrite the h5 file with the sorted node coordinates
        print("Overwriting the h5 file with the sorted node coordinates")
        wrongNumberViz['Mesh/0/mesh/geometry'][...] = correctNumberNodes
        correctNumberViz['Mesh/0/mesh/geometry'][...] = correctNumberNodes

        # Also sort the vectors in the h5 file
        print("Sorting the vectors in the h5 file")
        # Also sort the vectors in the h5 file
        for i in range(len(wrongNumberViz["VisualisationVector"].keys())):
            if i % 100 == 0:
                print(f"Sorting the vector {i} out of {len(wrongNumberViz['VisualisationVector'].keys())}")
            velocity_vector = wrongNumberViz["VisualisationVector"][str(i)][:, :]
            ordered_velocity_vector = velocity_vector[map_index_wrongNumberNodes]
            wrongNumberViz["VisualisationVector"][str(i)][...] = ordered_velocity_vector
        
        for i in range(len(correctNumberViz["VisualisationVector"].keys())):
            if i % 100 == 0:
                print(f"Sorting the vector {i} out of {len(correctNumberViz['VisualisationVector'].keys())}")
            velocity_vector = correctNumberViz["VisualisationVector"][str(i)][:, :]
            ordered_velocity_vector = velocity_vector[map_index_correctNumberNodes]
            correctNumberViz["VisualisationVector"][str(i)][...] = ordered_velocity_vector
        
        print("Done sorting the vectors in the h5 file")
        ordered_map_index_wrongNumberNodes = np.argsort(map_index_wrongNumberNodes)

        # Also sort the topology in the h5 file
        # this loop replaces the node numbers in the topology array one by one
        print("Sorting the topology in the h5 file")
        wrongNumberTopology = wrongNumberViz['Mesh/0/mesh/topology'][:]
        wrongNumberTopology = np.rint(ordered_map_index_wrongNumberNodes[wrongNumberTopology])

        wrongNumberViz['Mesh/0/mesh/topology'][...] = wrongNumberTopology
        correctNumberViz['Mesh/0/mesh/topology'][...] = wrongNumberTopology


if __name__ == '__main__':
    args = parse_args()
    folder = args.folder
    correct = args.correct
    wrong = args.wrong
    main(folder, correct, wrong)