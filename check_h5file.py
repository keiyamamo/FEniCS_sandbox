import os
import h5py
import numpy as np
from argparse import ArgumentParser
import time

"""
Author: Kei Yamamoto <keiya@simula.no>
Last updated: 2023/05/19
When restarting a simulation in turtleFSI, the visualization files are not always correct due to different mesh partitioning.
This scripts fixes the visualization files by checking mesh in h5 files and swapping the node numbering, tpoology, and vector values.
After running this script, you can use the combine_xdmf.py script to merge the visualization files.
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
    wrongNumberViz = h5py.File(wrongNumberVizPath, 'r+')
    correctNumberViz = h5py.File(correctNumberVizPath, 'r+')

    # Get the mesh coordinates from the mesh
    wrongNumberNodes = wrongNumberViz['Mesh/0/mesh/geometry'][:]
    correctNumberNodes = correctNumberViz['Mesh/0/mesh/geometry'][:]

    # Here, we simply copy toplogy from the correct file to the wrong file if they are not the same
    if (correctNumberViz['Mesh/0/mesh/topology'][:] != wrongNumberViz['Mesh/0/mesh/topology'][:]).all():
        print('Topology is not the same')
        wrongNumberViz['Mesh/0/mesh/topology'][...] = correctNumberViz['Mesh/0/mesh/topology'][:]
        print('Topology is now fixed')
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
    if  correctNumberNodes.shape[1] == 2:
        sorted_correctNumberNodes = np.lexsort((indexed_correctNumberNodes[:, 1], indexed_correctNumberNodes[:, 2]))
        sorted_wrongNumberNodes = np.lexsort((indexed_wrongNumberNodes[:, 1], indexed_wrongNumberNodes[:, 2]))
    elif correctNumberNodes.shape[1] == 3:
        sorted_correctNumberNodes = np.lexsort((correctNumberNodes[:, 1], correctNumberNodes[:, 2], correctNumberNodes[:, 3]))
        sorted_wrongNumberNodes = np.lexsort((wrongNumberNodes[:, 1], wrongNumberNodes[:, 2], wrongNumberNodes[:, 3]))
    
    # sort the node coordinates based on the index
    # First, extract the index from the sorted node coordinates
    map_index_correctNumberNodes = indexed_correctNumberNodes[sorted_correctNumberNodes][:, 0].astype(int)
    map_index_wrongNumberNodes = indexed_wrongNumberNodes[sorted_wrongNumberNodes][:, 0].astype(int)
    
    # Then, sort the node coordinates based on the index
    correctNumberNodes = correctNumberNodes[map_index_correctNumberNodes]

    # Overwrite the h5 file with the sorted node coordinates
    wrongNumberViz['Mesh/0/mesh/geometry'][...] = correctNumberNodes
    correctNumberViz['Mesh/0/mesh/geometry'][...] = correctNumberNodes

    # Also sort the vectors in the h5 file
    for i in range(len(wrongNumberViz["VisualisationVector"].keys())):
        wrongNumberViz["VisualisationVector"][str(i)][...] = np.array(wrongNumberViz["VisualisationVector"][str(i)])[map_index_wrongNumberNodes]
    
    for i in range(len(correctNumberViz["VisualisationVector"].keys())):
        correctNumberViz["VisualisationVector"][str(i)][...] = np.array(correctNumberViz["VisualisationVector"][str(i)])[map_index_correctNumberNodes]
    
    ordered_map_index_wrongNumberNodes = np.argsort(map_index_wrongNumberNodes)

    # # Also sort the tpoology in the h5 file
    # this loop replaces the node numbers in the topology array one by one
    wrongNumberTopology = wrongNumberViz['Mesh/0/mesh/topology'][:]

    for row in range(wrongNumberTopology.shape[0]):
        for column in range(wrongNumberTopology.shape[1]):
            wrongNumberTopology[row,column] = np.rint(ordered_map_index_wrongNumberNodes[wrongNumberTopology[row,column]])

    wrongNumberViz['Mesh/0/mesh/topology'][...] = wrongNumberTopology
    correctNumberViz['Mesh/0/mesh/topology'][...] = wrongNumberTopology
    
    # close the files
    wrongNumberViz.close()
    correctNumberViz.close()


if __name__ == '__main__':
    args = parse_args()
    folder = args.folder
    correct = args.correct
    wrong = args.wrong
    main(folder, correct, wrong)