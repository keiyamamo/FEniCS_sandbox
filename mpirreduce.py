from mpi4py import MPI
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
data = rank
data = comm.reduce(data, op=MPI.SUM, root=0)

print("On rank", rank, "data=", data)