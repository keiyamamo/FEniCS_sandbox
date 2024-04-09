import subprocess   

cmd = "mpirun -np 2 turtleFSI -dt 0.1"
results = subprocess.check_output(cmd, shell=True)