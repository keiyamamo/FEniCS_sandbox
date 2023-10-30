import subprocess
import re

cmd = ("turtleFSI -p turtle_demo -dt 0.01 -T 0.01 --verbose True" +
           " --theta 0.51 --folder tmp --sub-folder 1")
result = subprocess.check_output(cmd, shell=True)

target_time_step = 1

output_regular_expression = (
r"Solved for timestep {}, t = (\d+\.\d+) in (\d+\.\d+) s").format(target_time_step)
output_match = re.search(output_regular_expression, str(result))

from IPython import embed; embed(); exit(1)