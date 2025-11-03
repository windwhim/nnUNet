# 执行命令
import subprocess
import time

command = [
    "nUNetv2_train",
    "-i",
    "102",
    "-c",
    "2d" "-f",
    "0",
    "-tr",
    "nnUNetTrainerFrame",
]

time.sleep(3600)
result = subprocess.run(command, capture_output=True, text=True)

# 输出结果
print("Return code:", result.returncode)
print("Output:", result.stdout)
if result.stderr:
    print("Error:", result.stderr)