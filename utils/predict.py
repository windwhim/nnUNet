import subprocess

command_map = {
    "101": {
        "d": "101",
        "i": "/root/autodl-tmp/data/_raw/Dataset101_NoFrameFrame/imagesTs",
        "o": "/root/autodl-tmp/predict/NoFrameFrame",
        "tr": "nnUNetTrainerNoFrame",
    },
    "102": {
        "d": "102",
        "i": "/root/autodl-tmp/data/_raw/Dataset102_NoFrameLens/imagesTs",
        "o": "/root/autodl-tmp/predict/NoFrameLens",
        "tr": "nnUNetTrainerNoFrame",
    },
    "001": {
        "d": "001",
        "i": "/root/autodl-tmp/data/_raw/Dataset001_FrameFrame/imagesTs",
        "o": "/root/autodl-tmp/predict/FrameFrame",
        "tr": "nnUNetTrainerFrame",
    },
    "002": {
        "d": "002",
        "i": "/root/autodl-tmp/data/_raw/Dataset002_FrameLens/imagesTs",
        "o": "/root/autodl-tmp/predict/FrameLens",
        "tr": "nnUNetTrainerFrame",
    },
}


def predict(index: str):
    command = [
        "nnUNetv2_predict",
        "-i",
        command_map[index]["i"],
        "-o",
        command_map[index]["o"],
        "-d",
        command_map[index]["d"],
        "-c",
        "2d",
        "-f",
        "0",
        "-tr",
        command_map[index]["tr"],
    ]

    # 定义命令及其参数

    # # 执行命令
    # result = subprocess.run(command, capture_output=True, text=True)

    # # 输出结果
    # print("Return code:", result.returncode)
    # print("Output:", result.stdout)
    # if result.stderr:
    #     print("Error:", result.stderr)

    # 执行命令并实时显示输出
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # 逐行读取输出
    for line in process.stdout:
        print(line, end="")  # 实时打印每一行输出

    # 等待命令执行完成
    process.wait()

    # 输出返回码
    print("\nReturn code:", process.returncode)


if __name__ == "__main__":
    predict("001")
