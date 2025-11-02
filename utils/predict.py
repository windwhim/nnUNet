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
        "i": "/root/autodl-tmp/data/_raw/Dataset201_FrameFrame/imagesTs",
        "o": "/root/autodl-tmp/predict/FrameFrame",
        "tr": "nnUNetTrainerFrame",
    },
    "002": {
        "d": "002",
        "i": "/root/autodl-tmp/data/_raw/Dataset202_FrameLens/imagesTs",
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

    # 执行命令
    result = subprocess.run(command, capture_output=True, text=True)

    # 输出结果
    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


if __name__ == "__main__":
    predict("102")
