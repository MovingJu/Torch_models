import subprocess

import modules

def main():

    file_path = "/app/models/running.txt"

    with open(file_path, "w") as file:
        file.write("Docker container is still running!")

    modules.Face_detection.main()

    subprocess.run([
        "rm", file_path
    ])

    return


if __name__ == "__main__":
    main()
