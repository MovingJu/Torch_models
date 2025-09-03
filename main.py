import subprocess

import modules

def main():

    modules.Face_detection.main()

    subprocess.run(["git", "add", "./models/*"])
    subprocess.run(["git", "commit", "-m", "Add model"])
    subprocess.run(["git", "push", "origin", "main"])

    subprocess.run(["sudo", "poweroff"])

    return


if __name__ == "__main__":
    main()
