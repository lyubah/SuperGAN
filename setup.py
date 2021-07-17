import subprocess
import sys

def install(package: str) -> None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

if __name__ == '__main__':
    with open('package_list.txt') as packages:
        for package in packages.readlines():
            install(package)
