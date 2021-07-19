"""
Sets up the virtual environment.
"""

import subprocess
import sys


def install(package_name: str) -> None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])


if __name__ == '__main__':
    with open('package_list.txt') as packages:
        for package in packages.readlines():
            install(package)
