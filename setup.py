from setuptools import setup, find_packages

setup(
    name="marlgrid",
    version="0.0.5",
    packages=find_packages(),
    install_requires=["numpy", "tqdm", "gym==0.23.1", "gym-minigrid", "numba"],
)
