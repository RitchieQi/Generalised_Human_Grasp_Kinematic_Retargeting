from setuptools import setup

setup(
    name='genhand_v1',
    version='0.1',
    packages=['dataset', 'network', 'optimisation', 'simulation'],
    package_dir={
        "": ".",
        "networks": "./networks",
        "optimisation": "./optimisation",
        "dataset": "./dataset",
        "simulation": "./simulation"
    },
    install_requires=[],
)