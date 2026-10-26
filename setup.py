from setuptools import setup, find_packages

setup(
    name="konnon-drawing-model",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "torch",
        "pillow",
        "matplotlib",
        "pyyaml",
        "psutil",
    ],
)
