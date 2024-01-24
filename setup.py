from setuptools import setup, find_packages

setup(
    name="stencilNet",
    version="1.0.0",
    url="https://github.com/jpalafou/stencilNet",
    author="Jonathan Palafoutas",
    author_email="jpalafou@princeton.edu",
    description="Differentiable finite volume solver written in Jax",
    packages=find_packages(),
    install_requires=["jax", "numpy"],
)
