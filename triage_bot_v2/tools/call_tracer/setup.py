from setuptools import setup, find_packages

setup(
    name="call_tracer",
    version="0.1.0",
    packages=find_packages(include=["src.*",]),
    package_dir={"": "src"},
)
