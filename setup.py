from setuptools import setup


def requirements(filename="requirements.txt"):
    with open(filename) as f:
        return f.readlines()


setup(
    install_requires=requirements("requirements.txt"),
    license="MIT License",
)
