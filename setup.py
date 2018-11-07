import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name='spacejam',
        version='1.0',
        packages=setuptools.find_packages()
)
