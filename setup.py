import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name='spacejam',
        version='0.0.3',
        author='Ian Weaver, Sherif Gerges, Lauren Yoo',
        author_email='iweaver@cfa.harvard.edu',
        license='MIT',
        description='for automatic differentiation and other looney things',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/cs207-SpaceJam/cs207-FinalProject',
        packages=setuptools.find_packages(),
        install_requires=['atomicwrites==1.2.1',
                          'attrs==18.2.0',
                          'coverage==4.5.2',
                          'more-itertools==4.3.0',
                          'numpy==1.15.4',
                          'pluggy==0.8.0',
                          'py==1.7.0',
                          'pytest==4.0.1',
                          'pytest-cov==2.6.0',
                          'six==1.11.0'
        ],
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
        python_requires='~=3.3',
)
