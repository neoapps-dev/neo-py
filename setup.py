from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neopy-utils",
    version="1.0.0",
    author="NEOAPPS",
    author_email="asd22.info@gmail.com",
    description="A powerful Python utility library with retry mechanisms, concurrency helpers, and more",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neoapps-dev/neo-py",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.7",
    keywords="retry, utilities, concurrency, memoize, rate limiting, validation",
    project_urls={
        "Bug Tracker": "https://github.com/neoapps-dev/neo-py/issues",
        "Documentation": "https://github.com/neoapps-dev/neo-py",
        "Source Code": "https://github.com/neoapps-dev/neo-py",
    },
    license='MIT',
    license_file='LICENSE',
)