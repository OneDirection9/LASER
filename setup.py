from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup


def get_readme():
    with open("README.md", "r") as f:
        content = f.read()
    return content


install_requires = [
    "transliterate",
    "jieba",
    "fairseq",  # fairseq has the requirement for torch
]

extras_require = {
    "dev": [
        "flake8==3.8.4",
        "isort==5.6.4",
        "black==20.8b1",
        "flake8-bugbear",
        "flake8-comprehensions",
    ],
}

# CFLAGS="-stdlib=libc++" python setup.py develop
setup(
    name="fairseq-laser",
    version="0.0.1",
    description="LASER build upon fairseq",
    long_description=get_readme(),
    keywords="computer vision",
    packages=find_packages(include=("laser.*",)),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Utilities",
    ],
    author="Zhipeng Han",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=install_requires,
    extras_require=extras_require,
)
