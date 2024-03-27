# -*- coding: utf-8 -*-

# File: setup.py
# License: MIT License
# Copyright: (c) 2023 Jungheil <jungheilai@gmail.com>
# Created: 2023-11-03
# Brief:
# --------------------------------------------------

import shutil
from distutils.core import setup

from Cython.Build import cythonize

shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)

setup(
    name="hello_badminton",
    version="2.3.0",
    description="A badminton reservation tool",
    author="Jungheil",
    author_email="jungheilai@gmail.com",
    ext_modules=cythonize(
        module_list=["hello_badminton/*.py", "hello_badminton/utils/*.py"],
        build_dir="build",
    ),
    license="MIT",
    script_args=["build"],
    options={"build": {"build_lib": "dist"}},
)

shutil.copy("main.py", "dist/main.py")
shutil.copy("config.dist.json", "dist/config.json")
shutil.copy("requirements.txt", "dist/requirements.txt")
shutil.copy("README.dist.md", "dist/README.md")
shutil.copytree("resources", "dist/resources")
