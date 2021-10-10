from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "REPO_NAME"
AUTHOR_USER_NAME = "c17hawke"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small package for DVC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="sunny.c17hawke@gmail.com",
    packages=[SRC_REPO],
    license="GNU",
    python_requires=">=3.6",
    install_requires=LIST_OF_REQUIREMENTS
)