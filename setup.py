from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    long_description = "\n" + f.read()

setup(
    name="tfdata-image-loader",
    version="1.0",
    description="A micro module for loading images for image classification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sebastian-sz/tfdata-image-loader",
    author="Sebastian Szymanski",
    author_email="mocart15@gmail.com",
    license="MIT",
    python_requires=">=3.6.0,<3.10",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    packages=["tfdata_image_loader"],
)
