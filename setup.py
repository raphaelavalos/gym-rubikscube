import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gym-rubikscube",
    version="0.0.1",
    author="R. Avalos",
    author_email="raphael@avalos.fr",
    description="Gym environment for the Rubik's Cube",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelavalos/gym-rubikscube",
    packages=['gym_rubikscube'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)