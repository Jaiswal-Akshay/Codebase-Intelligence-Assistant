from setuptools import find_packages, setup


setup(
    name="codebase-compass",
    version="0.1.0",
    description="A retrieval-augmented assistant for understanding software repositories",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "fastembed>=0.8.0",
        "numpy>=1.26",
    ],
    entry_points={
        "console_scripts": [
            "codebase-compass=codebase_assistant.cli:main",
        ]
    },
)
