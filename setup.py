from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MegmentationNetwork",  
    version="0.1.0",  
    author="Pecako2001", 
    author_email="koenvanwijlick@example.com",  
    description="A deep learning-based segmentation network for polygonal annotations.",
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",
    url="https://github.com/Pecako2001/MegmentationNetwork", 
    packages=find_packages(),  # Automatically find packages in your repository
    install_requires=requirements,  # Use the requirements.txt file to install dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  
    entry_points={
        'console_scripts': [
            'megmentation=megmentation.train:main', 
            'megmentation-val=megmentation.val:main', 
        ],
    },
)
