import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "1.0.0"

REPO_NAME = "Kidney-Disease-Classification-System"
AUTHOR_USER_NAME = "saksham-1304"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "sakshamsinghrathore1304@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Kidney Disease Classification from CT scans using VGG16 with k-fold CV, DVC, and MLflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)