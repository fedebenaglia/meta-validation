#  Meta-validation Methodology

**Version 1.0.0**

---

### Table of Contents
You're sections headers will be used to reference location of destination.

- [Description](#description)
- [Prerequisites](#prerequisites)
- [How To Use](#how-to-use)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

This is a  Python/Flask tool to generate two data visualizations: the Potential Robustness Diagram relates a set of similarities with the corresponding accuracy score; the External Performance Diagram displays the results of three performance analyses (discriminative, utility, calibration) as a function of the similarity between training/validation and external validation sets(see down below for resources).

You can test the tool directly from Heroku clicking on the following link
- [Meta-validation Methodology](https://prova-meta-validation.herokuapp.com/)

## Prerequisites

Make sure you have installed all of the following prerequisites on your development machine:

- Python 3.9, [click here for the download](https://prova-meta-validation.herokuapp.com/)
- pip

## How To Use

#### Setting up a development environment

- Open the Terminal and move on the cwd 
    ```
    # Clone the code repository into ~/dev/my_app
    mkdir -p ~/dev
    cd ~/dev
    git clone https://github.com/lingthio/Flask-User-starter-app.git my_app

    # Create the virtual environment
    mkvirtualenv -p PATH/TO/PYTHON env

    # Install required Python packages
    cd ~/dev/my_app
    workon my_app
    pip install -r requirements.txt
    ```




