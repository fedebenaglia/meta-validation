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
- Git

## How To Use

### Setting up a development environment

- Open the Terminal from youre code editor,move to project folder and clone the project from gitHub 
    ```
    # Clone the code repository into ~/dev/my_app
    cd ~/dev
    git clone GITHUB_REPO my_app
    ```
- Create the virual environment and activate it(the current code line works for windows)
    ```    
    # Create the virtual environment
    python -m venv MY_NAME_ENV

    # Activate the env
    MY_NAME_ENV\Scripts\activate.bat
    ```
- Let's install all the Libraries required to run the tool
    ```
    # Install required Python packages
    pip install -r requirements.txt
    ```
    Make sure all dependencies are installed on your env!

### Running the tool on LocalHost

- Make sure that env is active and type `python flask_start_multiple_views.py`

    ```
    * Serving Flask app 'flask_start_multiple_views' (lazy loading)
     * Environment: production
       WARNING: This is a development server. Do not use it in a production deployment.
       Use a production WSGI server instead.
     * Debug mode: on
     * Restarting with stat
     * Debugger is active!
     * Debugger PIN: PIN
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```

    Click on it or copy and paste the address into the browser to see the project running
    
    Now you can use the tool!

