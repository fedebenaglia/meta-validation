#  Meta-validation Methodology

**Version 1.0.0**


### Table of Contents
You're sections headers will be used to reference location of destination.

- [Description](#description)
- [Prerequisites](#prerequisites)
- [How To Use](#how-to-use)
- [Contributing](#Contributing-on-Heroku)
- [References](#references)

---

## Description

This is a  Python/Flask tool to generate two data visualizations: the Potential Robustness Diagram relates a set of similarities with the corresponding accuracy score; the External Performance Diagram displays the results of three performance analyses (discriminative, utility, calibration) as a function of the similarity between training/validation and external validation sets(see down below for resources).

You can test the tool directly from Heroku clicking on the following link
- [Meta-validation Methodology](https://prova-meta-validation.herokuapp.com/)

## Prerequisites

Make sure you have installed all of the following prerequisites on your development machine:

- Python 3.9
- pip
- Git

## How To Use

### Setting up a development environment

- Open the Terminal from youre code editor,move to project folder and clone the project from gitHub 

    ```
    # Clone the code repository into ~/dev/PROJECT_FOLDER
    cd ~/dev
    git clone GITHUB_REPO APP_NAME
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

### Running the tool 

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
    
## Contributing on Heroku

If you are a contributors you can collaborate with other developers, before you share the project you
need to download and install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)

- acces with your heroku account typing `heroku login`

    ```
    heroku: Press any key to open up the browser to login or q to exit: 
    ```
- If not installed you need to type `pip install gunicorn` to push your code on Heroku
- Use the Git commands to inizialize, add and commit your changes with:
    `git init`
    `git add .`
    `git commit -m "COMMIT_MESSAGE"`
- Add a remote to your local repository with the heroku command `heroku git:remote -a prova-meta-validation` 
- Deploy on Heroku with `git push heroku master`, this operation it can last a few minutes
    
## References

For more details on this score please refer to:
Cabitza, F., Campagner, A., Soares, F., de Guardiana Romualdo, L. G., Challa, F., Sulejmani, A.,
Seghezzi, M., Carobene,A. (2021). The importance of being external. methodological insights for 
the external validation of machine learning models in medicine. Computer Methods and Programs in Biomedicine.
Volume 208, September 2021, 106288 (10.1016/j.cmpb.2021.106288)
For more datails go to https://doi.org/10.1016/j.cmpb.2021.106288
    
    


