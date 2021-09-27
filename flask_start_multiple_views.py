
from flask import Flask, app, render_template, request, flash
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
from step_one import plot_step_one
from step_two import step_two
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from werkzeug.utils import secure_filename
from compute import compute

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
ALLOWED_FILES = {'csv'}
TEST_FOLDER = os.path.join(path, 'testfiles/plt2/datasets')

app = Flask(__name__)


if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["ALLOWED_FILES"] = ALLOWED_FILES
app.config["SECRET_KEY"] = 'f13783376341d2dddfaef175'
app.config['TEST_FOLDER'] = TEST_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_FILES


@app.route("/", methods=['POST', 'GET'])
def choose_form(test1=False, test2 = False):
    if request.method == 'POST' or (test1 is True or test2 is True):
        if request.form.get('submit', False) == 'submit_plt1':
            input = request.files['inputs']

            if input.filename == '':
                flash("input file is necessary!", category='warning')
                return render_template('tabs_visualizations.html')

            if  not allowed_file(input.filename):
                flash("input file need .csv extension!", category='danger')
                return render_template('tabs_visualizations.html')
            else:
                try:
                    data = (pd.read_csv(input, sep=';', squeeze=True))
                    sim = data['similarities'].to_numpy()
                    metrics = data['metrics'].to_numpy()
                    
                    if sim.size != metrics.size:
                        flash("similarities and metrics must have the same number of elements", category='danger')
                        return render_template('tabs_visualizations.html')

                    if np.isnan(np.sum(sim)) or np.isnan(np.sum(metrics)): 
                        flash("similarities and metrics must have any missing values", category='danger')
                        return render_template('tabs_visualizations.html')
                except:
                    flash("Unable to read input data, incorrect data format in csv files, please check the data and try again", category='danger')
                    return render_template('tabs_visualizations.html')
               
            
            img_step1 = plot_step_one(sim, metrics, scatter=True)

            return render_template('results_plt1.html', img_url = img_step1)

        if request.form.get('submit', False) == 'submit_plt2':
            
            sims = request.files['sims']
            datasets = request.files.getlist('datasets[]')
            offsets_auc = request.files['offsets_auc']
            offsets_nb = request.files['offsets_nb']
            offsets_brier = request.files['offsets_brier']

            if (sims.filename == ''):
                flash("file with * are necessary, please insert all mandatory files!", category='danger')
                return render_template('tabs_visualizations.html')

            if (not allowed_file(sims.filename)):
                flash("mandatory files need .csv extension!", category='danger')
                return render_template('tabs_visualizations.html')

            if ((offsets_auc.filename != '' and not allowed_file(offsets_auc.filename)) or (offsets_nb.filename != '' and not allowed_file(offsets_nb.filename))
                or (offsets_brier.filename != '' and not allowed_file(offsets_brier.filename))):
                flash("if inserted, offsets files need .csv extension!",category='danger')
                return render_template('tabs_visualizations.html')

            print(sims)
            try:
                sims = (pd.read_csv(sims, sep=';', usecols=['similarities'],header=0, squeeze=True)).to_numpy()
                # inizializzazione array per passagio dati a compute
                labels = np.array([])
                texts = np.array([]) 
                aucs = np.array([])
                nbs = np.array([])
                briers = np.array([])
                instances = np.array([])
                samples_auc = np.array([])
                samples_nb = np.array([])
                samples_brier = np.array([])
                var_auc = np.array([])
                var_nb = np.array([])
                var_brier = np.array([])
                
                if offsets_auc.filename != '':
                    offsets_auc = pd.read_csv(offsets_auc, sep=';',header=0, squeeze=True)
                    offsets_x_auc = offsets_auc['offset_x'].fillna(0.1).to_numpy()
                    offsets_y_auc = offsets_auc['offset_y'].fillna(0.1).to_numpy()
                else:
                    offsets_x_auc = default_offsets(sims)
                    offsets_y_auc = default_offsets(sims)

                if offsets_nb.filename != '':
                    offsets_nb = pd.read_csv(offsets_nb, sep=';',header=0, squeeze=True)
                    offsets_x_nb = offsets_nb['offset_x'].fillna(0.1).to_numpy()
                    offsets_y_nb = offsets_nb['offset_y'].fillna(0.1).to_numpy()
                else:
                    offsets_x_nb = default_offsets(sims)
                    offsets_y_nb = default_offsets(sims)

                if offsets_brier.filename != '':
                    offsets_brier = pd.read_csv(offsets_brier, sep=';',header=0, squeeze=True)
                    offsets_x_brier = offsets_brier['offset_x'].fillna(0.1).to_numpy()
                    offsets_y_brier = offsets_brier['offset_y'].fillna(0.1).to_numpy()
                else:
                    offsets_x_brier = default_offsets(sims)
                    offsets_y_brier = default_offsets(sims)
 
                if(sims.size != offsets_x_auc.size or sims.size != offsets_x_nb.size or sims.size != offsets_x_brier.size):
                    flash("offsets_x and dataset files must have the same number of elements", category='danger')
                    return render_template('tabs_visualizations.html')

                if(sims.size != offsets_y_auc.size or sims.size != offsets_y_nb.size or sims.size != offsets_y_brier.size):
                    flash("offsets_y and dataset files must have the same number of elements", category='danger')
                    return render_template('tabs_visualizations.html')

                if np.isnan(np.sum(sims)):
                    flash("similarities must not have any missing values!", category='danger')
                    return render_template('tabs_visualizations.html')

                
                for dataset in datasets:
                    if dataset and not allowed_file(dataset.filename):
                        flash("dataset files need .csv extension!", category='danger')
                        return render_template('tabs_visualizations.html')     
               
               
                for dataset in datasets:
                    filename = secure_filename(dataset.filename)
                    print(dataset, filename)
                    singleDataset = pd.read_csv(dataset, sep=';', squeeze=True)
                    y_test = singleDataset['y_true'].to_numpy()
                    y_proba = singleDataset['y_proba'].to_numpy()
        
                    if np.isnan(np.sum(y_test)) or np.isnan(np.sum(y_proba)):
                        flash("y_test and y_proba must have the same number of elements", category='danger')
                        return render_template('tabs_visualizations.html')

                    if check_if_0_or_1(y_test):
                        flash("y_test must have only 0 or 1 as elements", category='danger')
                        return render_template('tabs_visualizations.html')

                    if not check_if_between_0_and_1(y_proba):
                        flash("y_proba must have only numbers between 0 and 1 as elements", category='danger')
                        return render_template('tabs_visualizations.html')

                    labels = np.append(labels, filename[:-4])   
                    texts= np.append(texts, filename[:2])
                    # chiamata compute 
                    results = compute(y_test, y_proba)
                    
                    aucs= np.append(aucs, results[0])
                    nbs = np.append(nbs, results[1])
                    briers = np.append(briers, results[2])
                    instances = np.append(instances, results[3])
                    samples_auc = np.append(samples_auc, results[4])
                    samples_nb = np.append(samples_nb, results[5])
                    samples_brier = np.append(samples_brier, results[6])
                    var_auc = np.append(var_auc, results[7])
                    var_nb = np.append(var_nb, results[8])
                    var_brier = np.append(var_brier, results[9])
                    

                for x in range(len(labels)):
                    labels[x] = labels[x] + " (" + texts[x] + ")"
            
                if sims.size != labels.size:
                    flash("the number of elements in similarities must be equal to the number of dataset files inserted ", category='danger')
                    return render_template('tabs_visualizations.html')

               
            except:
                flash("Unable to read input data, incorrect data format in csv files, please check the data and try again", category='danger')
                return render_template('tabs_visualizations.html')

            img_step2 = step_two(sims, aucs, nbs, briers, labels, texts, instances, samples_auc, samples_nb, samples_brier,
                                 offsets_x_auc, offsets_y_auc, offsets_x_nb, offsets_y_nb, offsets_x_brier, offsets_y_brier, var_auc, var_nb, var_brier)

            return render_template('results_plt2.html', img_url=img_step2)

        if test1 is True:
            data = (pd.read_csv("./testfiles/plt1/inputsplt1.csv", sep=';', squeeze=True))
            sim = data['similarities'].to_numpy()
            metrics = data['metrics'].to_numpy()
            img_step1 = plot_step_one(sim, metrics, scatter=True)
            return render_template('results_plt1.html', img_url = img_step1)
        
        if test2 is True:
            aucs = np.array([])
            nbs = np.array([])
            briers = np.array([])
            instances = np.array([])
            samples_auc = np.array([])
            samples_nb = np.array([])
            samples_brier = np.array([])
            var_auc = np.array([])
            var_nb = np.array([])
            var_brier = np.array([])
            labels = ['Bergamo', 'Brazil_0', 'Brazil_1', 'Brazil_2', 'Desio', 'Ethiopia', 'HSR Nov', 'Spain']
            texts = ['Be', 'B0', 'B1', 'B2', 'De', 'Et', 'HS', 'Sp']
            sims = (pd.read_csv("./testfiles/plt2/sims.csv", sep=';',usecols=['similarities'],squeeze=True)).to_numpy()
            files = ['Bergamo.csv', 'Brazil_0.csv', 'Brazil_1.csv', 'Brazil_2.csv', 'Desio.csv', 'Ethiopia.csv', 'HSR_Nov.csv', 'Spain.csv']
            for x in range(len(files)):
                data = (pd.read_csv(os.path.join(app.config['TEST_FOLDER'], files[x]), sep=';', squeeze=True))
                y_test = data['y_true'].to_numpy()
                y_proba = data['y_proba'].to_numpy()

                results = compute(y_test, y_proba)

                aucs= np.append(aucs, results[0])
                nbs = np.append(nbs, results[1])
                briers = np.append(briers, results[2])
                instances = np.append(instances, results[3])
                samples_auc = np.append(samples_auc, results[4])
                samples_nb = np.append(samples_nb, results[5])
                samples_brier = np.append(samples_brier, results[6])
                var_auc = np.append(var_auc, results[7])
                var_nb = np.append(var_nb, results[8])
                var_brier = np.append(var_brier, results[9])
                

            offsets_x_auc = default_offsets(sims)    
            offsets_y_auc = default_offsets(sims) 
            offsets_x_nb = default_offsets(sims) 
            offsets_y_nb = default_offsets(sims) 
            offsets_x_brier = default_offsets(sims) 
            offsets_y_brier = default_offsets(sims) 

            for x in range(len(labels)):
                labels[x] = labels[x] + " (" + texts[x] + ")"
                
            
            img_step2 = step_two(sims, aucs, nbs, briers, labels, texts, instances, samples_auc, samples_nb, samples_brier,
                                 offsets_x_auc, offsets_y_auc, offsets_x_nb, offsets_y_nb, offsets_x_brier, offsets_y_brier, var_auc, var_nb, var_brier)    
                                 
            return render_template('results_plt2.html', img_url=img_step2)                     

    else:
        return render_template('tabs_visualizations.html')



@app.route('/test/plt1')
def testplt1():
    return choose_form(test1 = True)


@app.route('/test/plt2')
def testplt2():
    return choose_form(test2 = True)    


def check_if_between_0_and_1(arr):
    is_between = True       
    for elem in arr:
        if elem < 0 or elem > 1:
            return False
    return is_between


def check_if_0_or_1(arr):
    is_0_or_1 = True
    for elem in arr:
        if elem != 0 or elem != 1:
            return False
    return is_0_or_1


def check_if_contains_zero(arr):

    is_zero = False
    for elem in arr:
        if elem == 0:
            return True
    return is_zero


def check_if_positive(arr):
    is_negative = False
    for elem in arr:
        if elem < 0:
            return True
    return is_negative


def default_offsets(arr):
    offset = np.empty(len(arr))
    offset.fill(.01)
    return offset


if __name__ == '__main__':
    app.run(debug=True)
