import pickle
from image import load_dataset, plot_image, normaliseImg, plot_many, plot_scatter
# for model_results in []
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
"""['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 
'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted',
 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
"""
mpl.style.use('ggplot')
mpl.rcParams["savefig.format"] = 'svg'

def get_scatterplot_data(perturbation, data):
    x = []
    y = []
    for each in data:
        name, accuracy = each[0], each[1]
        perturb_name = name.split('\\')[1]
        perturb_intensity = name.split('\\')[-1]
        # print(perturb_name, perturb_intensity, accuracy)
        if perturb_name==perturbation:
            
            x.append(float(perturb_intensity))
            y.append(float(accuracy))
    
    if perturbation=='hsv_hue_noise_increase':
        x = [element / 179 for element in x]
    if perturbation=='hsv_sat_noise_increase':
        x = [element / 255 for element in x]
    if perturbation=='image_contrast_decrease':
        x = [-element for element in x]
    return x, y

def choose_test_folder(kfld):
    testing_folders = ['A', 'B', 'C']
    for folder in testing_folders:
        if folder not in kfld:
            return folder

def sort_lists(x, y_res, std_res, y_svm, std_svm):
    x = np.array(x)
    y_res = np.array(y_res)
    std_res = np.array(std_res)
    y_svm = np.array(y_svm)

    i = np.argsort(x)
    return x[i], y_res[i], std_res[i], y_svm[i], std_svm[i]

def plot_multiscatter(perturbation, x, y_res, std_res, y_svm, std_svm, x_label, y_label, title, save=False, standard=True, tight=True):
  
    x, y_res, std_res, y_svm, std_svm = sort_lists(x, y_res, std_res, y_svm, std_svm)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y_res, std_res, linestyle='dashed', marker='o', linewidth=1, markersize=2.5, label="ResNet18")
    ax.errorbar(x, y_svm, std_svm, linestyle='dashed', marker='o', linewidth=1, markersize=2.5, label="SVM+BoVW")
    if standard:
        ax.errorbar(np.linspace(min(x), max(x), num=10), [0.5]*10, linestyle='dashed', linewidth=1, label="Random Guessing")
        # plt.text(6.8, 0.48, 'Worse than random guessing', size=8)
        plt.yticks(np.arange(0.4, 1.05, step=0.05))
    # plt.locator_params(axis="x", nbins=10)
    if perturbation=='image_contrast_decrease':
        plt.xticks(x, [-element for element in x])
    else:
        plt.xticks(x, x)
    
    # plt.locator_params(axis="y", nbins=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='lower left')

    if tight:
        plt.tight_layout()

    if save:
        plt.savefig(f"result_plots"/{perturbation}, dpi=250)
    else:
        plt.show()

def make_nice_plot(perturbation, save, standard, tight):
    y_svms = []
    y_ress = []
    for kfld in ['AB', 'AC', 'BC']:

        svm_results_path = f'SVM_{kfld}_results_on_{choose_test_folder(kfld)}.pkl'
        resnet_results_path = f'ResNet18_{kfld}_results_on_{choose_test_folder(kfld)}.pkl'


        with open(svm_results_path, 'rb') as f:
            data_svm = pickle.load(f)
        
        with open(resnet_results_path, 'rb') as fl:
            data_res = pickle.load(fl)


        x_svm, y_svm = get_scatterplot_data(perturbation, data_svm)
        x_res, y_res = get_scatterplot_data(perturbation, data_res)


        y_svms.append(y_svm)
        y_ress.append(y_res)

    y_ress = np.array(y_ress)
    y_svms = np.array(y_svms)
    means_ress = np.mean(y_ress, axis=0)
    means_svms = np.mean(y_svms, axis=0)
    std_ress = np.std(y_ress, axis=0)
    std_svms = np.std(y_svms, axis=0)

    plot_title = f"Robustness Against {perturbation.replace('_', ' ').title()}"

    plot_multiscatter(perturbation, x_svm, means_ress, std_ress, means_svms, std_svms, x_label='Perturbation Level', y_label='Classification Accuracy', title=plot_title, save=save, standard=standard, tight=tight)


perturbations = ['gaussian_blurring', 'gaussian_pixel_noise', 'hsv_hue_noise_increase', 'hsv_sat_noise_increase', 'image_brightness_decrease', 'image_brightness_increase', 'image_contrast_decrease', 'image_contrast_increase', 'occlusion']
for perturbation in perturbations:

    make_nice_plot(perturbation, save=False, standard=True, tight=False)

