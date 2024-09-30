import matplotlib.pyplot as plt
import seaborn as sns
# import sys
import numpy as np
import pandas as pd


def plot_pred_vs_actual_survey(trained_estimator_dict, cap_x_df, y_df, data_set_name):
    """

    :param trained_estimator_dict:
    :param cap_x_df:
    :param y_df:
    :param data_set_name: data set name (train, validation or test)
    :return:
    """

    for estimator_name, estimator in trained_estimator_dict.items():
        pred_y_df = estimator.predict(cap_x_df)
        plot_title = f'estimator_name: {estimator_name}; data_set_name: {data_set_name}'
        plot_pred_vs_actual(pred_y_df, y_df, plot_title)


def plot_pred_vs_actual(pred_y_df, train_y_df, plot_title):
    """

    :param pred_y_df:
    :param train_y_df:
    :param plot_title: include estimator name and data set name (train, validation or test)
    :return:
    """
    plt.scatter(train_y_df, pred_y_df)  # plot predicted y vs true y
    plt.plot(train_y_df, train_y_df, 'b')  # plot a line of slope 1 demonstrating perfect predictions
    plt.grid()
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title(plot_title)
    plt.show()


def cv_scores_dict_to_cv_scores_df(cv_scores_dict, print_df=True):

    return_dict = cross_val_evaluation(cv_scores_dict)
    scores_analysis_dict = return_dict['cv_scores_analysis_dict']

    cv_scores_analysis_df = pd.DataFrame(scores_analysis_dict)

    min_score = cv_scores_analysis_df.min().min()
    max_score = cv_scores_analysis_df.max().max()

    if print_df:
        print('\n', cv_scores_analysis_df.mean(axis=0).to_frame().rename(columns={0: 'mean_metric_value'}))

    return_dict = {
        'cv_scores_analysis_df': cv_scores_analysis_df,
        'min_score': min_score,
        'max_score': max_score
    }

    return return_dict


def cross_val_evaluation(scores_dict):
    """
    Takes in a scores dict from a sklearn cross_validation() function and return a score_analysis_dict that can be
    used to analyze the cross validation.
    :param scores_dict:
    :return:
    """

    scoring_list = scores_dict['scoring']
    del scores_dict['scoring']

    max_score = -1 * np.inf
    min_score = np.inf
    cv_scores_analysis_dict = {}
    for metric in scoring_list:
        for regressor_name, scoring_dict in scores_dict.items():
            for key, scores in scoring_dict.items():
                if metric in key:
                    scores_list = scoring_dict[key]
                    metric_ = ''
                    if 'neg' in key:
                        scores_list = -1 * scores_list
                        metric_ = ''.join([metric_ + string[0] for string in metric.split('_')[1:]])
                    else:
                        metric_ = ''.join([metric_ + string[0] for string in metric.split('_')])
                    max_ = max(scores_list)
                    if max_ > max_score:
                        max_score = max_
                    min_ = min(scores_list)
                    if min_ < min_score:
                        min_score = min_
                    cv_scores_analysis_dict[regressor_name + '_' + metric_] = scores_list

    return_dict = {
        'cv_scores_analysis_dict': cv_scores_analysis_dict,
        'min_score': min_score,
        'max_score': max_score
    }

    return return_dict


def cv_scores_analysis(score_analysis, min_score, max_score, target_attr=None, histograms=False, boxplot=True,
                       catplot=True):

    if isinstance(score_analysis, pd.DataFrame):
        analysis_df = score_analysis
    else:
        analysis_df = pd.DataFrame(score_analysis)

    if histograms:
        for attr in analysis_df.columns:
            sns.histplot(data=analysis_df, x=attr)
            plt.xlim([min_score, max_score])
            plt.grid()
            plt.title(f'{attr}\nmean: {analysis_df[attr].mean():.3f}; stdev: {analysis_df[attr].std():.3f}')
            plt.show()

    if catplot:
        sns.catplot(data=analysis_df.mean(axis=0).to_frame().T, s=100)
        plt.xticks(rotation=90)
        plt.title(f'means of cross validation approaches')
        if target_attr:
            plt.ylabel(target_attr)
        plt.grid()
        plt.show()

    if boxplot:
        sns.boxplot(data=analysis_df, showmeans=True, meanprops={'marker': 'x', 'markerfacecolor': 'black',
                                                                 'markeredgecolor': 'black', 'markersize': '6'})
        plt.xticks(rotation=90)
        plt.title(f'boxplot of cross validation approaches - black x is the mean')
        if target_attr:
            plt.ylabel(target_attr)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    pass
