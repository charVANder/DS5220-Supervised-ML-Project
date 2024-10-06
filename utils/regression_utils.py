import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math


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


def cv_scores_dict_to_cv_scores_df(cv_scores_dict):

    return_dict = cross_val_evaluation(cv_scores_dict)
    df_row_dict_list = return_dict['df_row_dict_list']

    cv_scores_analysis_df = pd.DataFrame(df_row_dict_list)

    print('\n', cv_scores_analysis_df.groupby(['regressor_name', 'score_name_', 'score_type']).mean().reset_index(),
          '\n')

    min_score = cv_scores_analysis_df.score.min()
    max_score = cv_scores_analysis_df.score.max()

    return_dict = {
        'cv_scores_analysis_df': cv_scores_analysis_df,
        'min_score': min_score,
        'max_score': max_score
    }

    return return_dict


def remove_neg_from_score_name_and_make_neg_score_positive(scores, score_name=None):
    """

    :param score_name:
    :param scores:
    :return:
    """

    # if scores are negative then change the sign to positive - they are negative because scikit-learn follows a
    # convention where higher scores are better in optimization
    neg_score_flag = False
    if (scores < 0).all():  # r2 sometimes returns small negative values
        neg_score_flag = True
        scores = -1 * scores

    # remove the words 'neg', 'train' and 'test' from score_name
    score_name = '_'.join([token for token in score_name.split('_') if 'neg' not in token])
    score_name = '_'.join([token for token in score_name.split('_') if 'train' not in token])
    score_name = '_'.join([token for token in score_name.split('_') if 'test' not in token])

    return_dict = {
        'scores': scores,
        'score_name': score_name,
        'neg_score_flag': neg_score_flag
    }

    return return_dict


def cross_val_evaluation(scores_dict):
    """
    Takes in a scores dict from a sklearn cross_validation() function and return a score_analysis_dict that can be
    used to analyze the cross validation.
    :param scores_dict:
        first key:value pair
            key = 'scoring', value = a list of scores (metrics) evaluated in sklearn cross_validate() function
        remaining key:value pair(s)
            key = estimator name, value = scores dictionary returned from sklearn cross_validate() function
    :return:
    """

    scoring_list = scores_dict['scoring']
    del scores_dict['scoring']

    max_score = -1 * np.inf
    min_score = np.inf
    df_row_dict_list = []
    for score_name in scoring_list:  # we evaluate a score_name across all the estimators in the survey

        for regressor_name, scoring_dict in scores_dict.items():  # iterate through the estimators

            for cv_score_name, scores in scoring_dict.items():  # iterate though an estimators cross_validate() scores

                score_type = 'test'
                if 'train' in cv_score_name:
                    score_type = 'train'

                if score_name in cv_score_name:  # once we iterate to the score_name we are working on do stuff

                    # get the list of scores
                    scores_list = scoring_dict[cv_score_name]

                    # make scores positive and remove 'neg' from score names if scores are negative
                    return_dict = remove_neg_from_score_name_and_make_neg_score_positive(scores_list, cv_score_name)
                    scores_list = return_dict['scores']
                    score_name_ = return_dict['score_name']

                    # get the min and max score from the scores_list
                    max_ = max(scores_list)
                    if max_ > max_score:
                        max_score = max_

                    min_ = min(scores_list)
                    if min_ < min_score:
                        min_score = min_

                    # save the scores list to cv_scores_analysis_dict
                    for score in scores_list:
                        df_row_dict_list.append(
                            {
                                'regressor_name': regressor_name,
                                'score_name_': score_name_,
                                'score': score,
                                'score_type': score_type
                            }
                        )

    return_dict = {
        'df_row_dict_list': df_row_dict_list,
        'min_score': min_score,
        'max_score': max_score
    }

    return return_dict


def cv_scores_analysis(score_analysis, splitter, target_attr=None, boxplot=True, catplot=True):

    if isinstance(score_analysis, pd.DataFrame):
        analysis_df = score_analysis
    else:
        analysis_df = pd.DataFrame(score_analysis)

    for score_name_ in analysis_df.score_name_.unique():

        # get all estimators with scoring = score_name_
        temp_df = analysis_df.loc[analysis_df.score_name_ == score_name_, :]

        # get the mean score for each estimator by score type (train and test)
        plot_df = (temp_df[['regressor_name', 'score', 'score_type']].groupby(['regressor_name', 'score_type']).
                   score.mean().reset_index())

        hue_order = ['train', 'test']

        if catplot:
            sns.catplot(data=plot_df, x='regressor_name', y='score', hue='score_type', s=100, hue_order=hue_order)
            plt.xticks(rotation=90)
            plt.title(f'means of {splitter.get_n_splits()}-fold cross validation scores\n{score_name_}')
            if target_attr:
                plt.ylabel(f'{target_attr}')
            plt.grid()
            plt.show()

        if boxplot:
            sns.boxplot(data=temp_df, x='regressor_name', y='score', hue='score_type', showmeans=True,
                        meanprops={'marker': 'x', 'markerfacecolor': 'black', 'markeredgecolor': 'black',
                                   'markersize': '6'}, hue_order=hue_order)
            plt.xticks(rotation=90)
            plt.title(f'boxplot of {splitter.get_n_splits()}-fold cross validation scores\n{score_name_}; '
                      f'x marker = mean')
            if target_attr is not None:
                plt.ylabel(f'{target_attr}')
            if score_name_ == 'r2':
                plt.ylabel(f'r2')
            plt.grid()
            plt.show()


def fix_negative_scores_and_name_utility(gs_cv_results_df, mean_train_score, mean_test_score, score):

    # make scores positive and remove 'neg' from score names if scores are negative
    return_dict = remove_neg_from_score_name_and_make_neg_score_positive(gs_cv_results_df[mean_train_score], score)
    gs_cv_results_df[mean_train_score] = return_dict['scores']
    score_ = return_dict['score_name']
    neg_score_flag = return_dict['neg_score_flag']

    mean_train_score_ = 'mean_train_' + score_

    return_dict = remove_neg_from_score_name_and_make_neg_score_positive(gs_cv_results_df[mean_test_score], score)
    gs_cv_results_df[mean_test_score] = return_dict['scores']
    score_ = return_dict['score_name']

    mean_test_score_ = 'mean_test_' + score_

    return_dict = {
        'gs_cv_results_df': gs_cv_results_df,
        'mean_train_score_': mean_train_score_,
        'mean_test_score_': mean_test_score_,
        'neg_score_flag': neg_score_flag
    }

    return return_dict


def prepare_data_for_flexibility_plot(gs_cv_results_df, mean_train_score, rank_test_score, mean_test_score,
                                      neg_score_flag, std_train_score, std_test_score):

    # determine proper sort
    if neg_score_flag:
        ascending = False  # smaller is better metric
    else:
        ascending = True  # bigger is better metric

    # sort by train score and label with index for plotting
    gs_cv_results_df = gs_cv_results_df.sort_values(mean_train_score, ascending=ascending).reset_index(drop=True). \
        reset_index()
    gs_cv_results_df = gs_cv_results_df[['index', rank_test_score, mean_train_score, std_train_score, mean_test_score,
                                         std_test_score]]

    return gs_cv_results_df


def flexibility_plot_util(gs_cv_results_df, mean_train_score, mean_train_score_, mean_test_score, mean_test_score_,
                          rank_test_score, an_estimator_name, score, std_train_score, std_test_score, error_bars=False):

    # plot train and test rmse
    jitter = 0
    sns.lineplot(x='index', y=mean_train_score, data=gs_cv_results_df, label=mean_train_score_, marker='o',
                 color='blue')
    if error_bars:
        for index in gs_cv_results_df.index:
            lower = gs_cv_results_df.loc[index, mean_train_score] - gs_cv_results_df.loc[index, std_train_score]
            upper = gs_cv_results_df.loc[index, mean_train_score] + gs_cv_results_df.loc[index, std_train_score]
            plt.plot([index-jitter, index-jitter], [lower, upper], color='blue')

    sns.lineplot(x='index', y=mean_test_score, data=gs_cv_results_df, label=mean_test_score_, marker='o', color='red')
    if error_bars:
        for index in gs_cv_results_df.index:
            lower = gs_cv_results_df.loc[index, mean_test_score] - gs_cv_results_df.loc[index, std_test_score]
            upper = gs_cv_results_df.loc[index, mean_test_score] + gs_cv_results_df.loc[index, std_test_score]
            plt.plot([index+jitter, index+jitter], [lower, upper], color='red')

    # draw a vertical line at the best index
    best_index = gs_cv_results_df.loc[gs_cv_results_df[rank_test_score] == 1, 'index'].values[0]
    plt.axvline(x=best_index, color='k', linestyle='--')

    # plot title and axis labels
    plt.title(f'{an_estimator_name} flexibility plot\nmin test error at best_index = {best_index}')
    plt.xlabel('flexibility')
    plt.ylabel('_'.join([token for token in score.split('_') if 'neg' not in token]))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # make index an integer on plot
    new_list = range(math.floor(min(gs_cv_results_df.index)), math.ceil(max(gs_cv_results_df.index)) + 1)
    if 10 <= len(new_list) < 100:
        skip = 10
    elif 100 <= len(new_list) < 1000:
        skip = 100
    else:
        skip = 500
    plt.xticks(np.arange(min(new_list), max(new_list) + 1, skip))
    plt.grid()
    plt.show()

    return best_index


def flexibility_plot_regr(gs_cv_results_df, an_estimator_name, score):

    # establish score names
    mean_train_score = 'mean_train_' + score
    std_train_score = 'std_train_' + score
    mean_test_score = 'mean_test_' + score
    std_test_score = 'std_test_' + score
    rank_test_score = 'rank_test_' + score

    # if scores are negative make scores positive and remove 'neg' from score name
    return_dict = fix_negative_scores_and_name_utility(gs_cv_results_df, mean_train_score, mean_test_score, score)
    gs_cv_results_df = return_dict['gs_cv_results_df']
    mean_train_score_ = return_dict['mean_train_score_']
    mean_test_score_ = return_dict['mean_test_score_']
    neg_score_flag = return_dict['neg_score_flag']

    # sort by train score and label with index for plotting
    gs_cv_results_df = prepare_data_for_flexibility_plot(gs_cv_results_df, mean_train_score, rank_test_score,
                                                         mean_test_score, neg_score_flag, std_train_score,
                                                         std_test_score)

    # make the plot
    best_index = flexibility_plot_util(gs_cv_results_df, mean_train_score, mean_train_score_, mean_test_score,
                                       mean_test_score_, rank_test_score, an_estimator_name, score, std_train_score,
                                                         std_test_score)

    return gs_cv_results_df, best_index


if __name__ == "__main__":
    pass
