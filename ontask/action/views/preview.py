# -*- coding: utf-8 -*-

"""Views to preview resulting text in the action."""
from typing import Optional

from django import http
from django.contrib.auth.decorators import user_passes_test
from django.template.loader import render_to_string
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt

from ontask import models
from ontask.action import services
# from ontask.action.views.cohmetrixBR import run_coh_metrix
from ontask.core import ajax_required, get_action, is_instructor

import json
import urllib.parse as parse
import urllib.request as request

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score  # , mean_squared_error
import os
import xgboost as xgb

from nltk.tokenize import word_tokenize
import string, re
import spacy
import nltk

nltk.download('punkt')
nlp = spacy.load('pt')

classifiers = []
for i in range(11):
    classifiers.append(xgb.XGBClassifier(n_estimators=500, use_label_encoder=False))


def read_classes(path):
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    print(CURR_DIR)
    data = pd.read_csv(CURR_DIR + '/' + path)
    id_name = data.keys()[0]
    vector = []
    print(id_name)
    # Remove id col
    del data[id_name]
    # 11 -> classes number
    for i in range(11):
        vector.append(data[data.keys()[i]].values.tolist())
    return vector


def read_data(csv):
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    print(CURR_DIR)
    data = pd.read_csv(CURR_DIR + '/' + csv)
    # id col name
    id_name = data.keys()[0]
    new_data = data.copy()

    # Remove id col
    # del new_data[id_name]
    return data, new_data.values.tolist()


def search(lista, valor):
    return [lista.index(x) for x in lista if valor in x]


def extract_liwc(text):
    # reading liwc
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    print(CURR_DIR)
    wn = open(CURR_DIR + '/' + 'LIWC2007_Portugues_win.dic.txt', 'r', encoding='utf-8', errors='ignore').read().split(
        '\n')
    word_set_liwc = []
    for line in wn:
        words = line.split('\t')
        if words != []:
            word_set_liwc.append(words)

    # indexes of liwc
    indices = open(CURR_DIR + '/' + 'indices.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')

    words_line = []
    for word in word_tokenize(text):
        if word not in string.punctuation + "\..." and word != '``' and word != '"':
            words_line.append(word.lower())

    # initializing liwc with zero
    liwc = [0] * len(indices)

    print("writing liwc ")

    for word in words_line:
        position = search(word_set_liwc, word)
        if position != []:
            tam = len(word_set_liwc[position[0]])
            for i in range(tam):
                if word_set_liwc[position[0]][i] in indices:
                    position_indices = search(indices, word_set_liwc[position[0]][i])
                    liwc[position_indices[0]] = liwc[position_indices[0]] + 1

    return liwc


def clean_sentence(sentence):
    # sent = sentence.replace("&gt", " ").strip()
    sent = re.sub(' {1,}', ' ', sentence).strip(' ').replace("&gt", " ")
    doc = nlp(sent)
    cleaned_text = ""
    for i, token in enumerate(doc):
        if token.text != " ":
            cleaned_text += token.text + " "
    return cleaned_text


# ------------------------------------ADITIONAL FEATURES---------------------------------------------------------------#
# additional features
def additionals(post):
    original_post = post.lower()
    post_nlp = nlp(post)

    greeting = sum([word_tokenize(original_post).count(word) for word in
                    ['olá', 'oi', 'como vai', 'tudo bem', 'como está', 'como esta', 'bom dia', 'boa tarde',
                     'boa noite']])
    compliment = sum([word_tokenize(original_post).count(word) for word in
                      ['parabéns', 'parabens', 'excelente', 'fantástico', 'fantastico', 'bom', 'bem', 'muito bom',
                       'muito bem', 'ótimo', 'otimo', 'incrivel', 'incrível', 'maravilhoso', 'sensacional',
                       'irrepreensível', 'irrepreensivel', 'perfeito']])
    ners = len(post_nlp.ents)

    return [greeting, compliment, ners]


def cross_validation(X, y, k, ntree, mtry, resultados):
    # SAVE RESULTS FOR EACH ROUND OF CROSS-VALIDATION
    resultados_parciais = {}
    resultados_parciais.update({'acurácia': []})
    resultados_parciais.update({'kappa': []})

    # cross-validation
    rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=1, random_state=54321)
    matriz_confusao = np.zeros((2, 2))

    for train_index, test_index in rkf.split(X, y):
        X_train, X_test = [X[i] for i in train_index], [X[j] for j in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[j] for j in test_index]

        X_train_np = np.asarray(X_train)
        X_test_np = np.asarray(X_test)
        y_train_np = np.asarray(y_train)
        y_test_np = np.asarray(y_test)

        classificador = xgb.XGBClassifier(n_estimators=ntree, use_label_encoder=False)
        classificador.fit(X_train_np, y_train_np)
        y_pred = classificador.predict(X_test_np)
        y_pred_np = np.asarray(y_pred)

        resultados_parciais["acurácia"].append(accuracy_score(y_pred_np, y_test_np))
        resultados_parciais["kappa"].append(cohen_kappa_score(y_pred_np, y_test_np))

        # THE FINAL CONFUSION MATRIX WILL BE THE SUM OF CONFUSION MATRICES FOR EACH KFOLD ROUND
        matriz_confusao = matriz_confusao + confusion_matrix(y_pred=y_pred_np, y_true=y_test_np)

    # SAVING PARAMETERS AND EXPERIMENT RESULTS
    resultados['ntree'].append(classificador.n_estimators)
    error_by_class(matriz_confusao, resultados)

    media = np.mean(resultados_parciais["acurácia"])
    std = np.std(resultados_parciais["acurácia"])
    resultados["acurácia"].append(str(round(media, 4)) + "(" + str(round(std, 4)) + ")")

    resultados["accuracy"].append(round(media, 4))
    resultados["erro"].append(round(1 - media, 4))

    media = np.mean(resultados_parciais["kappa"])
    std = np.std(resultados_parciais["kappa"])
    resultados["kappa"].append(str(round(media, 4)) + "(" + str(round(std, 4)) + ")")

    return resultados, classificador


def error_by_class(matriz_confusao, resultados):
    tam = matriz_confusao.shape[0]

    for i in range(tam):
        acerto = matriz_confusao[i][i]
        total = sum(matriz_confusao[i])

        taxa_erro = round(1 - (acerto / total), 4)
        print(taxa_erro)

        resultados["erro_classe_" + str(i)].append(taxa_erro)


def extract_features_cohmetrix(text):
    jobs_object = []
    params = {
        'text': text
    }

    query_params = parse.urlencode(params)
    request_search_url = f"http://localhost:4200/api/test?{query_params}"
    request_search = request.Request(request_search_url, method='GET')

    response = request.urlopen(request_search)
    response_json = response.read()
    jobs_object = json.loads(response_json)['features']
    features = []
    for line in jobs_object:
        if str(line) == "nan":
            features.append(float(0))
        else:
            features.append(float(line))

    return features


@csrf_exempt
@user_passes_test(is_instructor)
@ajax_required
@get_action(pf_related='actions')
def preview_next_all_false(
        request: http.HttpRequest,
        pk: Optional[int] = None,
        idx: Optional[int] = None,
        workflow: Optional[models.Workflow] = None,
        action: Optional[models.Action] = None,
) -> http.JsonResponse:
    """Preview message with all conditions evaluating to false.

    Previews the message that has all conditions incorrect in the position
    next to the one specified by idx

    The function uses the list stored in rows_all_false and finds the next
    index in that list (or the first one if it is the last. It then invokes
    the preview_response method

    :param request: HTTP Request object
    :param pk: Primary key of the action
    :param idx: Index of the preview requested
    :param workflow: Current workflow being manipulated
    :param action: Action being used in preview (set by the decorators)
    :return: JSON Response with the rendering of the preview
    """
    del workflow
    # Get the list of indexes
    idx_list = action.rows_all_false

    if not idx_list:
        # If empty, or None, something went wrong.
        return http.JsonResponse({'html_redirect': reverse('home')})

    # Search for the next element bigger than idx
    next_idx = next((nxt for nxt in idx_list if nxt > idx), None)

    if not next_idx:
        # If nothing found, then take the first element
        next_idx = idx_list[0]

    # Return the rendering of the given element
    return preview_response(request, pk, idx=next_idx, action=action)


@csrf_exempt
@user_passes_test(is_instructor)
@ajax_required
@get_action(pf_related='actions')
def preview_response(
        request: http.HttpRequest,
        pk: int,
        idx: int,
        workflow: Optional[models.Workflow] = None,
        action: Optional[models.Action] = None,
) -> http.JsonResponse:
    """Preview content of action.

    HTML request and the primary key of an action to preview one of its
    instances. The request must provide and additional parameter idx to
    denote which instance to show.

    :param request: HTML request object
    :param pk: Primary key of the an action for which to do the preview
    :param idx: Index of the reponse to preview
    :param workflow: Current workflow being manipulated
    :param action: Might have been fetched already
    :return: http.JsonResponse
    """
    del pk, workflow
    # If the request has the 'action_content', update the action
    action_content = request.POST.get('action_content')
    if action_content:
        action.set_text_content(action_content)

    # Initial context to render the response page.
    context = {'action': action, 'index': idx}
    if (
            action.action_type == models.Action.EMAIL_REPORT
            or action.action_type == models.Action.JSON_REPORT
    ):
        services.create_list_preview_context(action, context)
    else:
        services.create_row_preview_context(
            action,
            idx,
            context,
            request.GET.get('subject_content'))
    print(request.content_params)
    print(request.path)
    print(request.POST.items())
    print(request.POST.values())
    print(request.POST.get('action_content'))
    print(context)
    return http.JsonResponse({
        'html_form': render_to_string(
            'action/includes/partial_preview.html',
            context,
            request=request)})


@csrf_exempt
@user_passes_test(is_instructor)
@ajax_required
@get_action(pf_related='actions')
def preview_feedback(
        request: http.HttpRequest,
        pk: int,
        idx: int,
        workflow: Optional[models.Workflow] = None,
        action: Optional[models.Action] = None,
) -> http.JsonResponse:
    """Preview content of action.

    HTML request and the primary key of an action to preview one of its
    instances. The request must provide and additional parameter idx to
    denote which instance to show.

    :param request: HTML request object
    :param pk: Primary key of the an action for which to do the preview
    :param idx: Index of the reponse to preview
    :param workflow: Current workflow being manipulated
    :param action: Might have been fetched already
    :return: http.JsonResponse
    """
    del pk, workflow
    # If the request has the 'action_content', update the action
    action_content = request.POST.get('action_content')
    texto = str(action_content).replace("<p>", "").replace("</p>", "")
    data = {}
    data_string = ""

    sentence_clean = clean_sentence(texto)
    liwc = extract_liwc(sentence_clean)
    adds = additionals(sentence_clean)
    # cohmetrix = extract_features_cohmetrix(sentence_clean)
    cohmetrix = [1, 1, 10, 1.0, 0.0, 8.0, 0.0, 2.2, 1.398411797560202, 5.5, 2.958039891549808, 0,
                 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 6.646578958679581, 1.8941022799930982, 6.646578958679581,
                 383.0, 584.6666666666666, 302.0, 357.3333333333333, 386.6666666666667,
                 -12.784999999999997, 17.029999999999998, 0]
    features = []
    for x in liwc:
        features.append(x)
    for y in cohmetrix:
        features.append(y)
    features.append(adds[0])
    features.append(adds[1])
    features.append(adds[2])

    newfeatures = []
    newfeatures.append(features)
    np_features = np.asarray(newfeatures)

    for j in range(11):
        if j == 1 or j == 3 or j == 6 or j == 9:
            y_pred = 0
        else:
            global classifiers
            y_pred = classifiers[j].predict(np_features)
        print("classe predita ", y_pred)
        data['classe' + str(j)] = y_pred[0]
        data_string += 'classe' + str(j + 1) + ": " + str(y_pred[0]) + " <br> "

    if action_content:
        action.set_text_content(action_content)
    print("ACTION CONTENT - PREVIEW: " + action_content)
    # Initial context to render the response page.
    context = {'action': action, 'index': idx}
    print(context)
    if (
            action.action_type == models.Action.EMAIL_REPORT
            or action.action_type == models.Action.JSON_REPORT
    ):
        services.create_list_preview_context(action, context)
    else:
        services.create_row_preview_context(
            action,
            idx,
            context,
            request.GET.get('subject_content'))
    print(request)
    print(request.GET.get('subject_content'))
    print(context)
    # context.update({'data': data})
    # data = {}
    for i in range(0, 11):
        if i % 2 == 0:
            context.update({'classe' + str(i): data['classe' + str(i)]})
        else:
            context.update({'classe' + str(i): data['classe' + str(i)]})
    return http.JsonResponse({
        'html_form': render_to_string(
            'action/includes/partial_preview_feedback_result.html',
            context,
            request=request)})


@csrf_exempt
@user_passes_test(is_instructor)
@ajax_required
@get_action(pf_related='actions')
def init_classifier(
        request: http.HttpRequest,
        pk: int,
        idx: int,
        workflow: Optional[models.Workflow] = None,
        action: Optional[models.Action] = None,
) -> http.JsonResponse:
    """Preview content of action.

    HTML request and the primary key of an action to preview one of its
    instances. The request must provide and additional parameter idx to
    denote which instance to show.

    :param request: HTML request object
    :param pk: Primary key of the an action for which to do the preview
    :param idx: Index of the reponse to preview
    :param workflow: Current workflow being manipulated
    :param action: Might have been fetched already
    :return: http.JsonResponse
    """
    del pk, workflow
    # If the request has the 'action_content', update the action
    action_content = request.POST.get('action_content')

    csv_classes = 'classes.csv'
    classes = read_classes(csv_classes)

    csv_features = 'features.csv'
    data_train, features = read_data(csv_features)

    data = {}
    for j in range(len(classes)):
        resultados = {}
        resultados.update({'ntree': []})
        resultados.update({'mtry': []})
        resultados.update({'acurácia': []})
        resultados.update({'kappa': []})
        resultados.update({'accuracy': []})
        resultados.update({'erro': []})
        resultados.update({"erro_classe_" + str(0): []})
        resultados.update({"erro_classe_" + str(1): []})

        y_train = classes[j]

        global classifiers
        resultados, classifiers[j] = cross_validation(features, y_train, k=10, ntree=500, mtry=37,
                                                      resultados=resultados)
        print(resultados)
        data['dados'] = "Ready"

    if action_content:
        action.set_text_content(data['dados'])
    print("ACTION CONTENT - PREVIEW: " + action_content)
    # Initial context to render the response page.
    context = {'action': action, 'index': idx}
    print(context)
    if (
            action.action_type == models.Action.EMAIL_REPORT
            or action.action_type == models.Action.JSON_REPORT
    ):
        services.create_list_preview_context(action, context)
    else:
        services.create_row_preview_context(
            action,
            idx,
            context,
            request.GET.get('subject_content'))
    print(request.path)
    # new_url = str(request.path).replace('1/classifier/', 'edit/')
    # print(new_url)
    print(request.GET.get('subject_content'))
    # context.update({'show_values': 'yes'})
    return http.JsonResponse({
        'html_form': render_to_string(
            'action/includes/partial_preview_feedback.html',
            context,
            request=request)})
