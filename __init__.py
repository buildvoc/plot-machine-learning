import dash
from dash.dependencies import Output, Input
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from glob import glob
import json
from os.path import getmtime
from os.path import basename

# production
inputFolder = '/mnt/volume_annif_projects/data-sets/bldg-regs/docs/validate/nn-bv-stw-ensemble-en/*.json'
notesFile = '/mnt/volume_annif_projects/data-sets/bldg-regs/docs/validate/nn-bv-stw-ensemble-en/MachineLearning.md'

# testing
#inputFolder = 'json/*.json'
#notesFile = 'Machine_Learning.md'

seconds = 5  # change to 60 for a minute


def cleanTitles():
    # remove unnececary data from path
    jsonData = getjson()
    jsonData = [basename(x) for x in jsonData]
    jsonData = [x[0:-5] for x in jsonData]

    return (jsonData)


def getjson():
    # get json files and sort by creation date
    jsonData = glob(inputFolder)
    jsonData.sort(key=getmtime)
    return (jsonData)


def parsejson():
    # get ml data from jsonfiles

    count = 1  # count is used to configure the number of ticks on the xaxis

    template = {"Index": [],
                "Precision_doc_avg": [],
                'Recall_doc_avg': [],
                'F1_score_doc_avg': []}

    jsonData = getjson()

    # read json files
    for file in jsonData:
        with open(file, 'r') as f:
            data = json.load(f)
            data['Index'] = count

            for k in template.keys():
                template[k].append(data[k])

        count += 1
    df = pd.DataFrame(template)
    return (df)


def parseNotes():

    template = {'titles': [],
                'ml model': [],
                'date': [],
                'sources': [],
                'vocab': [],
                'training': [],
                'incremental learning': [],
                'comments': []}

    with open(notesFile, 'r') as f:
        data = f.readlines()

    data = [x.replace('\n', '') for x in data]
    titles = [x.replace('## ', '') for x in data if '##' in x]
    text = [data[data.index(x) + 1: data.index(x) + 1 + 7]
            for x in data if '##' in x]

    for note in text:
        for j in note:
            j = j.split('=')
            j = [x.strip() for x in j]

            template[j[0].lower()].append(j[1])

    template['titles'] = titles

    df = pd.DataFrame(template)
    return (df)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        html.H2(
            'Building Regulation Guidance (Machine Learning (ML) Natural language processing (NLP) automated subject indexing'),
        dcc.Graph(id='ml', animate=True),  # ml graph
        html.Div(id='table'),  # notes table
        dcc.Interval(id='graph-update', interval=seconds*1000),
    ]
)


# updater for ML Graph

@ app.callback(Output('ml', 'figure'),
               [Input('graph-update', 'n_intervals')])
def update_graph_scatter(input_data):

    df = parsejson()

    fig = go.Figure()

    # pres_doc_avg line
    fig.add_trace(go.Scatter(
        x=df['Index'], y=df['Precision_doc_avg'], mode='lines', name='Precision_doc_avg', line_shape='spline'))

    # recall_doc_avg line
    fig.add_trace(go.Scatter(
        x=df['Index'], y=df['Recall_doc_avg'], mode='lines', name='Recall_doc_avg', line_shape='spline'))

    # f1_score_doc_avg line
    fig.add_trace(go.Scatter(
        x=df['Index'], y=df['F1_score_doc_avg'], mode='lines', name='F1_score_doc_avg', line_shape='spline'))

    fig.update_layout(
        xaxis=dict(
            # set custom x axis ticks and ticklabels
            tickmode='array',
            tickvals=list(df['Index']),
            ticktext=cleanTitles(),))

    return (fig)


# updater for Notes Table

@ app.callback(Output('table', 'children'),
               [Input('graph-update', 'n_intervals')])
def search_fi(n_intervals):
    df = parseNotes()
    tbl = dbc.Table.from_dataframe(
        df, bordered=True, responsive=True, striped=True)
    return (tbl)


server = app.server

if __name__ == '__main__':
    app.run_server(debug=False)
