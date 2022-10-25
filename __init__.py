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
inputFolder = "/mnt/volume_annif_projects/data-sets/bldg-regs/docs/validate/nn-bv-stw-ensemble-en/*.json"
notesFile = "/mnt/volume_annif_projects/data-sets/bldg-regs/docs/validate/nn-bv-stw-ensemble-en/MachineLearning.md"

# testing
# inputFolder = "data-sets/*.json"
# notesFile = "data-sets/MachineLearning.md"

seconds = 5  # change to 60 for a minute


def cleanTitles():
    # remove unnececary data from path
    jsonData = getjson()
    jsonData = [basename(x) for x in jsonData]
    jsonData = [x[0:-5] for x in jsonData]

    return jsonData


def getjson():
    # get json files and sort by creation date
    jsonData = glob(inputFolder)
    jsonData.sort(key=getmtime)
    return jsonData


def parsejson():
    # get ml data from jsonfiles

    count = 1  # count is used to configure the number of ticks on the xaxis

    template = {
        "Index": [],
        "Precision_doc_avg": [],
        "Recall_doc_avg": [],
        "F1_score_doc_avg": [],
    }

    jsonData = getjson()

    # read json files
    for file in jsonData:
        with open(file, "r") as f:
            data = json.load(f)
            data["Index"] = count

            for k in template.keys():
                template[k].append(data[k])

        count += 1

    template["Title"] = cleanTitles()
    df = pd.DataFrame(template)
    return df


def parseNotes():

    template = {
        "titles": [],
        "ml model": [],
        "date": [],
        "sources": [],
        "vocab": [],
        "vocab notes": [],
        "training": [],
        "incremental learning": [],
        "comments": [],
    }

    with open(notesFile, "r") as f:
        data = f.readlines()

    data = [x.replace("\n", "") for x in data]
    titles = [x.replace("## ", "") for x in data if "##" in x]
    text = [data[data.index(x) + 1 : data.index(x) + 1 + 8] for x in data if "##" in x]

    for note in text:
        for j in note:
            j = j.split("=")
            j = [x.strip() for x in j]

            template[j[0].lower()].append(j[1])

    template["titles"] = titles

    df = pd.DataFrame(template)
    return df


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        html.Label(
            html.Strong(
                "Building Regulation Guidance (Machine Learning (ML) Natural language processing (NLP) automated subject indexing"
            )
        ),
        html.Div(
            [
                html.Label(html.Strong("ML Model"), style=dict(width="10%")),
                dcc.Dropdown(
                    id="dropdown_model",
                    multi=True,
                    persistence=True,
                    persistence_type="local",
                    style=dict(width="90%"),
                ),
            ],
            style=dict(display="flex"),
        ),
        html.Div(
            [
                html.Label(html.Strong("Sources"), style=dict(width="10%")),
                dcc.Dropdown(
                    id="dropdown_sources",
                    multi=True,
                    persistence=True,
                    persistence_type="local",
                    style=dict(width="90%"),
                ),
            ],
            style=dict(display="flex"),
        ),
        html.Div(
            [
                html.Label(html.Strong("Vocab"), style=dict(width="10%")),
                dcc.Dropdown(
                    id="dropdown_vocab",
                    multi=True,
                    persistence=True,
                    persistence_type="local",
                    style=dict(width="90%"),
                ),
            ],
            style=dict(display="flex"),
        ),
        html.Div(
            [
                html.Label(html.Strong("Training"), style=dict(width="10%")),
                dcc.Dropdown(
                    id="dropdown_training",
                    multi=True,
                    persistence=True,
                    persistence_type="local",
                    style=dict(width="90%"),
                ),
            ],
            style=dict(display="flex"),
        ),
        html.Strong(html.Label(id="graphError")),
        dcc.Graph(id="ml", animate=False),  # ml graph
        html.Div(id="table"),  # notes table
        dcc.Interval(id="update", interval=seconds * 1000),
    ]
)


# updater for ML Graph


@app.callback(
    [
        Output("ml", "figure"),
        Output("table", "children"),
        Output("dropdown_model", "options"),
        Output("dropdown_sources", "options"),
        Output("dropdown_vocab", "options"),
        Output("dropdown_training", "options"),
        Output("graphError", "children"),
    ],
    [
        Input("update", "n_intervals"),
        Input("dropdown_model", "value"),
        Input("dropdown_sources", "value"),
        Input("dropdown_vocab", "value"),
        Input("dropdown_training", "value"),
    ],
)
def updateAll(n_intervals, ddv_models, ddv_sources, ddv_vocab, ddv_training):

    df = parsejson()

    notes = parseNotes()
    query = []

    # dropdowns
    ddModels = notes["ml model"].unique()
    ddSources = notes["sources"].unique()
    ddVocab = notes["vocab"].unique()
    ddTraining = notes["training"].unique()

    if not (ddv_models == None or ddv_models == []):
        query.append(f"`ml model` == {ddv_models}")

    if not (ddv_sources == None or ddv_sources == []):
        query.append(f"sources == {ddv_sources}")

    if not (ddv_vocab == None or ddv_vocab == []):
        query.append(f"vocab == {ddv_vocab}")

    if not (ddv_training == None or ddv_training == []):
        query.append(f"training == {ddv_training}")

    if query != []:
        query = " and ".join(query)
        notes = notes.query(query)
        df = df.query(f'Title == {notes["titles"].values.tolist()}')
    else:
        notes = notes

    # graph
    fig = go.Figure()
    # pres_doc_avg line
    fig.add_trace(
        go.Scatter(
            x=df["Index"],
            y=df["Precision_doc_avg"],
            mode="lines",
            name="Precision_doc_avg",
            line_shape="spline",
        )
    )

    # recall_doc_avg line
    fig.add_trace(
        go.Scatter(
            x=df["Index"],
            y=df["Recall_doc_avg"],
            mode="lines",
            name="Recall_doc_avg",
            line_shape="spline",
        )
    )

    # f1_score_doc_avg line
    fig.add_trace(
        go.Scatter(
            x=df["Index"],
            y=df["F1_score_doc_avg"],
            mode="lines",
            name="F1_score_doc_avg",
            line_shape="spline",
        )
    )

    fig.update_layout(
        xaxis=dict(
            # set custom x axis ticks and ticklabels
            tickmode="array",
            tickvals=df["Index"],
            ticktext=df["Title"],
        )
    )

    # Error
    if len(df) == 1:
        errorMSG = "! Query contains only one record. Cannot Plot"
    else:
        errorMSG = ""

    # table
    tbl = dbc.Table.from_dataframe(notes, bordered=True, responsive=True, striped=True)

    return (fig, tbl, ddModels, ddSources, ddVocab, ddTraining, errorMSG)


server = app.server

if __name__ == "__main__":
    app.run_server(debug=False)
