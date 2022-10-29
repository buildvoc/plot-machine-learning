import dash
from dash.dependencies import Output, Input, State
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import dash_cytoscape as cyto

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

cyto_stylesheet = [
    {
        "selector": "node",
        "style": {
            "width": "mapData(size, 0, 1, 20, 100)",
            "height": "mapData(size, 0, 1, 20, 100)",
            "content": "data(label)",
            "font-size": "12px",
            "text-valign": "center",
            "text-halign": "center",
        },
    }
]

styles = {
    "container": {
        "position": "fixed",
        "display": "flex",
        "flex-direction": "column",
        "height": "100%",
        "width": "100%",
    },
    "cy-container": {"flex": "1", "position": "relative"},
    "cytoscape": {
        "position": "absolute",
        "width": "100%",
        "height": "100%",
        "z-index": 999,
    },
}


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
        "analyzer": [],
        "vocab": [],
        "vocab notes": [],
        "training": [],
        "training notes": [],
        "incremental learning": [],
        "comments": [],
    }

    with open(notesFile, "r") as f:
        data = f.readlines()

    data = [x.replace("\n", "") for x in data]
    titles = [x.replace("## ", "") for x in data if "##" in x]
    text = [data[data.index(x) + 1 : data.index(x) + 1 + 10] for x in data if "##" in x]

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
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Line Graph",
                    children=[
                        html.Label(
                            html.Strong(
                                "Building Regulation Guidance (Machine Learning (ML) Natural language processing (NLP) automated subject indexing"
                            )
                        ),
                        html.Div(
                            [
                                html.Label(
                                    html.Strong("ML Model"), style=dict(width="20%")
                                ),
                                dcc.Dropdown(
                                    id="dropdown_model",
                                    multi=True,
                                    persistence=True,
                                    persistence_type="local",
                                    style=dict(width="80%"),
                                ),
                            ],
                            style=dict(display="flex"),
                        ),
                        html.Div(
                            [
                                html.Label(
                                    html.Strong("Analyzer"), style=dict(width="20%")
                                ),
                                dcc.Dropdown(
                                    id="dropdown_analyzer",
                                    multi=True,
                                    persistence=True,
                                    persistence_type="local",
                                    style=dict(width="80%"),
                                ),
                            ],
                            style=dict(display="flex"),
                        ),
                        html.Div(
                            [
                                html.Label(
                                    html.Strong("Sources"), style=dict(width="20%")
                                ),
                                dcc.Dropdown(
                                    id="dropdown_sources",
                                    multi=True,
                                    persistence=True,
                                    persistence_type="local",
                                    style=dict(width="80%"),
                                ),
                            ],
                            style=dict(display="flex"),
                        ),
                        html.Div(
                            [
                                html.Label(
                                    html.Strong("Vocab"), style=dict(width="20%")
                                ),
                                dcc.Dropdown(
                                    id="dropdown_vocab",
                                    multi=True,
                                    persistence=True,
                                    persistence_type="local",
                                    style=dict(width="80%"),
                                ),
                            ],
                            style=dict(display="flex"),
                        ),
                        html.Div(
                            [
                                html.Label(
                                    html.Strong("Training"), style=dict(width="20%")
                                ),
                                dcc.Dropdown(
                                    id="dropdown_training",
                                    multi=True,
                                    persistence=True,
                                    persistence_type="local",
                                    style=dict(width="80%"),
                                    optionHeight=50,
                                ),
                            ],
                            style=dict(display="flex"),
                        ),
                        html.Strong(html.Label(id="graphError")),
                        dcc.Graph(id="ml", animate=False),  # ml graph
                        html.Div(id="table"),  # notes table
                        dcc.Interval(id="update-line", interval=seconds * 1000),
                    ],
                ),
                dcc.Tab(
                    label="Network Graph",
                    children=[
                        html.Div(
                            style=styles["container"],
                            children=[
                                html.Div(
                                    className="cy-container",
                                    style=styles["cy-container"],
                                    children=[
                                        cyto.Cytoscape(
                                            id="network graph",
                                            style=styles["cytoscape"],
                                            stylesheet=cyto_stylesheet,
                                            layout={
                                                "name": "cose",
                                                "animate": False,
                                            },
                                            responsive=True,
                                        )
                                    ],
                                ),
                            ],
                        ),
                        dcc.Interval(id="update-network", interval=seconds * 1000),
                    ],
                ),
            ]
        )
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
        Output("dropdown_analyzer", "options"),
        Output("graphError", "children"),
    ],
    [
        Input("update-line", "n_intervals"),
        Input("dropdown_model", "value"),
        Input("dropdown_sources", "value"),
        Input("dropdown_vocab", "value"),
        Input("dropdown_training", "value"),
        Input("dropdown_analyzer", "value"),
    ],
)
def updateLine(
    n_intervals,
    ddv_models,
    ddv_sources,
    ddv_vocab,
    ddv_training,
    ddv_analyzer,
):

    df = parsejson()

    notes = parseNotes()
    query = []

    # dropdowns
    ddModels = notes["ml model"].unique()
    ddSources = notes["sources"].unique()
    ddVocab = notes["vocab"].unique()
    ddTraining = notes["training"].unique()
    ddAnalyzer = notes["analyzer"].unique()

    # not is used outside the brackets to prevent a bug.
    if not (ddv_models == None or ddv_models == []):
        query.append(f"`ml model` == {ddv_models}")

    if not (ddv_sources == None or ddv_sources == []):
        query.append(f"sources == {ddv_sources}")

    if not (ddv_vocab == None or ddv_vocab == []):
        query.append(f"vocab == {ddv_vocab}")

    if not (ddv_training == None or ddv_training == []):
        query.append(f"training == {ddv_training}")

    if not (ddv_analyzer == None or ddv_analyzer == []):
        query.append(f"analyzer == {ddv_analyzer}")

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
        ),
        uirevision="False",
    )

    # Error
    if len(df) == 1:
        errorMSG = "! Query contains only one record. Cannot Plot"
    else:
        errorMSG = ""

    # table
    tbl = dbc.Table.from_dataframe(notes, bordered=True, responsive=True, striped=True)

    return (
        fig,
        tbl,
        ddModels,
        ddSources,
        ddVocab,
        ddTraining,
        ddAnalyzer,
        errorMSG,
    )


@app.callback(
    Output("network graph", "elements"), Input("update-network", "n_intervals")
)
def updateNetwork(n_intervals):
    df = parsejson()
    notes = parseNotes()
    metrics = ["ml model", "sources", "analyzer", "vocab", "training"]

    elements = (
        [
            # Nodes elements
            {
                "data": {
                    "id": f"{m}-{row['titles']}",
                    "label": f"{row[m][0:10]}...",
                }
            }
            for m in metrics
            for index, row in notes.iterrows()
        ]
        + [
            {
                "data": {
                    "id": f"F1-{row['titles']}",
                    "label": f"F1 Score",
                    "size": df["F1_score_doc_avg"][index],
                },
                "classes": "F1",
            }
            for index, row in notes.iterrows()
        ]
        + [
            {
                "data": {
                    "source": f"{m}-{row['titles']}",
                    "target": f"F1-{row['titles']}",
                }
            }
            for m in metrics
            for index, row in notes.iterrows()
        ]
    )
    return elements


server = app.server

if __name__ == "__main__":
    app.run_server(debug=False)
