# Plotly Dash Graph Building  Regulation Guidance (Machine Learning (ML) Natural language processing (NLP) automated subject indexing
## Introduction
Machine Learning using plotly and dash graph in table to track review changes in performance to algorithm configurations.
example can be seen http://plotly-ml.buildvoc.co.uk/

![image](https://user-images.githubusercontent.com/76884997/197250805-1d2f9787-5724-4f9e-baf6-626ddf391769.png)

## Graph json generation
The graph is automatically updated from the folder on the server. All you have to do is create a metric file in .json and that will be pushed to the graph.

## Table in markdown
On the machine learning notes,
to be able to track configuration changes use the markdown template then save file on the server and the table is automatically updated.

## Annif Metric Creation
in annif command line 
annif eval bldg-omikuji-parabel-en --metrics-file data-sets/bldg-regs/docs/validate/2021-10.1-bldg-omikuji-parabel-en.json data-sets/bldg-regs/docs/validate/validate.tsv 
annif eval command https://annif.readthedocs.io/en/stable/source/commands.html#annif-eval
