`python -m snapclassify.export --threshold <yours>` writes the app's model here: `model.onnx` (about 6 MB),
`labels.json` and `config.json` (the "not sure" threshold). Commit all three: Streamlit Community Cloud serves
the app from your GitHub repo.
