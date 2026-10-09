# SnapClassify

TailFinder (a fictional pet-adoption marketplace) lists cats and dogs from partner shelters. Volunteers upload a
photo and type the breed, and a third of listings arrive with the breed blank or wrong, so adopters who filter by
breed never see those pets. **SnapClassify** is the listing tool's breed suggester: a small image model,
transfer-learned on the Oxford-IIIT Pet dataset, that suggests three breeds from a photo, or says "not sure", and
runs on a free CPU host.

The listings product manager's brief: *"Suggest the breed from the photo, three options for the volunteer to pick
from, and if you're not sure, say so. It has to be cheap enough to run on every listing, and I want to know
whether we'd be better off just calling one of the big AI models."*

## How it fits together
- **Make the model, with PyTorch** (in the container, or on a free GPU with `notebooks/train_colab.ipynb`):
  `snapclassify/data.py` (the photos, splits and transforms) -> `snapclassify/train.py` (transfer learning, saved to
  `models/`) -> `snapclassify/evaluate.py` (test-split metrics and the Gemini comparison, in `reports/eval.md`) ->
  `snapclassify/export.py` (the ONNX model, the breed names and the "not sure" threshold, in `app/model/`).
- **Suggest a breed, without PyTorch:** `app/streamlit_app.py` calls `snapclassify/predict.py`, which runs
  `app/model/model.onnx` on ONNX Runtime.

## Commands (in the container, at /workspace)
```
pytest tests/visible -v                                  # the visible tests
bash run_assessment.sh                                   # what the grader runs; prints TEST_STATUS_SUCCESS
python -m snapclassify.train --synthetic                 # seconds: generated images, random weights, no download
python -m snapclassify.train --subset 10                 # 10 breeds of real photos on a CPU (downloads ~800 MB once)
python -m snapclassify.evaluate --subset 10              # the test split -> reports/eval.md
python -m snapclassify.evaluate --subset 10 --compare 50 # + Gemini on 50 of those photos (GEMINI_API_KEY in .env)
python -m snapclassify.export --threshold 0.5            # models/ -> app/model/, with your threshold
python -m snapclassify.predict my_pet.jpg                # one photo through the ONNX model
streamlit run app/streamlit_app.py                       # the app, on http://localhost:8501
```

## What is committed, and what isn't
- Committed: the code, `reports/`, and `app/model/` (`model.onnx`, `labels.json`, `config.json`), which the live
  app serves.
- Never committed (git-ignored): `.env` (your key), `data/pets/` (the dataset: CC BY-SA 4.0, about 800 MB) and
  `models/` (training checkpoints).

## Deploying the app
Streamlit Community Cloud runs the app from a GitHub repository. Push this repository to GitHub, sign in at
[share.streamlit.io](https://share.streamlit.io) with GitHub, choose **Create app**, and set the main file path to
`app/streamlit_app.py`. Community Cloud installs `app/requirements.txt` (the requirements file next to the main
file wins over the one at the repository root), so the app runs without PyTorch.

## Runbook
- **Retrain and redeploy:** `python -m snapclassify.train --subset 10`, then `python -m snapclassify.evaluate --subset 10`,
  then `python -m snapclassify.export --threshold <yours>`; commit `app/model/` and `reports/`, and push to GitHub.
  The running app keeps the model it loaded in memory, so reboot it from its "Manage app" menu to load the new one.
- **Change the "not sure" threshold:** choose it from the threshold table in `reports/eval.md`, set `"threshold"` in
  `app/model/config.json`, commit, push, and reboot the app.
- **The app is asleep:** apps without traffic for 12 hours go to sleep. Open the URL and choose
  "Yes, get this app back up!"; anyone who can view the app can wake it.
- **The app is over its memory limit:** the free host allows at most 2.7 GB. Check that `app/requirements.txt` still
  has no `torch` and that `snapclassify/predict.py` imports none, then reboot the app from its "Manage app" menu.
- **A breed it keeps getting wrong:** find it in the per-breed table and the confusions in `reports/eval.md`;
  collect more photos of it and of the breed it is confused with, retrain, and compare the two tables.

## Known limits
- Pure breeds only: the breeds of the Oxford-IIIT Pet dataset (37, or 10 for a CPU run). A mixed breed gets the
  closest pure breed, or "not sure".
- Photos unlike the dataset (puppies, kennels, poor light, odd angles) get more "not sure" answers and more mistakes.
- Softmax confidence is not calibrated: "70% sure" is not right 70% of the time.

## Credits
Breed model trained on the Oxford-IIIT Pet Dataset (O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar,
2012), licensed CC BY-SA 4.0. The dataset is never committed or shipped with the app.
