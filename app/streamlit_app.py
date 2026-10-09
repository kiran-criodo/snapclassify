"""SnapClassify: the breed suggester TailFinder's volunteers use when they list a pet. Given.

    streamlit run app/streamlit_app.py            # locally, after python -m snapclassify.export
Streamlit Community Cloud: point a new app at this file; it installs app/requirements.txt (no PyTorch).
Uploads are processed in memory and never stored or sent anywhere.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the repo root, for the snapclassify package
from snapclassify.constants import ATTRIBUTION  # noqa: E402
from snapclassify.predict import MODEL_DIR, load_model, predict  # noqa: E402

st.set_page_config(page_title="SnapClassify", page_icon="🐾")
st.title("SnapClassify")
st.caption("Upload a photo of the pet you're listing: SnapClassify suggests three breeds to pick from, "
           "or says it isn't sure.")


@st.cache_resource
def model():
    return load_model(MODEL_DIR)


if not (MODEL_DIR / "model.onnx").exists():
    st.error("No model yet: run `python -m snapclassify.export --threshold ...` and commit app/model/.")
    st.stop()

session, labels, threshold = model()
upload = st.file_uploader("Pet photo", type=["jpg", "jpeg", "png", "webp"])
if upload is not None:
    try:
        image = Image.open(upload)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("That file isn't an image we can read. Try a JPEG or PNG.")
        st.stop()
    st.image(image, width=360)
    t0 = time.perf_counter()
    result = predict(image, session, labels, threshold)
    ms = (time.perf_counter() - t0) * 1000
    if result["not_sure"]:
        st.warning("Not sure. Please choose the breed yourself; these are the closest matches.")
    else:
        st.success(f"Suggested breed: **{result['suggestion']}**")
    for t in result["top"]:
        st.progress(t["confidence"], text=f"{t['breed']}: {t['confidence']:.0%}")
    st.caption(f"{ms:.0f} ms on a CPU · \"not sure\" below {threshold:.0%} confidence · {len(labels)} pure breeds: "
               "mixed breeds will get the closest pure breed")

st.divider()
st.caption(ATTRIBUTION + " Photos you upload are processed in memory and not stored.")
