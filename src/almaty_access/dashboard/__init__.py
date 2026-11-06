"""Phase 7 dashboard: a Streamlit-free data layer plus export stages.

The Streamlit app (``app/streamlit_app.py``) is a thin veneer over this
package so every piece of logic — data-directory resolution, layer
tables, colors, isochrone reach, Altair chart specs, the committed
data bundle and the static web map — is importable and unit-tested
without a browser or a Streamlit runtime.
"""
