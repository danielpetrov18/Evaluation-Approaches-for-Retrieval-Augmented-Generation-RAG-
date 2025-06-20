"""
R2R support the creation of indices for semantic search.
Both HNSW and IVF_FLAT are supported.

https://r2r-docs.sciphi.ai/api-and-sdks/indices/indices
"""

# pylint: disable=C0301
# pylint: disable=E0401

from typing import Union

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from backend.index import list_indices, create_index

if __name__ == "__page__":
    st.title("📊 Indices")

    with st.sidebar:
        st.markdown("""
### About Indices in R2R

R2R supports the creation of indices to speed up semantic search in Retrieval-Augmented Generation (RAG) applications. There are two supported **Approximate Nearest Neighbor (ANN)** index types:

- **HNSW** (Hierarchical Navigable Small World)
- **IVF_FLAT** (Inverted File Flat Index)

---

**Default Indices**  
R2R comes with **prebuilt keyword-based indices** (e.g., `GIN` on `tsvector`) that are useful for full-text filtering, but **not optimized for semantic similarity**.
The ones for semantic similarity can be identified by the `index_method` of `hnsw` or `ivf_flat`.

---

**ANN Indices for Semantic Search**  
To enable vector-based semantic retrieval, you can define your own index on the `chunks` table. These ANN index definitions typically include:
- `index_method`: `hnsw` or `ivf_flat`
- `index_measure`: similarity metric (e.g., `cosine_distance`)
- `index_arguments`: tuning parameters like `m`, `ef_construction`, `ef`

---

**Small Corpus Warning**  
For small document sets (e.g., < 500 chunks), creating an ANN index might **slow down performance** due to graph overhead. Use ANN indexing **only when you have a large corpus** where retrieval benefits from sublinear scaling.

---

Learn More: 
- [R2R Index Docs](https://r2r-docs.sciphi.ai/api-and-sdks/indices/indices)
- [HNSW vs IVF_FLAT](https://medium.com/@emreks/comparing-ivfflat-and-hnsw-with-pgvector-performance-analysis-on-diverse-datasets-e1626505bc9a)
- [HNSW](https://towardsdatascience.com/similarity-search-part-4-hierarchical-navigable-small-world-hnsw-2aad4fe87d37/)
""")


    tab_list, tab_create = st.tabs(["List Indices", "Create Index"])

    with tab_list:
        if st.button(label="Fetch Indices", key="fetch_indices_btn"):
            list_indices()

    with tab_create:
        with st.expander("Upload Instructions", expanded=False):
            st.markdown(
                """Upload a YAML file that defines an index configuration with the following structure.
Below is an example of a valid YAML configuration for creating an index in R2R."""
            )

            st.code("""
                my_index_name: # Required
                    index_method: hnsw  # Required
                    index_measure: cosine_distance # Required
                    index_arguments:
                        m: 16
                        ef_construction: 64
                        ef: 40 
            """, language="yaml", line_numbers=True)

        uploaded_file: Union[UploadedFile, None] = st.file_uploader(
            label="Upload YAML Index File",
            type=["yaml", "yml"]
        )

        if st.button(label="Create Index", key="create_index_btn"):
            if not uploaded_file:
                st.error("Please upload a YAML file to create an index.")
            else:
                create_index(uploaded_file)
