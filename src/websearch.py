from langchain_community.retrievers import PubMedRetriever


def search_pubmed(query, k=3):
    docs = PubMedRetriever(top_k_results=k).invoke(query)
    for d in docs:
        d.metadata = {
            "source_type": "pubmed",
            "title": d.metadata.get("Title", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{d.metadata.get('uid', '')}/",
        }
    return docs
