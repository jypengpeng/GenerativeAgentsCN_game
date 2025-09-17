"""generative_agents.storage.index"""

import os
import time
import json
import difflib
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index import core as index_core
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

from modules import utils


class LlamaIndex:
    def __init__(self, embedding_config, path=None):
        self._config = {"max_nodes": 0}
        self._mode = "vector"  # "vector" | "simple"

        # Simple mode: no embeddings, in-memory docs + json persistence
        if embedding_config["provider"] == "none":
            self._mode = "simple"
            self._docs = {}
            self._path = path
            # load persisted docs/config if present
            if path and os.path.exists(path):
                docs_fp = os.path.join(path, "index_docs.json")
                cfg_fp = os.path.join(path, "index_config.json")
                if os.path.exists(docs_fp):
                    try:
                        with open(docs_fp, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        for node_id, d in data.get("docs", {}).items():
                            self._docs[node_id] = self._simple_node(
                                d.get("text", ""), node_id, d.get("metadata", {})
                            )
                        # restore max_nodes
                        self._config["max_nodes"] = data.get("max_nodes", len(self._docs))
                    except Exception:
                        self._docs = {}
                if os.path.exists(cfg_fp):
                    try:
                        self._config = utils.load_dict(cfg_fp)
                    except Exception:
                        pass
            return

        if embedding_config["provider"] == "hugging_face":
            embed_model = HuggingFaceEmbedding(model_name=embedding_config["model"])
        elif embedding_config["provider"] == "ollama":
            embed_model = OllamaEmbedding(
                model_name=embedding_config["model"],
                base_url=embedding_config["base_url"],
                ollama_additional_kwargs={"mirostat": 0},
            )
        elif embedding_config["provider"] == "openai":
            embed_model = OpenAIEmbedding(
                model_name=embedding_config["model"],
                api_base=embedding_config["base_url"],
                api_key=embedding_config["api_key"],
            )
        else:
            raise NotImplementedError(
                "embedding provider {} is not supported".format(embedding_config["provider"])
            )

        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        Settings.num_output = 1024
        Settings.context_window = 4096
        if path and os.path.exists(path):
            self._index = index_core.load_index_from_storage(
                index_core.StorageContext.from_defaults(persist_dir=path),
                show_progress=True,
            )
            self._config = utils.load_dict(os.path.join(path, "index_config.json"))
        else:
            self._index = index_core.VectorStoreIndex([], show_progress=True)
        self._path = path

    # ------- simple mode helpers -------
    def _simple_node(self, text, id_, metadata):
        class SimpleNode:
            def __init__(self, text, id_, metadata):
                self.text = text
                self.id_ = id_
                self.metadata = metadata or {}

        return SimpleNode(text, id_, metadata)

    def _docs_map(self):
        if getattr(self, "_mode", "vector") == "simple":
            return self._docs
        return self._index.docstore.docs

    def add_node(
        self,
        text,
        metadata=None,
        exclude_llm_keys=None,
        exclude_embedding_keys=None,
        id=None,
    ):
        if self._mode == "simple":
            metadata = metadata or {}
            id = id or "node_" + str(self._config["max_nodes"])
            self._config["max_nodes"] += 1
            node = self._simple_node(text, id, metadata)
            self._docs[id] = node
            return node
        while True:
            try:
                metadata = metadata or {}
                exclude_llm_keys = exclude_llm_keys or list(metadata.keys())
                exclude_embedding_keys = exclude_embedding_keys or list(metadata.keys())
                id = id or "node_" + str(self._config["max_nodes"])
                self._config["max_nodes"] += 1
                node = TextNode(
                    text=text,
                    id_=id,
                    metadata=metadata,
                    excluded_llm_metadata_keys=exclude_llm_keys,
                    excluded_embed_metadata_keys=exclude_embedding_keys,
                )
                self._index.insert_nodes([node])
                return node
            except Exception as e:
                print(f"LlamaIndex.add_node() caused an error: {e}")
                time.sleep(5)

    def has_node(self, node_id):
        return node_id in self._docs_map()

    def find_node(self, node_id):
        return self._docs_map()[node_id]

    def get_nodes(self, filter=None):
        def _check(node):
            if not filter:
                return True
            return filter(node)

        return [n for n in self._docs_map().values() if _check(n)]

    def remove_nodes(self, node_ids, delete_from_docstore=True):
        if self._mode == "simple":
            for nid in node_ids:
                if nid in self._docs:
                    del self._docs[nid]
            return
        self._index.delete_nodes(node_ids, delete_from_docstore=delete_from_docstore)

    def cleanup(self):
        now, remove_ids = utils.get_timer().get_date(), []
        for node_id, node in self._docs_map().items():
            create = utils.to_date(node.metadata["create"]) if node.metadata.get("create") else now
            expire = utils.to_date(node.metadata["expire"]) if node.metadata.get("expire") else now
            if create > now or expire < now:
                remove_ids.append(node_id)
        self.remove_nodes(remove_ids)
        return remove_ids

    def retrieve(
        self,
        text,
        similarity_top_k=5,
        filters=None,
        node_ids=None,
        retriever_creator=None,
    ):
        if self._mode == "simple":
            # collect candidates
            docs = list(self._docs_map().values())
            if node_ids:
                node_ids = set(node_ids)
                docs = [d for d in docs if d.id_ in node_ids]

            # apply metadata filters (ExactMatchFilter only)
            if filters is not None and hasattr(filters, "filters"):
                for f in getattr(filters, "filters", []) or []:
                    key = getattr(f, "key", None)
                    val = getattr(f, "value", None)
                    if key is not None:
                        docs = [d for d in docs if d.metadata.get(key) == val]

            # scoring
            def _ratio(a, b):
                try:
                    return difflib.SequenceMatcher(None, a, b).ratio()
                except Exception:
                    return 0.0

            if text:
                for d in docs:
                    d._relevance = _ratio(text, d.text)
            else:
                for d in docs:
                    d._relevance = 0.0

            # normalize fields
            def _normalize(values, default=0.0):
                if not values:
                    return {}
                vmin, vmax = min(values.values()), max(values.values())
                if vmax - vmin == 0:
                    return {k: 0.5 for k in values}
                return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}

            # recency from access time
            access_map = {}
            for d in docs:
                try:
                    access_map[d.id_] = utils.to_date(d.metadata.get("access")).timestamp()
                except Exception:
                    access_map[d.id_] = 0
            recency_n = _normalize(access_map)

            relevance_map = {d.id_: getattr(d, "_relevance", 0.0) for d in docs}
            relevance_n = _normalize(relevance_map)

            importance_map = {}
            for d in docs:
                try:
                    importance_map[d.id_] = float(d.metadata.get("poignancy", 0))
                except Exception:
                    importance_map[d.id_] = 0.0
            importance_n = _normalize(importance_map)

            # weights roughly mirror AssociateRetriever defaults
            for d in docs:
                score = (
                    0.5 * recency_n.get(d.id_, 0.0)
                    + 3.0 * relevance_n.get(d.id_, 0.0)
                    + 2.0 * importance_n.get(d.id_, 0.0)
                )
                d._score = score
            docs = sorted(docs, key=lambda x: getattr(x, "_score", 0.0), reverse=True)
            return docs[:similarity_top_k]

        try:
            retriever_creator = retriever_creator or VectorIndexRetriever
            return retriever_creator(
                self._index,
                similarity_top_k=similarity_top_k,
                filters=filters,
                node_ids=node_ids,
            ).retrieve(text)
        except Exception as e:
            # print(f"LlamaIndex.retrieve() caused an error: {e}")
            return []

    def query(
        self,
        text,
        similarity_top_k=5,
        text_qa_template=None,
        refine_template=None,
        filters=None,
        query_creator=None,
    ):
        if self._mode == "simple":
            nodes = self.retrieve(text, similarity_top_k=similarity_top_k, filters=filters)
            return "\n".join([n.text for n in nodes])
        kwargs = {
            "similarity_top_k": similarity_top_k,
            "text_qa_template": text_qa_template,
            "refine_template": refine_template,
            "filters": filters,
        }
        while True:
            try:
                if query_creator:
                    query_engine = query_creator(retriever=self._index.as_retriever(**kwargs))
                else:
                    query_engine = self._index.as_query_engine(**kwargs)
                return query_engine.query(text)
            except Exception as e:
                print(f"LlamaIndex.query() caused an error: {e}")
                time.sleep(5)

    def save(self, path=None):
        path = path or self._path
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        if self._mode == "simple":
            docs_fp = os.path.join(path, "index_docs.json")
            data = {
                "max_nodes": self._config.get("max_nodes", 0),
                "docs": {nid: {"text": d.text, "metadata": d.metadata} for nid, d in self._docs.items()},
            }
            with open(docs_fp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            utils.save_dict(self._config, os.path.join(path, "index_config.json"))
            return
        self._index.storage_context.persist(path)
        utils.save_dict(self._config, os.path.join(path, "index_config.json"))

    @property
    def nodes_num(self):
        return len(self._docs_map())
