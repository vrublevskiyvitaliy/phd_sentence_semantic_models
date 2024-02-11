# phd_sentence_semantic_models

Repository with materials for disertation:

* Models for representing the semantics of natural language sentences
* Моделі представлення семантики речень природної мови

Repository structure:
* dependancy_tree_graph_model
    * Hand-crafted features based on dependancy tree with basic machine learning models.
* grammar
    * Simple Grammar Error correction using Earley algorithm (old version, python 2 is used)
* models:
    * Contains base and modification Translformer models 
    * File `deberta_model_attention_change.py` contains main model `DebertaForSequenceClassificationV2` with updates to the attention layer.
    * File `enriched_tokeniser.py` contains updated and enriched tokeniser that is used by Transformer models. 
        * Use `preprocess_dataset_final` with config with `tokeniser_list=['attention_dep']` for `DebertaForSequenceClassificationV2`.
* transformer_notebooks:
    * Contains notebooks with full E2E training and testing for Transformer models
    * `Deberta_with_Attention_Change` contains the experiment results with `DebertaForSequenceClassificationV2`.
* utils:
    * Train eval cycle
    * Seed set up for reproducability
* llm
    * Notebooks to enrich datasets with LLM paraphrases of sentances (not covered in dissertation)
    * `Train Llama 2 on MRPC` notebook to train LLM to identify paraphrases [results](https://huggingface.co/VitaliiVrublevskyi/Llama-2-7b-hf-finetuned-mrpc-v0.4)