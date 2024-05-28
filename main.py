import asyncio
import logging
import os.path
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings, PromptTemplate,
)

from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from evaluating import evaluate_answers
from read_questions_and_answers import read_q_and_a_catalog, write_answers_to_excel, write_evaluations_to_excel


def generate_answers(catalog: list[dict], embedding_model: str, top_k: int) -> list[dict]:
    """
    Generiert Antworten fuer einen Fragen-Katalog
    :param catalog: Liste mit Dictionaries, die Fragen-Nr, Frage und Antworten enthaelt
    :param embedding_model: Entweder OpenAI Text-Embedding-3-Large oder Lokal
    :param top_k: Anzahl der Top-Uebereinstimmungen
    :return: Liste mit Dictionaries, die Fragen, Antworten und Metadaten enthaelt
    """
    index = handle_embedding(embedding_model)

    logging.info(f"Embedding Model: {embedding_model}")

    qa_prompt_str = (
        "Kontextinformationen stehen unterhalb.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Beantworte die Query basierend auf den Kontextinformationen und ohne vorheriges Wissen, "
        "und gebe wenn nötig die Bezeichnung der jeweiligen Maske mit, die direkt über der Maskenbeschriftung in "
        "Klammern nach der Kapitelüberschrift steht. Die Bezeichnung endet immer mit '.w'\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    qa_prompt_tmpl = PromptTemplate(qa_prompt_str)

    data_as_list = []  # Liste zum Schreiben der Antworten in eine Excel-Datei

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
    )

    # Fuer jede Frage wird eine Antwort mit den Kontextinformationen generiert, die mit ihren Metadaten und Scores in
    # einer Excel-Datei gespeichert wird
    for entry in catalog:
        question_nr = entry["nr"]
        question = entry["question"]
        reference_answer = entry["answer"]

        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        generated_answer = query_engine.query(question)
        similarity_scores = ""
        nodes = ""
        for source_node in generated_answer.source_nodes:
            similarity_score = source_node.get_score()
            similarity_scores += f"{similarity_score:.2f}, "
            nodes += f"({source_node.node.text}), "
        print(f'Antwort: {generated_answer}')

        metadata_list = list(generated_answer.metadata.values())
        metadata_string = ""
        for metadata in metadata_list:
            metadata_text = "(" + metadata["page_label"] + ", " + metadata["file_name"] + ")"
            metadata_string += f"{metadata_text}, "

        data_unit = {"embedding_model": embedding_model,
                     "nr": question_nr,
                     "top_k": top_k,
                     "generated_answer": generated_answer.response,
                     "reference_answer": reference_answer,
                     "nodes_similarity_score": similarity_scores,
                     "metadata": metadata_string,
                     "nodes": nodes
                     }

        data_as_list.append(data_unit)

    # Liste in den entsprechenden Sheets in einer Excel-Datei speichern
    write_answers_to_excel(embedding_model=embedding_model, llm="gpt-4-turbo", data=data_as_list)
    return data_as_list


async def evaluate(generated_answers_list: list[dict], embedding_model: str, top_k: int):
    """
    Evaluiert fuer eine gegebene Liste an Antworten mit Referenz-Antworten einen Similarity-Score und schreibt die
    Ergebnisse in eine Excel-Datei :param generated_answers_list: :param embedding_model:  Entweder OpenAI
    Text-Embedding-3-Large oder Lokal :param top_k: Anzahl der Top-Uebereinstimmungen :return: None
    """
    evaluations_list = []
    for entry in generated_answers_list:
        question_nr = entry["nr"]
        generated_answer = entry["generated_answer"]
        reference_answer = entry["reference_answer"]
        nodes_similarity_score = entry["nodes_similarity_score"]
        similarity_score, passing_score = await evaluate_answers(reference_answer, generated_answer)
        logging.info(f"Question no.: {question_nr}, Nodes Similarity: {nodes_similarity_score}, Similarity Score: {similarity_score}, Passing: {passing_score}")

        evaluation_unit = {"nr": question_nr, "top_k": top_k, "nodes_similarity_score": nodes_similarity_score, "similarity_score": similarity_score,
                           "passing_score": passing_score}
        evaluations_list.append(evaluation_unit)

    # Nodes-Similarity-Score, Simularity-Score und Passing-Score in das entsprechende Sheet einer Excel-Datei schreiben
    write_evaluations_to_excel(embedding_model=embedding_model, llm="gpt-4-turbo", data=evaluations_list)


async def main():
    # Logging Konfiguration
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting...")

    catalog = read_q_and_a_catalog("./questions_and_answers/HASy_Fragenkatalog.xlsx")

    logging.info("Read questions and answers from HASy_Fragenkatalog.xlsx")

    starting_time = datetime.now()
    logging.info(f"Started at: {starting_time.strftime('%d-%m-%Y %H:%M:%S')}")

    # Embedding Model aus der Konfiguration laden (Umgebungsvariable)
    embedding_model = os.getenv('EMBEDDING_MODEL', 'OpenAI')

    # Hauptteil, Antworten generieren und evaluieren
    top_k = 5
    generated_answers_as_list = generate_answers(catalog, embedding_model, top_k)
    await evaluate(generated_answers_as_list, embedding_model, top_k)

    end_time = datetime.now()
    logging.info(f"Finished at: {end_time.strftime('%d-%m-%Y %H:%M:%S')}")

    processing_time = end_time - starting_time
    logging.info(f"Total processing time: {processing_time}")


def handle_embedding(embedding_model: str):
    llm = OpenAI(model="gpt-4-turbo")
    Settings.llm = llm
    if embedding_model == "OpenAI":
        persist_dir = "./storage/openai"
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    else:
        persist_dir = "./storage/bge_onnx"
        if not os.path.exists("./bge_onnx"):
            OptimumEmbedding.create_and_save_optimum_model(
                "BAAI/bge-small-en-v1.5", "./bge_onnx"
            )
        Settings.embed_model = OptimumEmbedding(folder_name="./bge_onnx")

    # Index erstellen fuer bessere Performance
    if not os.path.exists(persist_dir):
        # Dokumente laden und Index erstellen
        docs = SimpleDirectoryReader("./documentation").load_data()
        index = VectorStoreIndex.from_documents(docs)
        # Speichern fuer weitere Nutzung
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        # Bereits vorhandenen Index laden
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

    return index


if __name__ == "__main__":
    asyncio.run(main())
