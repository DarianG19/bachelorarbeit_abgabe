from llama_index.core.evaluation import SemanticSimilarityEvaluator, CorrectnessEvaluator


async def evaluate_answers(reference: str, response: str) -> tuple[float | None, bool | None]:
    """
    Bewertet die Qualität einer Antwort und Referenzantwort anhand der semantischen Ähnlichkeit.
    :param reference: Referenzantwort
    :param response: Gegebene Antwort
    :return: Score und Passing-Wert
    """
    similarity_score, result_passing = await evaluate_similarity(response=response, reference=reference)
    return similarity_score, result_passing


async def evaluate_similarity(response: str, reference: str) -> tuple[float | None, bool | None]:
    """
    Bewertet die Qualität einer Antwort und Referenzantwort anhand der semantischen Ähnlichkeit.
    Konkret wird der Ähnlichkeit zwischen den Einbettungen der generierten Antwort und der Referenzantwort berechnet.
    :param response: Gegebene Antwort
    :param reference: Referenzantwort
    :return: Score und Passing-Wert
    """
    evaluator = SemanticSimilarityEvaluator()
    result = await evaluator.aevaluate(response=response, reference=reference)
    return result.score, result.passing
