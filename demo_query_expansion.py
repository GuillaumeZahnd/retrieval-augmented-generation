import os

from query_expansion import QueryExpansion


if __name__ == "__main__":

    user_query = "what is a permanent?"
    nb_variants = 5

    with open(os.path.join("data", "mtg.txt"), "r") as fid:
        context = fid.read()

    query_expansion = QueryExpansion(nb_variants=nb_variants)

    alternative_queries_raw = query_expansion.expand_query(
        user_query=user_query,
        context=context)

    alternative_queries = query_expansion.format_llm_output(raw_output=alternative_queries_raw)

    print("\nUser query:")
    print(user_query)
    print("\nAlternative queries:")
    for q in range(len(alternative_queries)):
        print("[{}] {}".format(q, alternative_queries[q]))
