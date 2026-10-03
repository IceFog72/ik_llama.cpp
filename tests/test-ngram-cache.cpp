#include "ngram-cache.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdio>

static void check_draft(std::vector<llama_token> inp, int ngram_min, int ngram_max, int n_draft,
                        common_ngram_cache context, common_ngram_cache dynamic, common_ngram_cache corpus,
                        const std::vector<llama_token> & expected) {
    const auto original_inp = inp;
    const auto original_context = context;
    const auto original_dynamic = dynamic;
    const auto original_corpus = corpus;
    std::vector<llama_token> draft{inp.empty() ? 0 : inp.back()};
    common_ngram_cache_draft(inp, draft, n_draft, ngram_min, ngram_max, context, dynamic, corpus);
    assert(draft == expected);
    assert(inp == original_inp);
    assert(context == original_context && dynamic == original_dynamic && corpus == original_corpus);
}

static common_ngram key(std::initializer_list<llama_token> tokens) {
    const std::vector<llama_token> values(tokens);
    return common_ngram(values.data(), values.size());
}

int main() {
    // Empty caches still construct lookup keys. Cover every supported range and the short-history boundary.
    for (int length = 0; length <= LLAMA_NGRAM_MAX + 1; ++length) {
        std::vector<llama_token> inp(length, 7);
        for (int min = LLAMA_NGRAM_MIN; min <= LLAMA_NGRAM_MAX; ++min) {
            for (int max = min; max <= LLAMA_NGRAM_MAX; ++max) {
                check_draft(inp, min, max, 4, {}, {}, {}, {length ? 7 : 0});
            }
        }
    }

    common_ngram_cache cache;
    cache[key({1, 2})][3] = 4;
    cache[key({1, 2, 3})][4] = 4;
    cache[key({1, 2, 3, 4})][5] = 4;
    cache[key({4})][99] = 4; // Once history grows, prefer the available four-token match.
    for (const auto & inp : {std::vector<llama_token>{1, 2}, std::vector<llama_token>{1, 2, 3},
                            std::vector<llama_token>{1, 2, 3, 4}}) {
        std::vector<llama_token> expected{inp.back()};
        for (llama_token token = inp.back() + 1; token <= 5; ++token) {
            expected.push_back(token);
        }
        check_draft(inp, 1, 4, 4, cache, {}, {}, expected);
        check_draft(inp, 1, 4, 4, {}, cache, {}, expected);
    }
    check_draft({1, 2}, 1, 4, 1, cache, {}, {}, {2, 3});
    check_draft({1, 2}, 1, 4, 0, cache, {}, {}, {2});

    common_ngram_cache corpus;
    corpus[key({1, 2})][3] = 2;
    corpus[key({2, 3})][4] = 2;
    // No primary n-gram fits yet, but static fallback can grow the history.
    check_draft({1, 2}, 4, 4, 4, {}, {}, corpus, {2, 3, 4});
    check_draft({1}, 1, 4, 4, cache, {}, corpus, {1});
    std::puts("PASS: n-gram short history, cache fallbacks, growing history, limits, and cache immutability");
}
