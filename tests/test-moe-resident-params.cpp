#include "common.h"
#include "llama.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool value, const char * message) {
    if (!value) { fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

static bool parse_workspace(int residency, const char * workspace) {
    gpt_params params;
    params.moe_resident = residency;
    std::vector<std::string> args{"test", "--moe-resident-ubatch", workspace};
    std::vector<char *> argv;
    for (auto & arg : args) argv.push_back(arg.data());
    try {
        return gpt_params_parse_ex(int(argv.size()), argv.data(), params);
    } catch (const std::invalid_argument & error) {
        check(std::string(error.what()).find("--moe-resident-ubatch requires auto residency") != std::string::npos,
                "workspace failure reports its configuration requirement");
        return false;
    }
}

int main(int argc, char ** argv) {
    for (int residency : {0, 1, 8}) {
        check(!parse_workspace(residency, "512"), "CLI rejects nonzero workspace without auto residency");
        check(parse_workspace(residency, "0"), "CLI permits disabling the workspace");
    }
    check(parse_workspace(-1, "512"), "CLI accepts workspace with auto residency");
    check(parse_workspace(-1, "0"), "CLI accepts disabling workspace with auto residency");

    // A real vocabulary-only model reaches the public API's parameter checks
    // without allocating an inference arena or requiring a large model.
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    mparams.n_gpu_layers = 0;
    const char * path = argc > 1 ? argv[1] : "models/ggml-vocab-llama-spm.gguf";
    auto * model = llama_model_load_from_file(path, mparams);
    check(model != nullptr, "API fixture loads a real model object");
    if (model) {
        std::string log;
        llama_log_set([](ggml_log_level, const char * text, void * data) {
            *static_cast<std::string *>(data) += text;
        }, &log);
        for (int residency : {0, 1, 8}) {
            auto params = llama_context_default_params();
            params.moe_resident = residency;
            params.moe_resident_ubatch = 512;
            log.clear();
            auto * ctx = llama_init_from_model(model, params);
            check(ctx == nullptr, "API rejects invalid workspace parameters");
            check(log.find("adaptive residency workspace requires auto residency") != std::string::npos,
                    "API rejects workspace before context allocation");
            if (ctx) llama_free(ctx);
        }
        llama_log_set(nullptr, nullptr);
        llama_free_model(model);
    }
    llama_backend_free();
    if (!failures) fprintf(stderr, "VERIFIED workspace-params\n");
    return failures ? 1 : 0;
}
