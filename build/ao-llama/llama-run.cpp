#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#define CTX_SIZE 2048
#define BATCH_SIZE 512
#define MAX_TOKEN_LEN 256

struct params_struct {
    std::string model;
    std::string prompt;
    int n_threads = 4;
    int n_threads_batch = 4;
};

// Global variables
static params_struct params;
static llama_model* model = nullptr;
static llama_batch batch = {};
static llama_context* ctx = nullptr;
static int tks_processed = 0;

extern "C" bool l_llama_on_progress(float progress, void* user_data);
extern "C" void l_llama_on_log(enum ggml_log_level level, const char* text, void* user_data);

// Helper function to safely free resources
void cleanup_resources() {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (batch.token) {
        llama_batch_free(batch);
        batch = {};
    }
}

std::vector<llama_token> tokenize(llama_context* ctx, const std::string& text, bool add_bos) {
    if (!ctx || text.empty()) {
        return {};
    }

    std::vector<llama_token> tokens;
    tokens.resize(text.length() + (add_bos ? 1 : 0));
    
    int n = llama_tokenize(
        llama_get_model(ctx),
        text.c_str(),
        text.length(),
        tokens.data(),
        tokens.size(),
        add_bos,
        true
    );

    if (n < 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Tokenization failed", nullptr);
        return {};
    }

    tokens.resize(n);
    return tokens;
}


bool decode_helper(llama_context* ctx, llama_batch& batch, std::vector<float>& batch_logits, int n_batch, int n_vocab) {
    int prev_outputs = 0;
    for (int i = 0; i < (int)batch.n_tokens; i += n_batch) {
        const int n_tokens = std::min<int>(n_batch, batch.n_tokens - i);
        
        llama_batch batch_view = {
            n_tokens,
            batch.token    + i,
            nullptr,
            batch.pos      + i,
            batch.n_seq_id + i,
            batch.seq_id   + i,
            batch.logits   + i,
        };

        if (llama_decode(ctx, batch_view) != 0) {
            return false;
        }

        int n_outputs = 0;
        for (int i = 0; i < n_tokens; ++i) {
            n_outputs += batch_view.logits[i] != 0;
        }

        memcpy(batch_logits.data() + size_t(prev_outputs)*n_vocab, 
               llama_get_logits(ctx), 
               size_t(n_outputs)*n_vocab*sizeof(float));

        prev_outputs += n_outputs;
    }
    return true;
}

extern "C" int llama_load(char* model_path) {
    if (!model_path) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Invalid model path", nullptr);
        return 1;
    }

    cleanup_resources();
    params.model = model_path;

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true; 
    
    model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    return 0;
}

bool isCtxFull() {
    return ctx && tks_processed >= llama_n_ctx(ctx);
}

void llama_reset_context() {
    cleanup_resources();
    tks_processed = 0;

    batch = llama_batch_init(BATCH_SIZE, 0, 1);
    
    llama_context_params ctx_params = llama_context_default_params();
    const int n_ctx_train = llama_n_ctx_train(model);
    ctx_params.n_ctx = std::min(CTX_SIZE, n_ctx_train);
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch;
    ctx_params.offload_kqv = true; 
    ctx_params.embeddings = false;  

    ctx = llama_new_context_with_model(model, ctx_params);

    if (!ctx) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Failed to create context", nullptr);
        return;
    }

    l_llama_on_log(GGML_LOG_LEVEL_INFO, "Context created successfully", nullptr);
}

extern "C" int llama_set_prompt(char* prompt) {
    if (!prompt || !model) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Invalid prompt or model not loaded", nullptr);
        return 1;
    }
    fprintf(stderr, "Resetting context...\n");
    llama_reset_context();
    params.prompt = prompt;

    fprintf(stderr, "Tokenizing prompt...\n");

    std::vector<llama_token> tokens_list = tokenize(ctx, params.prompt, true);
    if (tokens_list.empty()) {
        return 1;
    }

    if (isCtxFull()) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Context full", nullptr);
        return 1;
    }

    // Handle encoder models specifically
    if (llama_model_has_encoder(model)) {
        fprintf(stderr, "Encoding prompt...\n");
        for (size_t i = 0; i < tokens_list.size(); i++) {
            common_batch_add(batch, tokens_list[i], i, {0}, false);
            tks_processed++;
        }

        if (llama_encode(ctx, batch) != 0) {
            fprintf(stderr, "Encoding failed...\n");
            l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Encode failed", nullptr);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = llama_token_bos(model);
        }

        fprintf(stderr, "Adding decoder start token...\n");
        common_batch_clear(batch);
        common_batch_add(batch, decoder_start_token_id, 0, {0}, false);
    } else {
        // Regular model handling
        fprintf(stderr, "Adding tokens to batch...\n");
        for (size_t i = 0; i < tokens_list.size(); i++) {
            common_batch_add(batch, tokens_list[i], i, {0}, false);
            tks_processed++;
        }
    }

    if (batch.n_tokens > 0) {
        fprintf(stderr, "Setting logit for last token...\n");
        batch.logits[batch.n_tokens - 1] = true;
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Decoding failed...\n");
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Decode failed", nullptr);
        return 1;
    }

    return 0;
}

extern "C" char* llama_next() {
    if (!ctx || !model) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Context or model not initialized", nullptr);
        return nullptr;
    }

    if (isCtxFull()) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Context is full", nullptr);
        return nullptr;
    }

    auto n_vocab = llama_n_vocab(model);
    auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    
    if (!logits) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Failed to get logits", nullptr);
        return nullptr;
    }

    char* token = (char*)malloc(MAX_TOKEN_LEN);
    if (!token) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Memory allocation failed", nullptr);
        return nullptr;
    }

    // Initialize sampler chain
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    
    // Add sampling strategies
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));     // Limit to top 40 tokens
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1)); // Keep tokens with cumulative probability <= 0.95
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));    // Temperature for controlling randomness
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));       // Random seed for reproducibility

    // Sample the next token using the sampler chain
    const llama_token new_token_id = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);
    llama_sampler_free(smpl);

    // Check for end of generation
    if (llama_token_is_eog(model, new_token_id)) {
        free(token);
        return nullptr;
    }

    // Convert token to string
    char piece[MAX_TOKEN_LEN];
    if (llama_token_to_piece(llama_get_model(ctx), new_token_id, piece, sizeof(piece), 0, false) < 0) {
        free(token);
        return nullptr;
    }

    strncpy(token, piece, MAX_TOKEN_LEN - 1);
    token[MAX_TOKEN_LEN - 1] = '\0';
    tks_processed++;

    // Prepare next batch
    llama_batch_free(batch);
    batch = llama_batch_init(BATCH_SIZE, 0, 1);
    
    // Add the new token to the batch
    common_batch_add(batch, new_token_id, tks_processed, { 0 }, true);

    // Decode the new batch
    if (llama_decode(ctx, batch) != 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Failed to decode", nullptr);
        free(token);
        return nullptr;
    }

    return token;
}

extern "C" char* llama_run(int len) {
    if (len <= 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Invalid length", nullptr);
        return nullptr;
    }

    char* response = (char*)malloc(len * MAX_TOKEN_LEN);
    if (!response) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Memory allocation failed", nullptr);
        return nullptr;
    }
    response[0] = '\0';

    for (int i = 0; i < len; i++) {
        char* next_token = llama_next();
        if (!next_token) break;
        
        if (strlen(response) + strlen(next_token) >= len * MAX_TOKEN_LEN - 1) {
            free(next_token);
            break;
        }
        
        strcat(response, next_token);
        free(next_token);
    }

    return response;
}

extern "C" int llama_add(char* new_string) {
    if (!new_string || !ctx) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Invalid input or context not initialized", nullptr);
        return 1;
    }

    std::vector<llama_token> new_tokens_list = tokenize(ctx, new_string, true);
    if (new_tokens_list.empty()) {
        return 1;
    }

    if (isCtxFull()) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Context full, cannot add more tokens", nullptr);
        return 1;
    }

    for (size_t i = 0; i < new_tokens_list.size(); i++) {
        common_batch_add(batch, new_tokens_list[i], tks_processed + i, {0}, false);
        tks_processed++;
    }

    if (batch.n_tokens > 0) {
        batch.logits[batch.n_tokens - 1] = true;
    }

    if (llama_decode(ctx, batch) != 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Failed to decode", nullptr);
        return 1;
    }

    return 0;
}

extern "C" void llama_stop() {
    cleanup_resources();
    if (model) {
        llama_free_model(model);
        model = nullptr;
    }
    llama_backend_free();
}