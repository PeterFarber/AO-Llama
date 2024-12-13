#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "common.h"

#include <unordered_map>
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <thread>
#include <algorithm>
#include <cstring>   // for strncpy, etc.
#include <limits>    // for INFINITY
#include <random>    // for random sampling if desired

#define CTX_SIZE 8192
#define BATCH_SIZE 2048
#define MAX_TOKEN_LEN 256
#define DEBUG_MODE 0

#define DEBUG_PRINT(...) do { if (DEBUG_MODE) fprintf(stderr, __VA_ARGS__); } while (0)

struct params_struct {
    std::string model;
    std::string prompt;
    int n_threads = 4;
    int n_threads_batch = 4;
    float temperature = 0.7f;   
    float top_p = 0.9f;         
    int top_k = 40;             
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;     
    float min_p = 0.00f;        
};


static params_struct params;
static llama_model* model = nullptr;
static llama_batch batch = {};
static llama_context* ctx = nullptr;
static int tks_processed = 0;

extern "C" bool l_llama_on_progress(float progress, void* user_data);
extern "C" void l_llama_on_log(enum ggml_log_level level, const char* text, void* user_data);

struct TokenMapping {
    llama_token token_id;
    std::string raw_text;
    std::string replacement;
};

static std::vector<TokenMapping> special_tokens;

void initialize_special_tokens(llama_model* model) {
    special_tokens.clear();
    
    const int n_vocab = llama_n_vocab(model);
    if (n_vocab <= 0) {
        return;
    }

    const std::vector<std::pair<std::string, std::string>> token_mappings = {
        {"Ġ", " "},
        {"▁", " "},
        {"Ċ", "\n"},
        {"ĉ", "\n"},
        {"<0x0A>", "\n"},
        {"<0x20>", " "},
    };

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        const char* token_str = llama_token_get_text(model, token_id);
        if (!token_str) continue;
        
        std::string token_text(token_str);
        
        for (const auto& [pattern, replacement] : token_mappings) {
            if (token_text == pattern) {
                special_tokens.push_back({token_id, token_text, replacement});
                DEBUG_PRINT("Found special token: '%s' (id: %d) -> '%s'\n", 
                            token_text.c_str(), token_id, replacement.c_str());
                break;
            }
        }
    }

    DEBUG_PRINT("Initialized %zu special tokens\n", special_tokens.size());
}

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

    std::vector<llama_token> tokens(text.size() + (add_bos ? 1 : 0));
    int n = llama_tokenize(llama_get_model(ctx),
                           text.c_str(),
                           (int)text.size(),
                           tokens.data(),
                           (int)tokens.size(),
                           add_bos,
                           true);

    if (n < 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Tokenization failed", nullptr);
        return {};
    }

    tokens.resize(n);
    return tokens;
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

    initialize_special_tokens(model);
    fprintf(stderr, "Initialized %zu special tokens\n", special_tokens.size());
    
    const int n_vocab = llama_n_vocab(model);
    fprintf(stderr, "Model loaded with vocabulary size: %d\n", n_vocab);
    
    if (n_vocab <= 0) {
        fprintf(stderr, "Invalid vocabulary size\n");
        llama_free_model(model);
        model = nullptr;
        return 1;
    }

    return 0;
}

bool isCtxFull() {
    if (!ctx) return false;
    int max_ctx = llama_n_ctx(ctx);
    return (tks_processed >= max_ctx);
}

void llama_reset_context() {
    cleanup_resources();
    tks_processed = 0;

    DEBUG_PRINT("Initializing new batch...\n");
    batch = llama_batch_init(BATCH_SIZE, 0, 1);
    
    llama_context_params ctx_params = llama_context_default_params();
    const int n_ctx_train = llama_n_ctx_train(model);
    ctx_params.n_ctx = std::min(CTX_SIZE, n_ctx_train);
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.offload_kqv = true;
    ctx_params.embeddings = false;
    ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    ctx_params.rope_freq_base = 10000.0f;
    ctx_params.rope_freq_scale = 1.0f;

    DEBUG_PRINT("Creating new context (n_ctx: %d, n_threads: %d)...\n", 
                ctx_params.n_ctx, ctx_params.n_threads);
    
    ctx = llama_new_context_with_model(model, ctx_params);

    if (!ctx) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Failed to create context", nullptr);
        return;
    }

    DEBUG_PRINT("Context created successfully\n");
}

extern "C" int llama_set_prompt(char* prompt) {
    if (!prompt || !model) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Invalid prompt or model not loaded", nullptr);
        return 1;
    }
    
    llama_reset_context();
    params.prompt = prompt;

    std::vector<llama_token> tokens_list = tokenize(ctx, params.prompt, false);
    if (tokens_list.empty()) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Tokenization failed", nullptr);
        return 1;
    }

    common_batch_clear(batch);
    if (batch.n_tokens >= BATCH_SIZE) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Batch overflow with BOS token", nullptr);
        return 1;
    }

    common_batch_add(batch, llama_token_bos(model), 0, {0}, false);

    if (llama_decode(ctx, batch) != 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Decode failed for BOS token", nullptr);
        return 1;
    }
    tks_processed++;

    const size_t chunk_size = BATCH_SIZE - 1; 
    for (size_t i = 0; i < tokens_list.size(); i += chunk_size) {
        common_batch_clear(batch);
        
        size_t current_chunk_size = std::min(chunk_size, tokens_list.size() - i);
        
        for (size_t j = 0; j < current_chunk_size; j++) {
            if (batch.n_tokens >= BATCH_SIZE) {
                l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Batch overflow while adding tokens", nullptr);
                return 1;
            }
            bool is_last = (j == current_chunk_size - 1 && (i + current_chunk_size) == tokens_list.size());
            common_batch_add(batch, tokens_list[i + j], tks_processed, {0}, is_last);
            tks_processed++;
        }

        if (batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        if (llama_decode(ctx, batch) != 0) {
            l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Decode failed for chunk", nullptr);
            return 1;
        }

        fprintf(stderr, "Processed chunk %zu/%zu (tokens %zu-%zu)\n", 
                (i / chunk_size) + 1, 
                (tokens_list.size() + chunk_size - 1) / chunk_size,
                i, 
                i + current_chunk_size - 1);
    }

    if (batch.n_tokens > 0) {
        float* logits = llama_get_logits_ith(ctx, (int)batch.n_tokens - 1);
        if (logits) {
            float max_logit = -INFINITY;
            int vocab = llama_n_vocab(model);
            for (int i = 0; i < vocab; i++) {
                if (logits[i] > max_logit) {
                    max_logit = logits[i];
                }
            }
            fprintf(stderr, "Initial decode completed. Max logit: %f\n", max_logit);
        }
    }

    return 0;
}

std::string clean_token_text(const char* raw_text) {
    if (!raw_text) return "";
    std::string text(raw_text);
    if (text.empty()) return "";

    static const std::vector<std::string> skip_list = {
        "<|endoftext|>", "<s>", "", "<pad>", "<unk>", "âĢĶ", "ðŁĮĲðŁĴ", "ðŁ", "Ĳ", "Ĵ", "Į"
    };
    for (auto &sk : skip_list) {
        if (text == sk) {
            return "";
        }
    }

    for (const auto& special : special_tokens) {
        if (text == special.raw_text) {
            return special.replacement;
        }
    }

    std::string result;
    size_t i = 0;
    while (i < text.length()) {
        bool handled = false;
        for (const auto& special : special_tokens) {
            size_t len = special.raw_text.size();
            if (i + len <= text.size() && text.compare(i, len, special.raw_text) == 0) {
                result += special.replacement;
                i += len;
                handled = true;
                break;
            }
        }
        if (handled) continue;

        if (i + 3 < text.length() && (unsigned char)text[i] == 0xF0) {
            i += 4; // skip emoji/unicode sequences
            continue;
        }

        if (i + 1 < text.length() && (unsigned char)text[i] == 0xC4) {
            unsigned char next_char = (unsigned char)text[i + 1];
            if (next_char == 0xA0) { // Ġ
                result += ' ';
                i += 2;
                continue;
            } else if (next_char == 0x82) { // Ċ
                result += '\n';
                i += 2;
                continue;
            } else if (next_char == 0xAE || next_char == 0xB2 || next_char == 0xB4) {
                i += 2;
                continue;
            }
        }

        if (i + 2 < text.length() &&
            (unsigned char)text[i] == 0xE2 &&
            (unsigned char)text[i+1] == 0x96 &&
            (unsigned char)text[i+2] == 0x81) {
            result += ' ';
            i += 3;
            continue;
        }

        if (i + 2 < text.length() &&
            (unsigned char)text[i] == 0xC3 &&
            (unsigned char)text[i+1] == 0xA2 &&
            (unsigned char)text[i+2] == 0xC5) {
            // âĢĶ
            result += "...";
            i += 3;
            continue;
        }

        if (i + 1 < text.length() &&
            (unsigned char)text[i] == 0xC3 &&
            (unsigned char)text[i+1] == 0xB0) {
            // skip ðŁ
            i += 2;
            continue;
        }

        unsigned char c = (unsigned char)text[i];
        if (c >= 32 && c <= 126) {
            result += (char)c;
        }
        i++;
    }
    
    return result;
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

    if (batch.n_tokens == 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "No tokens in batch to get logits from", nullptr);
        return nullptr;
    }

    int last_idx = (int)batch.n_tokens - 1;
    float* logits = llama_get_logits_ith(ctx, last_idx);
    if (!logits) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Failed to get logits", nullptr);
        return nullptr;
    }

    const int n_vocab = llama_n_vocab(model);
    if (n_vocab <= 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Invalid vocabulary size", nullptr);
        return nullptr;
    }

    // Apply repetition penalty
    if (params.repeat_penalty != 1.0f && params.repeat_last_n > 0) {
        int n_tokens = std::min(params.repeat_last_n, tks_processed);
        if (n_tokens > 0 && (size_t)batch.n_tokens >= (size_t)n_tokens) {
            std::vector<llama_token> last_tokens(n_tokens);
            for (int i = 0; i < n_tokens; i++) {
                last_tokens[i] = batch.token[batch.n_tokens - n_tokens + i];
            }
            for (auto tk : last_tokens) {
                if (tk >= 0 && tk < n_vocab) {
                    logits[tk] /= params.repeat_penalty;
                }
            }
        }
    }

    // Apply temperature
    if (params.temperature > 0.0f) {
        for (int i = 0; i < n_vocab; i++) {
            logits[i] /= params.temperature;
        }
    }

    // Create a candidate list
    std::vector<std::pair<float,int>> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        candidates.push_back({logits[i], i});
    }

    // Sort by logit descending
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    // Top-k filtering
    if (params.top_k > 0 && params.top_k < n_vocab) {
        candidates.resize(params.top_k);
    }

    // Compute stable softmax on the truncated set
    // Find max logit
    float max_logit = -INFINITY;
    for (auto &c : candidates) {
        if (c.first > max_logit) max_logit = c.first;
    }

    float sum = 0.0f;
    for (auto &c : candidates) {
        float p = expf(c.first - max_logit);
        c.first = p; // reuse 'first' to store probability
        sum += p;
    }

    if (sum <= 0.0f) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "No candidates after top-k", nullptr);
        return nullptr;
    }

    // Normalize
    for (auto &c : candidates) {
        c.first /= sum;
    }

    // Top-p filtering: remove tokens from the tail until sum of probabilities <= top_p
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        float cum_sum = 0.0f;
        size_t cutoff = 0;
        for (; cutoff < candidates.size(); cutoff++) {
            cum_sum += candidates[cutoff].first;
            if (cum_sum > params.top_p) {
                cutoff++; // include current token
                break;
            }
        }
        if (cutoff < candidates.size()) {
            candidates.resize(cutoff);
        }
    }

    // min_p filtering: remove candidates with probability < min_p
    if (params.min_p > 0.0f && params.min_p < 1.0f) {
        std::vector<std::pair<float,int>> filtered;
        filtered.reserve(candidates.size());
        for (auto &c : candidates) {
            if (c.first >= params.min_p) {
                filtered.push_back(c);
            }
        }
        if (!filtered.empty()) {
            candidates = std::move(filtered);
        }
    }

    if (candidates.empty()) {
        // If we ended up with no candidates due to filtering, fallback to the top token
        // from the original sorted list (before top-p/min_p).
        l_llama_on_log(GGML_LOG_LEVEL_WARN, "All candidates removed by filtering. Falling back to highest logit token.", nullptr);
        candidates.push_back({1.0f, (int)0}); // top token from initial sort was candidates[0]
    }

    // Renormalize probabilities if we removed any candidates
    float new_sum = 0.0f;
    for (auto &c : candidates) {
        new_sum += c.first;
    }
    if (new_sum > 0.0f) {
        for (auto &c : candidates) {
            c.first /= new_sum;
        }
    } else {
        // If sum is 0 after filtering, fallback to highest probability token again
        l_llama_on_log(GGML_LOG_LEVEL_WARN, "Probability sum after filtering is 0. Falling back to first candidate.", nullptr);
        candidates.clear();
        candidates.push_back({1.0f, 0});
    }

    // Sample from distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int new_token_id = candidates[0].second;
    for (auto &c : candidates) {
        cumsum += c.first;
        if (r < cumsum) {
            new_token_id = c.second;
            break;
        }
    }

    DEBUG_PRINT("Selected token %d\n", new_token_id);

    if (new_token_id == llama_token_eos(model)) {
        DEBUG_PRINT("End of sequence token encountered\n");
        return nullptr;
    }

    // Get token string
    const char* token_str = llama_token_get_text(model, new_token_id);
    std::string cleaned;
    if (token_str) {
        cleaned = clean_token_text(token_str);
    } else {
        char piece[MAX_TOKEN_LEN];
        int token_len = llama_token_to_piece(model, new_token_id, piece, sizeof(piece)-1, 0, false);
        if (token_len <= 0) {
            l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Token conversion failed", nullptr);
            return nullptr;
        }
        piece[token_len] = '\0';
        cleaned = clean_token_text(piece);
    }

    tks_processed++;

    llama_batch_free(batch);
    batch = llama_batch_init(BATCH_SIZE, 0, 1);
    if (batch.n_tokens < BATCH_SIZE) {
        common_batch_add(batch, new_token_id, tks_processed - 1, {0}, true);
        if (batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = true;
        }
        if (llama_decode(ctx, batch) != 0) {
            l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Decode failed", nullptr);
            return nullptr;
        }
    } else {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Batch overflow after adding new token", nullptr);
        return nullptr;
    }

    char* ret = (char*)malloc(cleaned.size() + 1);
    if (!ret) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Memory allocation failed for token", nullptr);
        return nullptr;
    }
    strncpy(ret, cleaned.c_str(), cleaned.size() + 1);
    return ret;
}

extern "C" char* llama_run(int len) {
    if (len <= 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Invalid length", nullptr);
        return nullptr;
    }

    if (!ctx || !model) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Context or model not initialized", nullptr);
        return nullptr;
    }

    std::string response;
    response.reserve((size_t)len * MAX_TOKEN_LEN);

    for (int i = 0; i < len; i++) {
        char* next_token = llama_next();
        if (!next_token) {
            break;
        }
        response += next_token;
        free(next_token);
    }

    char* ret = (char*)malloc(response.size() + 1);
    if (!ret) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Memory allocation failed for response", nullptr);
        return nullptr;
    }
    strncpy(ret, response.c_str(), response.size()+1);
    return ret;
}

extern "C" int llama_add(char* new_string) {
    if (!new_string || !ctx) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Invalid input or context not initialized", nullptr);
        return 1;
    }

    std::string str(new_string);
    std::vector<llama_token> new_tokens_list = tokenize(ctx, str, true);
    if (new_tokens_list.empty()) {
        return 1;
    }

    if (isCtxFull()) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Context full, cannot add more tokens", nullptr);
        return 1;
    }

    for (size_t i = 0; i < new_tokens_list.size(); i++) {
        if (batch.n_tokens >= BATCH_SIZE) {
            l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Batch overflow while adding tokens", nullptr);
            return 1;
        }
        common_batch_add(batch, new_tokens_list[i], tks_processed, {0}, false);
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

extern "C" void llama_set_temperature(float temp) {
    if (temp >= 0.0f) {
        params.temperature = temp;
    }
}

extern "C" void llama_set_sampling_params(float temp, float top_p, int top_k, float repeat_penalty, int repeat_last_n, float min_p) {
    if (temp >= 0.0f) params.temperature = temp;
    if (top_p >= 0.0f && top_p <= 1.0f) params.top_p = top_p;
    if (top_k >= 0) params.top_k = top_k;
    if (repeat_penalty >= 0.0f) params.repeat_penalty = repeat_penalty;
    if (repeat_last_n >= 0) params.repeat_last_n = repeat_last_n;
    if (min_p >= 0.0f && min_p <= 1.0f) params.min_p = min_p;
}
