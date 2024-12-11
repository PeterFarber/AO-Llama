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

#define CTX_SIZE 8192
#define BATCH_SIZE 2048
#define MAX_TOKEN_LEN 256
#define DEBUG_MODE 0

#define DEBUG_PRINT(...) do { if (DEBUG_MODE) fprintf(stderr, __VA_ARGS__); } while (0)

struct params_struct {
    std::string model;
    std::string prompt;
    int n_threads = 1;
    int n_threads_batch = 1;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int top_k = 40;
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    float min_p = 0.05f;
};

// Global variables
static params_struct params;
static llama_model* model = nullptr;
static llama_batch batch = {};
static llama_context* ctx = nullptr;
static int tks_processed = 0;

extern "C" bool l_llama_on_progress(float progress, void* user_data);
extern "C" void l_llama_on_log(enum ggml_log_level level, const char* text, void* user_data);

// Simplify TokenMapping to just handle basic whitespace cases
struct TokenMapping {
    llama_token token_id;
    std::string raw_text;
    std::string replacement;
};

// Global vector for special tokens
static std::vector<TokenMapping> special_tokens;

void initialize_special_tokens(llama_model* model) {
    special_tokens.clear();
    
    // Get vocabulary size
    const int n_vocab = llama_n_vocab(model);
    
    // Only handle the most common special tokens
    const std::vector<std::pair<std::string, std::string>> token_mappings = {
        {"Ġ", " "},     // Space at start of word
        {"▁", " "},     // Alternative space marker
        {"Ċ", "\n"},    // Newline
        {"ĉ", "\n"},    // Alternative newline
        {"<0x0A>", "\n"}, // Hex newline
        {"<0x20>", " "}, // Hex space
    };

    // Scan vocabulary for special tokens
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        const char* token_str = llama_token_get_text(model, token_id);
        if (!token_str) continue;
        
        std::string token_text(token_str);
        
        // Check for exact matches in our mappings
        for (const auto& [pattern, replacement] : token_mappings) {
            if (token_text == pattern) {  // Only exact matches
                special_tokens.push_back({token_id, token_text, replacement});
                DEBUG_PRINT("Found special token: '%s' (id: %d) -> '%s'\n", 
                        token_text.c_str(), token_id, replacement.c_str());
                break;
            }
        }
    }

    DEBUG_PRINT("Initialized %zu special tokens\n", special_tokens.size());
}

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

    // Initialize special tokens
    initialize_special_tokens(model);
    fprintf(stderr, "Initialized %zu special tokens\n", special_tokens.size());
    
    // Add vocabulary size check
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
    return ctx && tks_processed >= llama_n_ctx(ctx);
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

    // Tokenize without BOS token since we'll add it manually
    std::vector<llama_token> tokens_list = tokenize(ctx, params.prompt, false);
    if (tokens_list.empty()) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Tokenization failed", nullptr);
        return 1;
    }

    // Process BOS token first
    common_batch_clear(batch);
    common_batch_add(batch, llama_token_bos(model), 0, {0}, false);
    
    if (llama_decode(ctx, batch) != 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Decode failed for BOS token", nullptr);
        return 1;
    }
    tks_processed++;

    // Process prompt tokens in chunks
    const size_t chunk_size = BATCH_SIZE - 1;  // Leave room for safety
    for (size_t i = 0; i < tokens_list.size(); i += chunk_size) {
        common_batch_clear(batch);
        
        // Calculate size of current chunk
        size_t current_chunk_size = std::min(chunk_size, tokens_list.size() - i);
        
        // Add tokens for this chunk
        for (size_t j = 0; j < current_chunk_size; j++) {
            bool is_last = (i + j == tokens_list.size() - 1);
            common_batch_add(batch, tokens_list[i + j], tks_processed++, {0}, is_last);
        }

        // Set logits only for the last token of the final chunk
        if (i + current_chunk_size >= tokens_list.size()) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        // Decode this chunk
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

    // Add debug logging after final decode
    if (batch.n_tokens > 0) {
        float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (logits) {
            float max_logit = -INFINITY;
            for (int i = 0; i < llama_n_vocab(model); i++) {
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
    std::string result;
    
    // Expanded list of tokens to skip
    if (text == "<|endoftext|>" || text == "<s>" || text == "" || 
        text == "<pad>" || text == "<unk>" || text == "âĢĶ" ||
        text == "ðŁĮĲðŁĴ" || text == "ðŁ" || text == "Ĳ" || 
        text == "Ĵ" || text == "Į") {
        return "";
    }

    // First check for exact matches in special_tokens
    for (const auto& special : special_tokens) {
        if (text == special.raw_text) {
            return special.replacement;
        }
    }
    
    // Process the text character by character
    for (size_t i = 0; i < text.length();) {
        bool handled = false;

        // Check for multi-byte special characters
        for (const auto& special : special_tokens) {
            if (text.compare(i, special.raw_text.length(), special.raw_text) == 0) {
                result += special.replacement;
                i += special.raw_text.length();
                handled = true;
                break;
            }
        }

        if (handled) continue;

        // Handle emoji and other Unicode sequences
        if ((unsigned char)text[i] == 0xF0 && i + 3 < text.length()) {
            // Skip 4-byte Unicode sequences (including emojis)
            i += 4;
            continue;
        }

        // Handle remaining special cases
        if ((unsigned char)text[i] == 0xC4 && i + 1 < text.length()) {
            if ((unsigned char)text[i + 1] == 0xA0) {  // Ġ
                result += ' ';
                i += 2;
                continue;
            } else if ((unsigned char)text[i + 1] == 0x82) {  // Ċ
                result += '\n';
                i += 2;
                continue;
            } else if ((unsigned char)text[i + 1] == 0xAE ||  // Į
                      (unsigned char)text[i + 1] == 0xB2 ||  // Ĳ
                      (unsigned char)text[i + 1] == 0xB4) {  // Ĵ
                i += 2;
                continue;
            }
        }
        
        // Handle ▁ (LOWER ONE EIGHTH BLOCK)
        if ((unsigned char)text[i] == 0xE2 && i + 2 < text.length() && 
            (unsigned char)text[i + 1] == 0x96 && (unsigned char)text[i + 2] == 0x81) {
            result += ' ';
            i += 3;
            continue;
        }
        
        // Handle âĢĶ (horizontal ellipsis)
        if ((unsigned char)text[i] == 0xC3 && i + 2 < text.length() && 
            (unsigned char)text[i + 1] == 0xA2 && (unsigned char)text[i + 2] == 0xC5) {
            result += "...";
            i += 3;
            continue;
        }

        // Handle ðŁ sequences
        if ((unsigned char)text[i] == 0xC3 && i + 1 < text.length() && 
            (unsigned char)text[i + 1] == 0xB0) {
            i += 2;  // Skip ðŁ
            continue;
        }
        
        // Only add printable ASCII characters
        if ((unsigned char)text[i] >= 32 && (unsigned char)text[i] <= 126) {
            result += text[i];
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

    // Get logits for the last token
    float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    if (!logits) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Failed to get logits", nullptr);
        return nullptr;
    }

    const int n_vocab = llama_n_vocab(model);
    
    // Apply repetition penalty
    if (params.repeat_penalty != 1.0f && params.repeat_last_n > 0) {
        std::vector<llama_token> last_tokens(params.repeat_last_n);
        int n_tokens = std::min(params.repeat_last_n, tks_processed);
        
        for (int i = 0; i < n_tokens; i++) {
            last_tokens[i] = batch.token[batch.n_tokens - n_tokens + i];
        }
        
        for (int i = 0; i < n_tokens; i++) {
            logits[last_tokens[i]] /= params.repeat_penalty;
        }
    }

    // Apply temperature
    if (params.temperature > 0) {
        for (int i = 0; i < n_vocab; i++) {
            logits[i] /= params.temperature;
        }
    }

    // Apply top_k sampling
    if (params.top_k > 0) {
        std::vector<std::pair<float, llama_token>> candidates;
        candidates.reserve(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            candidates.push_back(std::make_pair(logits[i], i));
        }
        
        std::partial_sort(candidates.begin(), 
                         candidates.begin() + params.top_k, 
                         candidates.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        for (int i = params.top_k; i < n_vocab; i++) {
            logits[candidates[i].second] = -INFINITY;
        }
    }

    // Apply top_p sampling (nucleus sampling)
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        std::vector<std::pair<float, llama_token>> candidates;
        candidates.reserve(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            candidates.push_back(std::make_pair(logits[i], i));
        }
        
        std::sort(candidates.begin(), candidates.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Calculate max_logit first
        float max_logit = candidates[0].first;
        
        float cumsum = 0.0f;
        for (int i = 0; i < n_vocab; i++) {
            cumsum += expf(candidates[i].first - max_logit);
            if (cumsum > params.top_p) {
                for (int j = i + 1; j < n_vocab; j++) {
                    logits[candidates[j].second] = -INFINITY;
                }
                break;
            }
        }
    }

    // After top_p sampling, add min_p:
    if (params.min_p > 0.0f) {
        float max_l = -INFINITY;
        for (int i = 0; i < n_vocab; i++) {
            if (logits[i] > max_l) {
                max_l = logits[i];
            }
        }
        
        const float min_logit = max_l - logf(1.0f / params.min_p);
        for (int i = 0; i < n_vocab; i++) {
            if (logits[i] < min_logit) {
                logits[i] = -INFINITY;
            }
        }
    }

    // Convert logits to probabilities with softmax
    float max_logit = -INFINITY;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    std::vector<float> probs(n_vocab);
    float sum = 0.0f;
    
    for (int i = 0; i < n_vocab; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }

    // Normalize probabilities
    for (int i = 0; i < n_vocab; i++) {
        probs[i] /= sum;
    }

    // Sample from distribution
    float r = (float)rand() / (float)RAND_MAX;
    llama_token new_token_id = 0;
    float cumsum = 0.0f;

    for (int i = 0; i < n_vocab; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            new_token_id = i;
            break;
        }
    }

    DEBUG_PRINT("First 5 logits: %f, %f, %f, %f, %f\n", 
            logits[0], logits[1], logits[2], logits[3], logits[4]);
    DEBUG_PRINT("Selected token %d with probability %f\n", new_token_id, probs[new_token_id]);

    // Check if the token is an EOS token
    if (new_token_id == llama_token_eos(model)) {
        DEBUG_PRINT("End of sequence token encountered\n");
        return nullptr;
    }

    char* token = (char*)malloc(MAX_TOKEN_LEN);
    if (!token) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Failed to allocate token memory", nullptr);
        return nullptr;
    }

    // Try getting the token string directly from llama
    const char* token_str = llama_token_get_text(model, new_token_id);
    if (token_str) {
        DEBUG_PRINT("Raw token text: '%s'\n", token_str);
        std::string cleaned = clean_token_text(token_str);
        DEBUG_PRINT("Cleaned token text: '%s'\n", cleaned.c_str());
        strncpy(token, cleaned.c_str(), MAX_TOKEN_LEN - 1);
        token[MAX_TOKEN_LEN - 1] = '\0';
    } else {
        // Fallback to token_to_piece if get_text fails
        char piece[MAX_TOKEN_LEN];
        int token_len = llama_token_to_piece(model, new_token_id, piece, sizeof(piece) - 1, 0, false);
        
        DEBUG_PRINT("Token to piece conversion returned length: %d\n", token_len);
        
        if (token_len <= 0) {
            l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Token conversion failed", nullptr);
            free(token);
            return nullptr;
        }
        
        piece[token_len] = '\0';
        std::string cleaned = clean_token_text(piece);
        strncpy(token, cleaned.c_str(), MAX_TOKEN_LEN - 1);
        token[MAX_TOKEN_LEN - 1] = '\0';
    }

    tks_processed++;

    // Prepare next batch
    llama_batch_free(batch);
    batch = llama_batch_init(BATCH_SIZE, 0, 1);
    common_batch_add(batch, new_token_id, tks_processed - 1, {0}, true);
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        l_llama_on_log(GGML_LOG_LEVEL_ERROR, "Decode failed", nullptr);
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