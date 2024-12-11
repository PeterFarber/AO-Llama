#ifndef LLAMA_RUN_H
#define LLAMA_RUN_H
#include <stdbool.h>
#include "llama.h"

int luaopen_llama(lua_State *L);

int llama_load(char* model_path);
int llama_set_prompt(char* prompt);
char* llama_run(int len);
char* llama_next();
int llama_add(char* new_string);
void llama_stop();
void llama_set_temperature(float temp);
void llama_set_sampling_params(float temp, float top_p, int top_k, float repeat_penalty, int repeat_last_n, float min_p);

#endif // LLAMA_RUN_H

