// llama lua wrappers

#include <stdlib.h>
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include "llama-run.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

extern lua_State *wasm_lua_state;

static int l_llama_load(lua_State *L) {
  const char* model_path = luaL_checkstring(L, 1);
  int result = llama_load((char*) model_path);
  lua_pushinteger(L, result);
  return 1;
}

// llama set prompt
static int l_llama_set_prompt(lua_State *L) {
  const char* prompt = luaL_checkstring(L, 1);
  llama_set_prompt((char*) prompt);
  return 0;
}

// llama add string to running session
static int l_llama_add(lua_State *L) {
  const char* prompt = luaL_checkstring(L, 1);
  llama_add((char*) prompt);
  return 0;
}

// Get a series of tokens as a string
static int l_llama_run(lua_State *L) {
  int tokens = luaL_checkinteger(L, 1);
  char* result = llama_run(tokens);
  lua_pushstring(L, result);
  free(result);
  return 1;
}

// Get the next token
static int l_llama_next(lua_State *L) {
  char* result = llama_next();
  lua_pushstring(L, result);
  free(result);
  return 1;
}

static int l_llama_stop(lua_State *L) {
  llama_stop();
  return 0;
}

// Add this with the other function declarations
static int l_llama_set_temperature(lua_State *L) {
    float temp = (float)luaL_checknumber(L, 1);
    llama_set_temperature(temp);
    return 0;
}

static int l_llama_set_sampling_params(lua_State *L) {
    float temp = (float)luaL_checknumber(L, 1);
    float top_p = (float)luaL_checknumber(L, 2);
    int top_k = (int)luaL_checkinteger(L, 3);
    float repeat_penalty = (float)luaL_checknumber(L, 4);
    int repeat_last_n = (int)luaL_checkinteger(L, 5);
    float min_p = (float)luaL_checknumber(L, 6);
    llama_set_sampling_params(temp, top_p, top_k, repeat_penalty, repeat_last_n, min_p);
    return 0;
}

// register function
int luaopen_llama(lua_State *L) {
  static const luaL_Reg llama_funcs[] = {
	  {"load", l_llama_load},
      {"set_prompt", l_llama_set_prompt},
      {"add", l_llama_add},
      {"run", l_llama_run},
      {"next", l_llama_next},
      {"stop", l_llama_stop},
      {"set_temperature", l_llama_set_temperature},
      {"set_sampling_params", l_llama_set_sampling_params},
      {NULL, NULL}  // Sentinel to indicate end of array
  };

  luaL_newlib(L, llama_funcs); // Create a new table and push the library function
  return 1;
}

// Handle callbacks in Lua
void l_llama_on_log(enum ggml_log_level level, const char * str, void* userdata) {
  lua_State *L = wasm_lua_state;

  lua_getglobal(L, "Llama");
  lua_getfield(L, -1, "onLog");
  lua_remove(L, -2); // Remove the llama table from the stack

  lua_pushinteger(L, level);
  lua_pushstring(L, str);
  lua_call(L, 2, 0);

  fflush(stderr);
}

bool l_llama_on_progress(float progress, void * user_data) {
  lua_State *L = wasm_lua_state;

  lua_getglobal(L, "Llama");
  lua_getfield(L, -1, "onProgress");
  lua_remove(L, -2); // Remove the llama table from the stack

  lua_pushnumber(L, progress);
  lua_call(L, 1, 0);
  return true;
}