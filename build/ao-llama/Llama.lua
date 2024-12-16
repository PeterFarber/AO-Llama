local Llama = {}
Llama.backend = require("llama")

function Llama.info()
    return "A decentralized LLM inference engine, built on top of llama.cpp."
end

function Llama.load(id)
    Llama.backend.load(id)
end

function Llama.setPrompt(prompt)
    Llama.backend.set_prompt(prompt)
end

function Llama.setTemperature(temp)
    Llama.backend.set_temperature(temp)
end

function Llama.setSamplingParams(temp, top_p, top_k, repeat_penalty, repeat_last_n, min_p)
    Llama.backend.set_sampling_params(temp, top_p, top_k, repeat_penalty, repeat_last_n, min_p)
end

function Llama.postProcess(text)
    -- Many LLaMA-based models use "Ġ" to represent a space.
    -- Replace "Ġ" with a space.
    text = string.gsub(text, "Ġ", " ")

    -- Some models also use "▁" to represent a space or newline:
    text = string.gsub(text, "▁", " ")

    -- Handle "ĊĊ" which represents double newlines in some models
    text = string.gsub(text, "ĊĊ", "\n\n")
    -- Handle single "Ċ" as a newline
    text = string.gsub(text, "Ċ", "\n")

    -- If you see other odd sequences like "âĢĻ", handle them as shown before.
    text = string.gsub(text, "âĢĻ", "'")
    text = string.gsub(text, "âĢĶ", "...")
    -- Add more replacements as needed based on what characters appear in your output.

    -- Trim leading/trailing whitespace:
    text = string.gsub(text, "^%s+", "")
    text = string.gsub(text, "%s+$", "")

    return text
end

function Llama.run(count)
    return Llama.backend.run(count)
end

function Llama.next()
    return Llama.backend.next()
end

function Llama.add(str)
    Llama.backend.add(str)
end

function Llama.stop()
    Llama.backend.stop()
end

-- Callback handling functions

Llama.logLevels = {
    [2] = "error",
    [3] = "warn",
    [4] = "info",
    [5] = "debug",
}

Llama.logLevel = 5
Llama.logToStderr = true
Llama.log = {}

function Llama.onLog(level, str)
    if level <= Llama.logLevel then
        if Llama.logToStderr then
            io.stderr:write(Llama.logLevels[level] .. ": " .. str)
            io.stderr:flush()
        end
        if not Llama.log[Llama.logLevels[level]] then
            Llama.log[Llama.logLevels[level]] = {}
        end
        table.insert(Llama.log[Llama.logLevels[level]], str)
    end
end

function Llama.onProgress(str)
    io.stderr:write(".")
    io.stderr:flush()
end

_G.Llama = Llama

return Llama
