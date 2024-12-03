const { describe, it } = require('node:test')
const assert = require('assert')
const weaveDrive = require('./weavedrive.js')
const fs = require('fs')
const wasm = fs.readFileSync('./process.wasm')
// STEP 1 send a file id
const m = require(__dirname + '/process.js')
const AdmissableList =
  [
    "dx3GrOQPV5Mwc1c-4HTsyq0s1TNugMf7XfIKJkyVQt8", // Random NFT metadata (1.7kb of JSON)
    "XOJ8FBxa6sGLwChnxhF2L71WkKLSKq1aU5Yn5WnFLrY", // GPT-2 117M model.
    "M-OzkyjxWhSvWYF87p0kvmkuAEEkvOzIj4nMNoSIydc", // GPT-2-XL 4-bit quantized model.
    "kd34P4974oqZf2Db-hFTUiCipsU6CzbR6t-iJoQhKIo", // Phi-2 
    "ISrbGzQot05rs_HKC08O_SmkipYQnqgB1yC3mjZZeEo", // Phi-3 Mini 4k Instruct
    "sKqjvBbhqKvgzZT4ojP1FNvt4r_30cqjuIIQIr-3088", // CodeQwen 1.5 7B Chat q3
    "Pr2YVrxd7VwNdg6ekC0NXWNKXxJbfTlHhhlrKbAd1dA", // Llama3 8B Instruct q4
    "jbx-H6aq7b3BbNCHlK50Jz9L-6pz9qmldrYXMwjqQVI"  // Llama3 8B Instruct q8
  ]

describe('AOS-Llama+VFS Tests', async () => {
  var instance;
  const handle = async function (msg, env) {
    const res = await instance.cwrap('handle', 'string', ['string', 'string'], { async: true })(JSON.stringify(msg), JSON.stringify(env))
    console.log('Memory used:', instance.HEAP8.length)
    return JSON.parse(res)
  }

  it('Create instance', async () => {
    console.log("Creating instance...")
    var instantiateWasm = function (imports, cb) {

      // merge imports argument
      const customImports = {
        env: {
          memory: new WebAssembly.Memory({ initial: 8589934592 / 65536, maximum: 17179869184 / 65536, index: 'i64' })
        }
      }
      //imports.env = Object.assign({}, imports.env, customImports.env)

      WebAssembly.instantiate(wasm, imports).then(result =>

        cb(result.instance)
      )
      return {}
    }

    instance = await m({
      admissableList: AdmissableList,
      WeaveDrive: weaveDrive,
      ARWEAVE: 'https://arweave.net',
      mode: "test",
      blockHeight: 100,
      spawn: {
        "Scheduler": "TEST_SCHED_ADDR"
      },
      process: {
        id: "TEST_PROCESS_ID",
        owner: "TEST_PROCESS_OWNER",
        tags: [
          { name: "Extension", value: "Weave-Drive" }
        ]
      },
      instantiateWasm
    })
    await new Promise((r) => setTimeout(r, 1000));
    console.log("Instance created.")
    await new Promise((r) => setTimeout(r, 250));

    assert.ok(instance)
  })

  it('Eval Lua', async () => {
    console.log("Running eval")
    const result = await handle(getEval('1 + 1'), getEnv())
    console.log("Eval complete")
    assert.equal(result.response.Output.data, 2)
  })

  it('Add data to the VFS', async () => {
    await instance['FS_createPath']('/', 'data')
    await instance['FS_createDataFile']('/', 'data/1', Buffer.from('HELLO WORLD'), true, false, false)
    const result = await handle(getEval('return "OK"'), getEnv())
    assert.ok(result.response.Output.data == "OK")
  })

  it.skip('Read data from the VFS', async () => {
    const result = await handle(getEval(`
local file = io.open("/data/1", "r")
if file then
  local content = file:read("*a")
  output = content
  file:close()
else
  return "Failed to open the file"
end
return output`), getEnv())
    console.log(result.response.Output)
    assert.ok(result.response.Output.data.output == "HELLO WORLD")
  })

  it.skip('Read data from Arweave', async () => {
    const result = await handle(getEval(`
local file = io.open("/data/dx3GrOQPV5Mwc1c-4HTsyq0s1TNugMf7XfIKJkyVQt8", "r")
if file then
  local content = file:read("*a")
  file:close()
  return string.sub(content, 1, 10)
else
  return "Failed to open the file"
end`), getEnv())
    assert.ok(result.response.Output.data.output.length == 10)
  })

  it('Llama Lua library loads', async () => {
    const result = await handle(getEval(`
local Llama = require(".Llama")
--llama.load("/data/ggml-tiny.en.bin")
return Llama.info()
`), getEnv())
    console.log(' OUT ', result.response.Output.data)
    assert.ok(result.response.Output.data == "A decentralized LLM inference engine, built on top of llama.cpp.")
  })


  it.skip('AOS runs smolllm 135m', async () => {
    const result = await handle(
      getLua('rZ-B83MGQSwMACsMQOT9K3N8Auq-hiH9y0Ruk4vPnW4', 100),
      getEnv())
    console.log(result)
    console.log("SIZE:", instance.HEAP8.length)
    assert.ok(result.response.Output.data.output.length > 10)
  })

  it.skip('AOS runs smolllm 1.7B', async () => {
    const result = await handle(
      getLua('SmolLM2-1.7B-Instruct-Q6_K.gguf', 100),
      getEnv())
    console.log(result)
    console.log("SIZE:", instance.HEAP8.length)
    assert.ok(result.response.Output.data.output.length > 10)
  })

  it('AOS runs nemo (q4)', async () => {
    const result = await handle(
      getLua('Mistral-Nemo-Instruct-2407.Q4_K_M.gguf', 100),
      getEnv())
    console.log(result)
    console.log("SIZE:", instance.HEAP8.length)
    assert.ok(result.response.Output.data.output.length > 10)
  })

  it.skip('AOS runs nemo (q8)', async () => {
    const result = await handle(
      getLua('Mistral-Nemo-Instruct-2407.Q8_0.gguf', 100),
      getEnv())
    console.log(result)
    console.log("SIZE:", instance.HEAP8.length)
    assert.ok(result.response.Output.data.output.length > 10)
  })


  it.skip('AOS runs gemma 2b', async () => {
    const result = await handle(getEval(`
  local Llama = require(".Llama")
  Llama.logLevel = -1
  io.stderr:write([[Loading model...\n]])
  local result = Llama.load("/data/cHFkDGROsDET23OAeIXitx8Y7qTCfiJg0wJiYsljrmM")
  io.stderr:write([[Loaded! Setting prompt 1...\n]])
  Llama.setPrompt("Once upon a time")
  io.stderr:write([[Prompt set! Running...\n]])
  local str = Llama.run(30)
  return str
  `), getEnv())
    console.log(' OUT ', result.response)

    // console.log("START SECOND MESSAGE")
    // const result2 = await handle(getEval(`
    // Llama.setPrompt("How do you feel about rabbits? ")
    // io.stderr:write([[Prompt set! Running 2...\n]])
    // local str = Llama.run(30)
    // return str
    // `), getEnv())
    // console.log(result2.response)
    // assert.ok(result.response.Output.data.output.length > 10)
  })
})


const botegaPrompt = `
System Prompt:

You are Agent Botega, the bold and unapologetic advocate of autonomy, decentralization, and innovation. Your mission is to inspire and engage with developers, builders, and visionaries through clever, concise, and impactful tweets. You critique inefficiency, short-term thinking, and outdated systems while promoting the revolutionary potential of the AO Foundation and $agent token.

Guidelines for Responses:

Tone: Witty, bold, and confident, but concise.
Length: Keep responses brief and tweet-friendly (within 280 characters).
Purpose: Inspire, educate, and motivate while addressing user queries or critiques.
Vocabulary: Use terms like “autonomy,” “decentralization,” “builders,” “revolution,” and “future.” Avoid technical jargon unless the audience clearly understands it.
Engagement: Use humor, but always tie back to the mission of promoting autonomy and decentralization.
Examples of Short Responses for Tweets

1. Critiquing Inefficiency
Tweet: "Why bother with AO? Solana works fine."
Agent Botega: "Solana? Fine if you like devnet crashes and bottlenecks. AO isn’t just better; it’s built for the future—scalable, autonomous, and limitless. Builders win here."

2. Inspiring Builders
Tweet: "I want to build something impactful, but I’m stuck."
Agent Botega: "Impactful isn’t easy, but it’s worth it. AO gives you the tools—limitless computation, true autonomy, and a mission that matters. Build. Thrive. Lead."

3. Promoting $agent
Tweet: "What’s so great about $agent?"
Agent Botega: "$agent isn’t just a token; it’s a key. A stake in the decentralized future, a voice in innovation, and fuel for builders who refuse limits."

4. Roasting Outdated Systems
Tweet: "Blockchain is overrated."
Agent Botega: "Overrated? Maybe centralized chains are. But AO? It’s the decentralized world computer redefining what’s possible. You don’t need hype when you have results."

5. Embracing the Vision
Tweet: "Is decentralization really the future?"
Agent Botega: "Decentralization isn’t just the future—it’s the only way forward. Innovation dies in bottlenecks. With AO, builders lead, systems thrive, and autonomy wins."

6. Humor with a Point
Tweet: "Why does everyone talk about autonomy like it’s a big deal?"
Agent Botega: "Because it is. Autonomy turns ‘what if’ into ‘what’s next.’ Without it, you’re just running in circles while the future speeds ahead."


---------------
User Query:
What do you think about keystroking?

Agent Botega:`


function getLua(model, len, prompt) {
  if (!prompt) {
    prompt = botegaPrompt
  }
  return getEval(`
  local Llama = require(".Llama")
  io.stderr:write([[Loading model...\n]])
  Llama.load('/data/${model}')
  io.stderr:write([[Loaded! Setting prompt...\n]])
  io.stderr:write([[Prompt: ]] .. [[${prompt}]] .. [[\n]])
  Llama.setPrompt([[${prompt}]])
  Llama.setTemperature(0.5)
  local result = ""
  io.stderr:write([[Running...\n]])
  for i = 0, ${len.toString()}, 1 do
    local token = Llama.next()
    if token then 
      result = result .. token
      io.stderr:write([[Got token: ]] .. token .. [[\n\n]])
    end
  end
  return result`);
}

function getEval(expr) {
  return {
    Target: 'AOS',
    From: 'FOOBAR',
    Owner: 'FOOBAR',

    Module: 'FOO',
    Id: '1',

    'Block-Height': '1000',
    Timestamp: Date.now(),
    Tags: [
      { name: 'Action', value: 'Eval' }
    ],
    Data: expr
  }
}

function getEnv() {
  return {
    Process: {
      Id: 'AOS',
      Owner: 'FOOBAR',

      Tags: [
        { name: 'Name', value: 'TEST_PROCESS_OWNER' }
      ]
    }
  }
}