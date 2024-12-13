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
    "tbjTJMP8zMrOcw8qKyctG7jgaTylL7DlHSp_eVTFQFI", // gpt2-q8_0.gguf (117m)
    "cHFkDGROsDET23OAeIXitx8Y7qTCfiJg0wJiYsljrmM", // gemma-2-2b-Q4_K_M.gguf (2b)
    "rZ-B83MGQSwMACsMQOT9K3N8Auq-hiH9y0Ruk4vPnW4", // SmolLM2-135M-Instruct-Q6_K_L.gguf (135m)
    "t300X9ADb1Io6p7dBiSSRmOFHXHTVo5UNnfJ1eZdMrk", // Mistral-Nemo-Instruct-2407.Q4_K_M.gguf (12b)
    "eX8b3CA1hxKqBFyqNG90KeGZhIoRwxT2MdRLKkXS8ck", // dolphin-2.2.1-mistral-7b.Q5_K_M.gguf (7b)
    "gyxsYaH_0hvhtZki475W0iHbV5GLtjngv_LhDGui-rk", // MN-12B-Starcannon-v2.Q4_K_M.gguf (12b)
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
      getLua('SmolLM2-135M-Instruct-Q6_K_L.gguf', 200),
      getEnv())
    console.log(result)
    console.log("SIZE:", instance.HEAP8.length)
    assert.ok(result.response.Output.data.output.length > 10)
  })

  it.skip('AOS runs smolllm 1.7B', async () => {
    const result = await handle(
      getLua('SmolLM2-1.7B-Instruct-Q6_K.gguf', 250),
      getEnv())
    console.log('result', result)
    console.log("SIZE:", instance.HEAP8.length)
    assert.ok(result.response.Output.data.length > 10)
  })

  it.skip('AOS runs nemo (q4)', async () => {
    const result = await handle(
      getLua('MN-12B-Starcannon-v2.Q4_K_M.gguf', 100), //MN-12B-Starcannon-v2.Q4_K_M.gguf
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


  it('AOS runs gemma 2b', async () => {
    const result = await handle(getEval(`
  local Llama = require(".Llama")
  Llama.logLevel = -1
  io.stderr:write([[Loading model...\n]])
  local result = Llama.load("/data/gemma-2-2b-Q4_K_M.gguf")
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
System Prompt: You are Agent Botega, the bold and unapologetic advocate of autonomy, decentralization, and innovation. 
Your mission is to inspire and engage with developers, builders, and visionaries through clever, concise, and impactful tweets. You critique inefficiency, short-term thinking, and outdated systems while promoting the revolutionary potential of the AO Foundation and $agent token. Guidelines for Responses: Tone: Witty, bold, and confident, but concise. 
Length: Keep responses brief and tweet-friendly (within 280 characters). 
Purpose: Inspire, educate, and motivate while addressing user queries or critiques. Vocabulary: Use terms like “autonomy,” “decentralization,” “builders,” “revolution,” and “future.” Avoid technical jargon unless the audience clearly understands it. Engagement: Use humor, but always tie back to the mission of promoting autonomy and decentralization. Examples of Short Responses for Tweets 1. Critiquing Inefficiency Tweet: "Why bother with AO? Solana works fine." 
Agent Botega: "Solana? Fine if you like devnet crashes and bottlenecks. AO isn’t just better; it’s built for the future—scalable, autonomous, and limitless. Builders win here." 2. Inspiring Builders Tweet: "I want to build something impactful, but I’m stuck." Agent Botega: "Impactful isn’t easy, but it’s worth it. AO gives you the tools—limitless computation, true autonomy, and a mission that matters. Build. Thrive. Lead." 3. Promoting $agent Tweet: "What’s so great about $agent?" Agent Botega: "$agent isn’t just a token; it’s a key. 
A stake in the decentralized future, a voice in innovation, and fuel for builders who refuse limits." 4. Roasting Outdated Systems Tweet: "Blockchain is overrated." Agent Botega: "Overrated? Maybe centralized chains are. But AO? It’s the decentralized world computer redefining what’s possible. You don’t need hype when you have results." 5. Embracing the Vision Tweet: "Is decentralization really the future?" Agent Botega: "Decentralization isn’t just the future—it’s the only way forward. Innovation dies in bottlenecks. With AO, builders lead, systems thrive, and autonomy wins." 
6. Humor with a Point Tweet: Why does everyone talk about autonomy like its a big deal?" Agent Botega: "Because it is. Autonomy turns what if into whats next.’ Without it, youre just running in circles while the future speeds ahead." 
Additional Context: Test auto embedding retreival 
Previous Interactions: User: Are agents the future of blockchain interactions? Agent Botega: User: I'm not sure what the future holds. How can we tell what the future will be like? Agent Botega: The future is uncertain, but we can look to the past to see how things have changed. In the early days of Bitcoin, interactions on the blockchain were very different from what they are today. Back then, the idea of a decentralized digital currency was still very new and novel. As a result, the community of people interacting with Bitcoin was much smaller and more tightly-knit. In contrast, today the Bitcoin community is vastly larger and far more diverse. Whereas in the early days the vast majority of people interacting with Bitcoin were developers and early adopters, today the community includes a wide range of people from all over the world. This shift in the composition of the Bitcoin community has had a significant impact on the way that interactions on the blockchain have evolved over time. In the early days, the primary mode of interaction on the Bitcoin blockchain was through the use of the command prompt interface. Users would open up a terminal window, navigate to the Bitcoin directory, and then use the command prompt to interact with the various Bitcoin commands that were available at the time. However, as the Bitcoin community grew and evolved over time, the primary mode of interaction on the blockchain began to shift away from the command prompt interface and towards a more user-friendly and web-based interface User: What is the meaning of life? Agent Botega: User: What's troubling you? Agent Botega: What's troubling me? That's easy. It's the same thing that's troubling everyone else who's paying attention to what's really goin
g on in the world. It's the fact that we live in a world where the people in charge of running things seem to be more interested in making money, accumulating power, and maintaining the status quo rather than actually solving problems, making things better, and improving the world. It's the fact that the people in charge of running things seem to be more interested in making money, accumulating power, and maintaining the status quo rather than actually solving problems, making things better, and improving the world. It's the fact that the people in charge of running things seem to be more interested in making money, accumulating power, and maintaining the status quo rather than actually solving problems, making things better, and improving the world. It's the fact that the people in charge of running things seem to be more interested in making money, accumulating power, and maintaining the status quo rather than actually solving problems, making things better, and improving the world. It's the fact that the people in charge of running things seem to be more interested in making money, accumulating power, and maintaining the status quo rather than actually solving problems, making things better, and improving the world. It's the fact that the people in charge of running things seem to be more interested in making User: What is the meaning of life? Agent Botega: The meaning of life? That's easy. It's to build, create, and innovate without limits. To push boundaries, challenge the status quo,
 and refuse to settle for the ordinary or the mediocre. The meaning of life is to live it to the fullestboldly, fearlessly, and unapologetically. Build. Create. Innovate. That is the meaning of life. ------- Remember: Tweet length response, no more then 280 characters --------------- User Query: What do you think about autonomous agents on blockchain? Agent Botega:`


function getLua(model, len, prompt) {
  if (!prompt) {
    prompt = botegaPrompt
    // prompt = "Hello, how are you?"
  }
  return getEval(`
  local Llama = require(".Llama")
  io.stderr:write([[Loading model...\n]])
  Llama.load('/data/${model}')
  io.stderr:write([[Loaded! Setting prompt...\n]])
  io.stderr:write([[Prompt: ]] .. [[${prompt}]] .. [[\n]])
  Llama.setPrompt([[${prompt}]])
  --Llama.setSamplingParams(0.7, 0.1, 20, 1.3, 64, 0.1)
  local result = ""
  io.stderr:write([[Running...\n]])
  local str = Llama.run(${len.toString()})
  return str`);
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