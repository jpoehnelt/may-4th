<script lang="ts">
  import { onMount } from "svelte";
  import { YodaModel } from "../lib/inference";
  import { loadTokenizer, type ByteLevelTokenizer } from "../lib/tokenizer";

  let model: YodaModel | null = $state(null);
  let tokenizer: ByteLevelTokenizer | null = $state(null);
  let status = $state("Initializing...");
  let isLoading = $state(true);
  let isTranslating = $state(false);

  let inputText = $state("");
  let outputText = $state("");
  let outputTokens: string[] = $state([]);
  let showOutput = $state(false);
  let elapsed = $state(0);

  // Token details for visualization
  let inputTokenDetails: { id: number; text: string }[] = $state([]);
  let outputTokenDetails: { id: number; text: string }[] = $state([]);

  let modelInfo = $state<{ valBpb: number; step: number; params: number } | null>(null);

  // Example sentences to show as placeholders
  const examples = [
    "The code has many bugs.",
    "She ran through the forest quickly.",
    "The mountain is covered by snow.",
    "I will finish the project tomorrow.",
    "The cat sat on the warm windowsill.",
  ];
  let placeholder = $state(examples[0]);

  let loadError = $state("");

  onMount(async () => {
    // Rotate placeholder
    let idx = 0;
    const interval = setInterval(() => {
      idx = (idx + 1) % examples.length;
      if (!inputText) placeholder = examples[idx];
    }, 3000);

    try {
      const loadStart = performance.now();
      status = "Loading tokenizer...";
      tokenizer = await loadTokenizer("/models/tokenizer.json");

      status = "Loading AI model (~12MB)...";
      const m = new YodaModel();
      await m.load("/models", (msg) => {
        status = msg;
      });
      model = m;
      modelInfo = m.info;

      // Ensure loading state is visible for at least 500ms
      const loadMs = performance.now() - loadStart;
      if (loadMs < 500) await new Promise(r => setTimeout(r, 500 - loadMs));

      isLoading = false;
      status = "Ready";
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("[load] FAILED:", e);
      loadError = msg;
      status = `Error: ${msg}`;
      isLoading = false;
    }

    return () => clearInterval(interval);
  });

  async function translate() {
    const text = inputText.trim();
    if (!text || !model || !tokenizer || isTranslating) return;

    isTranslating = true;
    outputTokens = [];
    outputText = "";
    outputTokenDetails = [];
    showOutput = true;
    const start = performance.now();
    elapsed = 0;

    // Tokenize input for visualization
    const inputIds = tokenizer.encode(text);
    inputTokenDetails = inputIds.map(id => ({ id, text: tokenizer.decode([id]) }));

    try {
      const result = await model.translate(text, tokenizer, { maxTokens: 80, copyBias: 1.5 }, (token, idx) => {
        outputTokens = [...outputTokens, token];
        outputText = outputTokens.join("");
        elapsed = (performance.now() - start) / 1000;
      });
      // Build output token details from the final tokens
      const outputIds = tokenizer.encode(outputText);
      outputTokenDetails = outputIds.map(id => ({ id, text: tokenizer.decode([id]) }));
      elapsed = (performance.now() - start) / 1000;
    } catch (e) {
      console.error("[translate] error:", e);
      outputText = `Error: ${e instanceof Error ? e.message : String(e)}`;
    }

    isTranslating = false;
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      translate();
    }
  }

  function tryExample(text: string) {
    inputText = text;
    translate();
  }

  function clear() {
    inputText = "";
    outputText = "";
    outputTokens = [];
    outputTokenDetails = [];
    inputTokenDetails = [];
    showOutput = false;
    elapsed = 0;
    copied = false;
  }

  // Copy output to clipboard
  let copied = $state(false);
  async function copyOutput() {
    if (!outputText) return;
    await navigator.clipboard.writeText(outputText);
    copied = true;
    setTimeout(() => copied = false, 2000);
  }

  // Shared token IDs (appear in both input and output)
  let sharedTokenIds = $derived(
    new Set(
      inputTokenDetails
        .map(t => t.id)
        .filter(id => outputTokenDetails.some(t => t.id === id))
    )
  );

  // Deterministic color for a token ID
  function tokenColor(id: number): string {
    const hue = (id * 137.508) % 360; // golden angle
    return `hsl(${hue}, 55%, 65%)`;
  }

  function tokenBg(id: number, shared: boolean): string {
    const hue = (id * 137.508) % 360;
    return shared
      ? `hsla(${hue}, 55%, 65%, 0.25)`
      : `hsla(${hue}, 55%, 65%, 0.12)`;
  }
</script>

<div class="translator" id="translator">
  <!-- Status / Loading -->
  {#if isLoading}
    <div class="loading-state">
      <div class="loader-ring">
        <div class="lightsaber-spinner"></div>
      </div>
      <p class="status-text">{status}</p>
      <p class="status-sub">First load downloads the model to your browser</p>
    </div>
  {:else}
    <!-- Error banner -->
    {#if loadError}
      <div class="error-banner" id="load-error">
        <p>⚠️ {loadError}</p>
      </div>
    {/if}

    <!-- Input -->
    <div class="input-group">
      <label for="english-input" class="input-label">
        <span class="label-icon">⟩</span> English
      </label>
      <div class="input-wrapper">
        <textarea
          id="english-input"
          bind:value={inputText}
          onkeydown={handleKeydown}
          placeholder={placeholder}
          rows="2"
          disabled={isTranslating}
        ></textarea>
        <button
          class="translate-btn"
          onclick={translate}
          disabled={isTranslating || !inputText.trim()}
          id="translate-button"
        >
          {#if isTranslating}
            <span class="btn-spinner"></span>
          {:else}
            Translate
          {/if}
        </button>
        <button
          class="clear-btn"
          onclick={clear}
          disabled={isTranslating || (!inputText.trim() && !showOutput)}
          id="clear-button"
          title="Clear"
        >✕</button>
      </div>
    </div>

    <!-- Output -->
    {#if showOutput}
      <div class="output-group" class:translating={isTranslating}>
        <div class="output-header">
          <label class="output-label">
            <span class="label-icon yoda-icon">⟩</span> Yoda
          </label>
          {#if !isTranslating && outputText}
            <button class="copy-btn" onclick={copyOutput} title="Copy to clipboard">
              {copied ? '✓ Copied' : '⎘ Copy'}
            </button>
          {/if}
        </div>
        <div class="output-box">
          <p class="output-text" id="yoda-output">
            {#if outputText}
              {outputText}
              {#if isTranslating}<span class="cursor">▊</span>{/if}
            {:else if isTranslating}
              <span class="cursor">▊</span>
            {/if}
          </p>
          {#if !isTranslating && outputText}
            <div class="output-stats">
              <span>{outputTokens.length} tokens</span>
              <span>·</span>
              <span>{elapsed.toFixed(2)}s</span>
              <span>·</span>
              <span>100% client-side</span>
            </div>
          {/if}
        </div>

        <!-- Token visualization -->
        {#if !isTranslating && outputText}
          <div class="token-viz">
            <div class="token-row">
              <span class="token-row-label">Input</span>
              <div class="token-chips">
                {#each inputTokenDetails as tok}
                  {@const shared = sharedTokenIds.has(tok.id)}
                  <span
                    class="token-chip" class:shared
                    style="color: {tokenColor(tok.id)}; background: {tokenBg(tok.id, shared)}; border-color: {tokenColor(tok.id)}"
                    title="id: {tok.id}{shared ? ' (shared)' : ''}"
                  >{tok.text.replace(/ /g, '·')}</span>
                {/each}
              </div>
            </div>
            <div class="token-row">
              <span class="token-row-label">Output</span>
              <div class="token-chips">
                {#each outputTokenDetails as tok}
                  {@const shared = sharedTokenIds.has(tok.id)}
                  <span
                    class="token-chip" class:shared
                    style="color: {tokenColor(tok.id)}; background: {tokenBg(tok.id, shared)}; border-color: {tokenColor(tok.id)}"
                    title="id: {tok.id}{shared ? ' (shared)' : ''}"
                  >{tok.text.replace(/ /g, '·')}</span>
                {/each}
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Examples -->
    <div class="examples">
      <p class="examples-label">{showOutput ? 'Try another:' : 'Try an example:'}</p>
      <div class="examples-grid">
        {#each examples as example}
          <button class="example-btn" onclick={() => tryExample(example)} disabled={isTranslating}>
            "{example}"
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Model info footer -->
  {#if modelInfo}
    <div class="model-info">
      <span>val_bpb {modelInfo.valBpb.toFixed(4)}</span>
      <span>·</span>
      <span>{modelInfo.params >= 1000000 ? (modelInfo.params / 1000000).toFixed(1) + 'M' : (modelInfo.params / 1000).toFixed(0) + 'K'} params</span>
      {#if modelInfo.sizeMb}
        <span>·</span>
        <span>{modelInfo.sizeMb} MB</span>
      {/if}
      <span>·</span>
      <span>step {modelInfo.step.toLocaleString()}</span>
      <span>·</span>
      <span>runs in your browser</span>
    </div>
  {/if}
</div>

<style>
  .translator {
    max-width: 840px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  /* Loading state */
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    padding: 3rem 1rem;
  }

  .loader-ring {
    width: 48px;
    height: 48px;
    position: relative;
  }

  .lightsaber-spinner {
    width: 48px;
    height: 48px;
    border: 3px solid rgba(74, 222, 128, 0.15);
    border-top-color: #4ade80;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .status-text {
    color: #4ade80;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }

  .status-sub {
    color: #94a3b8;
    font-size: 0.9rem;
  }

  /* Input group */
  .input-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .input-label, .output-label {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #cbd5e1;
  }

  .label-icon {
    color: #3b82f6;
    margin-right: 0.25rem;
  }

  .yoda-icon {
    color: #4ade80;
  }

  .input-wrapper {
    display: flex;
    gap: 0.5rem;
    align-items: stretch;
  }

  textarea {
    flex: 1;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #f8fafc;
    font-size: 1.35rem;
    font-weight: 500;
    font-family: inherit;
    resize: none;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    line-height: 1.6;
  }

  textarea:focus {
    border-color: rgba(59, 130, 246, 0.5);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  textarea::placeholder {
    color: #64748b;
    font-style: italic;
  }

  textarea:disabled {
    opacity: 0.5;
  }

  .translate-btn {
    background: linear-gradient(135deg, #4ade80, #22c55e);
    color: #022c22;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 1.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
    min-width: 110px;
    display: flex;
    align-items: center;
    justify-content: center;
    letter-spacing: 0.02em;
  }

  .translate-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(74, 222, 128, 0.3);
  }

  .translate-btn:active:not(:disabled) {
    transform: translateY(0);
  }

  .translate-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .clear-btn {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    width: 44px;
    font-size: 1.25rem;
    color: #64748b;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .clear-btn:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.08);
    color: #e2e8f0;
    border-color: rgba(255, 255, 255, 0.15);
  }

  .clear-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .btn-spinner {
    width: 18px;
    height: 18px;
    border: 2px solid rgba(0, 0, 0, 0.2);
    border-top-color: #022c22;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }

  /* Output group */
  .output-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    animation: fadeIn 0.3s ease;
  }

  .output-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .copy-btn {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    color: #94a3b8;
    font-size: 0.85rem;
    font-family: inherit;
    cursor: pointer;
    transition: all 0.2s;
  }

  .copy-btn:hover {
    background: rgba(74, 222, 128, 0.1);
    border-color: rgba(74, 222, 128, 0.3);
    color: #4ade80;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .output-box {
    background: rgba(74, 222, 128, 0.04);
    border: 1px solid rgba(74, 222, 128, 0.12);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    min-height: 3rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .translating .output-box {
    border-color: rgba(74, 222, 128, 0.25);
    box-shadow: 0 0 20px rgba(74, 222, 128, 0.05);
  }

  .output-text {
    color: #4ade80;
    font-size: 1.4rem;
    line-height: 1.6;
    margin: 0;
    font-weight: 500;
  }

  .cursor {
    color: #4ade80;
    animation: blink 0.6s step-end infinite;
  }

  @keyframes blink {
    50% { opacity: 0; }
  }

  .output-stats {
    display: flex;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: #475569;
    padding-top: 0.25rem;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
  }

  /* Examples */
  .examples {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .examples-label {
    font-size: 0.8rem;
    color: #64748b;
    margin: 0;
  }

  .examples-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .example-btn {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    color: #94a3b8;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s;
    font-style: italic;
  }

  .example-btn:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(74, 222, 128, 0.2);
    color: #cbd5e1;
  }

  /* Model info footer */
  .model-info {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 1rem;
    flex-wrap: wrap;
  }
  /* Token toggle */
  .token-toggle {
    background: none;
    border: none;
    color: #4ade80;
    cursor: pointer;
    font-size: 0.75rem;
    padding: 0;
    text-decoration: underline;
    text-underline-offset: 2px;
    transition: opacity 0.2s;
  }

  /* Token visualization */
  .token-viz {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.75rem 0;
    animation: fadeIn 0.3s ease;
  }

  .token-row {
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
  }

  .token-row-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #475569;
    min-width: 3.5rem;
    padding-top: 0.3rem;
    flex-shrink: 0;
  }

  .token-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
  }

  .token-chip {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.85rem;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    border: 1px solid;
    white-space: pre;
    line-height: 1.3;
    cursor: default;
    transition: transform 0.15s;
  }

  .token-chip:hover {
    transform: scale(1.08);
    z-index: 1;
  }

  .token-chip.shared {
    box-shadow: 0 0 6px currentColor;
    border-width: 1.5px;
  }
</style>
