/**
 * ONNX Runtime Web inference for the Kaminoan encoder-decoder model.
 *
 * Loads encoder.onnx + decoder.onnx and runs autoregressive decoding
 * with cross-attention from decoder to encoder hidden states.
 */

import * as ort from "onnxruntime-web/wasm";
import type { ByteLevelTokenizer } from "./tokenizer";

// Configure ONNX Runtime — load WASM backend files from CDN
// (Vite blocks importing .mjs files from public/, so we use CDN instead)
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/";

interface ModelMeta {
  val_bpb: number;
  step: number;
  params: number;
  size_mb?: number;
  vocab_size: number;
  d_model: number;
  max_src_len: number;
  max_tgt_len: number;
  eos_id: number;
  pad_id: number;
  bos_id: number;
}

export interface ModelInfo {
  valBpb: number;
  step: number;
  params: number;
  sizeMb?: number;
}

export class YodaModel {
  private encoderSession: ort.InferenceSession | null = null;
  private decoderSession: ort.InferenceSession | null = null;
  private meta: ModelMeta | null = null;

  public info: ModelInfo | null = null;

  /**
   * Load both ONNX sessions and model metadata.
   * Calls onProgress with status messages.
   */
  async load(
    basePath = "/models",
    onProgress?: (msg: string) => void
  ): Promise<void> {
    onProgress?.("Loading model metadata...");
    const metaResp = await fetch(`${basePath}/model_meta.json`);
    this.meta = await metaResp.json();

    const sizeStr = this.meta.size_mb ? ` (~${(this.meta.size_mb / 2).toFixed(1)}MB)` : "";
    
    onProgress?.(`Loading encoder${sizeStr}...`);
    this.encoderSession = await ort.InferenceSession.create(
      `${basePath}/encoder.onnx`,
      {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }
    );

    onProgress?.(`Loading decoder${sizeStr}...`);
    this.decoderSession = await ort.InferenceSession.create(
      `${basePath}/decoder.onnx`,
      {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }
    );

    this.info = {
      valBpb: this.meta!.val_bpb,
      step: this.meta!.step,
      params: this.meta!.params,
      sizeMb: this.meta!.size_mb,
    };

    onProgress?.("Model ready!");
  }

  get isLoaded(): boolean {
    return (
      this.encoderSession !== null &&
      this.decoderSession !== null &&
      this.meta !== null
    );
  }

  /**
   * Translate English text to Yoda-speak.
   *
   * @param text     - English input text
   * @param tokenizer - ByteLevelTokenizer instance
   * @param options  - maxTokens, copyBias
   * @param onToken  - called with each generated token string for streaming
   * @returns The full Yoda translation
   */
  async translate(
    text: string,
    tokenizer: ByteLevelTokenizer,
    options: { maxTokens?: number; copyBias?: number } = {},
    onToken?: (token: string, idx: number) => void
  ): Promise<string> {
    if (!this.isLoaded || !this.meta) {
      throw new Error("Model not loaded. Call load() first.");
    }

    let { maxTokens = 80, copyBias = 1.5 } = options;
    const { eos_id, bos_id } = this.meta;

    // Prevent out-of-bounds index on the positional embedding layer
    maxTokens = Math.min(maxTokens, this.meta.max_tgt_len - 1);

    // Encode source text
    const srcIds = tokenizer.encode(text);
    srcIds.push(eos_id);

    // Clamp to max source length
    const clampedSrc = srcIds.slice(0, this.meta.max_src_len);

    // Run encoder
    const srcTensor = new ort.Tensor(
      "int64",
      BigInt64Array.from(clampedSrc.map(BigInt)),
      [1, clampedSrc.length]
    );

    const encResult = await this.encoderSession!.run({ src_ids: srcTensor });
    const encOut = encResult["enc_out"];
    const srcPadMask = encResult["src_pad_mask"];

    // Collect source token IDs for copy bias
    const srcTokenSet = new Set(clampedSrc);

    // Autoregressive decoding
    const generatedIds: number[] = [];
    let tgtIds = [bos_id];

    for (let i = 0; i < maxTokens; i++) {
      const tgtTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(tgtIds.map(BigInt)),
        [1, tgtIds.length]
      );

      const decResult = await this.decoderSession!.run({
        tgt_ids: tgtTensor,
        enc_out: encOut,
        src_pad_mask: srcPadMask,
      });

      // Get logits for the last position
      const logits = decResult["logits"];
      const logitsData = logits.data as Float32Array;
      const vocabSize = this.meta!.vocab_size;
      const lastOffset = (tgtIds.length - 1) * vocabSize;

      // Apply copy bias — boost tokens present in the source
      if (copyBias > 0) {
        for (const tokenId of srcTokenSet) {
          if (tokenId < vocabSize) {
            logitsData[lastOffset + tokenId] += copyBias;
          }
        }
      }

      // Greedy: argmax over last position
      let maxVal = -Infinity;
      let maxIdx = 0;
      for (let j = 0; j < vocabSize; j++) {
        const val = logitsData[lastOffset + j];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = j;
        }
      }

      // Check for EOS
      if (maxIdx === eos_id) break;

      generatedIds.push(maxIdx);
      tgtIds.push(maxIdx);

      // Stream token callback
      const tokenStr = tokenizer.decode([maxIdx]);
      onToken?.(tokenStr, i);
    }

    return tokenizer.decode(generatedIds);
  }
}
