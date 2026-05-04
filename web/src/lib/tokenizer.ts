/**
 * Byte-level BPE tokenizer — TypeScript port of prepare.py's ByteLevelTokenizer.
 * Loads merge rules from tokenizer.json and encodes/decodes text.
 */

interface TokenizerData {
  vocab_size: number;
  merges: Record<string, number>;
}

export class ByteLevelTokenizer {
  private merges: Map<string, number> = new Map();
  private vocab: Map<number, Uint8Array> = new Map();
  public vocabSize: number = 0;

  /**
   * Load tokenizer from a parsed JSON object (the contents of tokenizer.json).
   */
  load(data: TokenizerData): void {
    this.vocabSize = data.vocab_size;

    // Build base vocab: 256 byte-level tokens
    for (let i = 0; i < 256; i++) {
      this.vocab.set(i, new Uint8Array([i]));
    }

    // Load merges
    this.merges.clear();
    for (const [key, value] of Object.entries(data.merges)) {
      this.merges.set(key, value);
      const [a, b] = key.split(",").map(Number);
      const aBytes = this.vocab.get(a)!;
      const bBytes = this.vocab.get(b)!;
      const merged = new Uint8Array(aBytes.length + bBytes.length);
      merged.set(aBytes);
      merged.set(bBytes, aBytes.length);
      this.vocab.set(value, merged);
    }
  }

  /**
   * Encode a string to an array of token IDs.
   */
  encode(text: string): number[] {
    const encoder = new TextEncoder();
    let ids: number[] = Array.from(encoder.encode(text));

    while (true) {
      // Find the pair with the lowest merge index
      let bestPair: string | null = null;
      let bestIdx = Infinity;

      for (let i = 0; i < ids.length - 1; i++) {
        const pair = `${ids[i]},${ids[i + 1]}`;
        const mergeIdx = this.merges.get(pair);
        if (mergeIdx !== undefined && mergeIdx < bestIdx) {
          bestPair = pair;
          bestIdx = mergeIdx;
        }
      }

      if (bestPair === null) break;

      // Merge all occurrences of the best pair
      const newIds: number[] = [];
      const [a, b] = bestPair.split(",").map(Number);
      let i = 0;
      while (i < ids.length) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
          newIds.push(bestIdx);
          i += 2;
        } else {
          newIds.push(ids[i]);
          i += 1;
        }
      }
      ids = newIds;
    }

    return ids;
  }

  /**
   * Decode an array of token IDs back to a string.
   */
  decode(ids: number[]): string {
    const chunks: Uint8Array[] = [];
    for (const id of ids) {
      const bytes = this.vocab.get(id);
      chunks.push(bytes ?? new Uint8Array([63])); // '?' for unknown
    }

    // Concatenate all byte arrays
    const totalLen = chunks.reduce((sum, c) => sum + c.length, 0);
    const result = new Uint8Array(totalLen);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    const decoder = new TextDecoder("utf-8", { fatal: false });
    return decoder.decode(result);
  }
}

/**
 * Fetch and initialize the tokenizer from the public models directory.
 */
export async function loadTokenizer(url = "/models/tokenizer.json"): Promise<ByteLevelTokenizer> {
  const response = await fetch(url);
  const data: TokenizerData = await response.json();
  const tokenizer = new ByteLevelTokenizer();
  tokenizer.load(data);
  return tokenizer;
}
