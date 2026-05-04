<script lang="ts">
  import { onMount } from "svelte";

  interface Experiment {
    exp: number | "REGIME_CHANGE";
    val_bpb?: number;
    result?: "IMPROVED" | "REGRESSED";
    hypothesis?: string;
    changes?: string;
    note?: string;
  }

  let experiments: Experiment[] = $state([]);
  let loading = $state(true);
  let error = $state("");

  let containerWidth = $state(1000);
  let width = $derived(Math.max(containerWidth, 400));
  let height = 400;
  let padding = { top: 40, right: 40, bottom: 60, left: 50 };

  // Hover state
  let hoveredExp: Experiment | null = $state(null);
  let mouseX = $state(0);
  let mouseY = $state(0);

  onMount(async () => {
    try {
      const res = await fetch("/experiments.jsonl");
      if (!res.ok) throw new Error("Failed to load experiments.jsonl");
      
      const text = await res.text();
      experiments = text
        .trim()
        .split("\n")
        .filter(line => line.trim().length > 0)
        .map(line => JSON.parse(line));
        
      loading = false;
    } catch (e) {
      error = String(e);
      loading = false;
    }
  });

  // Calculate points
  let dataPoints = $derived.by(() => {
    const points: { x: number, y: number, exp: Experiment, regime: number }[] = [];
    if (!experiments.length) return points;

    // Filter to only actual experiments with val_bpb
    const validExps = experiments.filter(e => typeof e.exp === "number" && e.val_bpb !== undefined);
    if (!validExps.length) return points;

    const minBpb = Math.min(...validExps.map(e => e.val_bpb!));
    const maxBpb = Math.max(...validExps.map(e => e.val_bpb!));
    
    // We want to space them evenly on X
    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;
    
    let currentRegime = 0;
    let xStep = innerWidth / Math.max(1, experiments.length - 1);

    experiments.forEach((exp, i) => {
      if (exp.exp === "REGIME_CHANGE") {
        currentRegime++;
      } else if (typeof exp.exp === "number" && exp.val_bpb !== undefined) {
        // Normalize Y (lower is better, so higher up on graph)
        // Wait, standard charts usually have lower values at the bottom.
        const yNorm = (exp.val_bpb - minBpb) / Math.max(0.01, (maxBpb - minBpb));
        const y = padding.top + innerHeight - (yNorm * innerHeight);
        const x = padding.left + (i * xStep);
        
        points.push({ x, y, exp, regime: currentRegime });
      }
    });

    return points;
  });

  // Group by regime for drawing lines
  let regimes = $derived.by(() => {
    const groups: (typeof dataPoints)[] = [];
    dataPoints.forEach(p => {
      if (!groups[p.regime]) groups[p.regime] = [];
      groups[p.regime].push(p);
    });
    return groups.filter(g => g && g.length > 0);
  });

  let regimeChanges = $derived.by(() => {
    if (!experiments.length) return [];
    const innerWidth = width - padding.left - padding.right;
    let xStep = innerWidth / Math.max(1, experiments.length - 1);
    
    return experiments
      .map((e, i) => ({ e, i }))
      .filter(({ e }) => e.exp === "REGIME_CHANGE")
      .map(({ e, i }) => ({
        x: padding.left + (i * xStep),
        note: e.note
      }));
  });

  let yTicks = $derived.by(() => {
    if (!dataPoints.length) return [];
    const validExps = dataPoints.map(p => p.exp.val_bpb!);
    const minBpb = Math.min(...validExps);
    const maxBpb = Math.max(...validExps);
    return [
      { val: maxBpb, y: padding.top },
      { val: (maxBpb + minBpb) / 2, y: padding.top + (height - padding.top - padding.bottom) / 2 },
      { val: minBpb, y: height - padding.bottom }
    ];
  });

</script>

<div class="chart-container">
  <div class="header">
    <h3>Training History</h3>
    <p>Validation Bits-Per-Byte across autonomous experiments</p>
  </div>

  {#if loading}
    <div class="state">Loading experiments...</div>
  {:else if error}
    <div class="state error">⚠️ {error}</div>
  {:else}
    <div class="svg-wrapper"
         bind:clientWidth={containerWidth}
         onmousemove={(e) => {
           const svg = e.currentTarget.querySelector('svg');
           if (!svg) return;
           const rect = svg.getBoundingClientRect();
           mouseX = e.clientX - rect.left;
           mouseY = e.clientY - rect.top;
           
           // Find closest point
           let closest = null;
           let minDist = 40; // max hover distance
           
           dataPoints.forEach(p => {
             const dist = Math.sqrt(Math.pow(p.x - mouseX, 2) + Math.pow(p.y - mouseY, 2));
             if (dist < minDist) {
               minDist = dist;
               closest = p.exp;
             }
           });
           
           hoveredExp = closest;
         }}
         onmouseleave={() => hoveredExp = null}
    >
      <svg {width} {height} viewBox="0 0 {width} {height}">
        <!-- Axes -->
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="rgba(255,255,255,0.1)" />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="rgba(255,255,255,0.1)" />

        <!-- Y Ticks -->
        {#each yTicks as tick}
          <line x1={padding.left - 5} y1={tick.y} x2={padding.left} y2={tick.y} stroke="rgba(255,255,255,0.2)" />
          <text x={padding.left - 10} y={tick.y + 4} fill="#64748b" font-size="10" text-anchor="end">{tick.val.toFixed(1)}</text>
        {/each}

        <!-- Regime Change Lines -->
        {#each regimeChanges as rc}
          <line x1={rc.x} y1={padding.top} x2={rc.x} y2={height - padding.bottom} stroke="rgba(239, 68, 68, 0.4)" stroke-dasharray="4 4" />
          <!-- <text x={rc.x + 5} y={padding.top + 10} fill="rgba(239, 68, 68, 0.8)" font-size="9" transform="rotate(90, {rc.x + 5}, {padding.top + 10})">Regime Change</text> -->
        {/each}

        <!-- Data Lines (grouped by regime) -->
        {#each regimes as regimePoints}
          {#if regimePoints.length > 1}
            <path 
              d={`M ${regimePoints.map(p => `${p.x},${p.y}`).join(' L ')}`} 
              fill="none" 
              stroke="#4ade80" 
              stroke-width="2" 
              opacity="0.6"
            />
          {/if}
        {/each}

        <!-- Data Points -->
        {#each dataPoints as point}
          <circle 
            cx={point.x} 
            cy={point.y} 
            r={hoveredExp === point.exp ? 6 : 4} 
            fill={point.exp.result === 'IMPROVED' ? '#4ade80' : '#f87171'} 
            stroke="#0f172a"
            stroke-width="2"
            style="transition: r 0.2s"
          />
        {/each}
      </svg>

      <!-- Tooltip -->
      {#if hoveredExp}
        <div class="tooltip" style="
          left: {mouseX > width / 2 ? mouseX - 295 : mouseX + 15}px;
          {mouseY > height / 2 ? `bottom: ${height - mouseY + 15}px;` : `top: ${mouseY + 15}px;`}
        ">
          <strong>Exp {hoveredExp.exp}</strong>
          <span class="badge {hoveredExp.result === 'IMPROVED' ? 'improved' : 'regressed'}">
            {hoveredExp.result}
          </span>
          <p class="bpb">val_bpb: {hoveredExp.val_bpb?.toFixed(4)}</p>
          {#if hoveredExp.changes}<p class="changes"><strong>Changes:</strong> {hoveredExp.changes}</p>{/if}
          {#if hoveredExp.hypothesis}<p class="hypothesis"><strong>Hypothesis:</strong> {hoveredExp.hypothesis}</p>{/if}
        </div>
      {/if}
    </div>

    <!-- Timeline Legend -->
    <div class="legend">
      <div class="legend-item"><span class="dot improved"></span> Improved</div>
      <div class="legend-item"><span class="dot regressed"></span> Regressed</div>
      <div class="legend-item"><span class="line regime"></span> Regime Change</div>
    </div>
  {/if}
</div>

<style>
  .chart-container {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .header h3 {
    margin: 0;
    font-size: 1.1rem;
    color: #e2e8f0;
  }

  .header p {
    margin: 0.25rem 0 0 0;
    font-size: 0.85rem;
    color: #94a3b8;
  }

  .state {
    padding: 2rem;
    text-align: center;
    color: #64748b;
  }
  .state.error { color: #f87171; }

  .svg-wrapper {
    position: relative;
    width: 100%;
    overflow-x: auto;
    cursor: crosshair;
  }

  svg {
    width: 100%;
    display: block;
  }

  .legend {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    font-size: 0.8rem;
    color: #94a3b8;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }
  .dot.improved { background: #4ade80; }
  .dot.regressed { background: #f87171; }
  
  .line.regime {
    width: 16px;
    height: 0;
    border-top: 2px dashed rgba(239, 68, 68, 0.6);
  }

  .tooltip {
    position: absolute;
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 0.75rem;
    width: 280px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    pointer-events: none;
    z-index: 10;
    color: #cbd5e1;
    font-size: 0.8rem;
  }

  .tooltip strong {
    color: #fff;
    font-size: 0.9rem;
  }

  .badge {
    float: right;
    font-size: 0.7rem;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-weight: 600;
  }
  .badge.improved { background: rgba(74, 222, 128, 0.2); color: #4ade80; }
  .badge.regressed { background: rgba(248, 113, 113, 0.2); color: #f87171; }

  .bpb {
    margin: 0.4rem 0;
    font-family: monospace;
    font-size: 0.85rem;
    color: #94a3b8;
  }

  .changes, .hypothesis {
    margin: 0.4rem 0 0 0;
    line-height: 1.4;
  }
  .changes strong, .hypothesis strong {
    color: #94a3b8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
</style>
