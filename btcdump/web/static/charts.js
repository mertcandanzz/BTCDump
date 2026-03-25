/**
 * Charts.js - Equity Curve, Feature Importance Bar Chart, Correlation Heatmap
 */

// ── Equity Curve ──
function drawEquityCurve(canvas, data) {
    // data: [[equity, drawdown], ...]
    if (!data || data.length < 2) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0e0e16'; ctx.fillRect(0, 0, w, h);

    const padL = 50, padR = 10, padT = 10, padB = 24;
    const cw = w - padL - padR, ch = h - padT - padB;

    const equities = data.map(d => d[0]);
    const drawdowns = data.map(d => d[1]);
    const minE = Math.min(...equities) * 0.98, maxE = Math.max(...equities) * 1.02;
    const rangeE = maxE - minE || 1;
    const stepX = cw / (data.length - 1);

    const ey = v => padT + ch - ((v - minE) / rangeE) * ch;

    // Grid
    ctx.strokeStyle = '#1c1c2e'; ctx.lineWidth = 0.5;
    ctx.font = '9px -apple-system,sans-serif'; ctx.fillStyle = '#6b6b80'; ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const v = minE + rangeE * i / 4;
        const y = ey(v);
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
        ctx.fillText('$' + v.toFixed(1), padL - 4, y + 3);
    }

    // Drawdown area (red)
    ctx.beginPath();
    data.forEach((d, i) => {
        const x = padL + i * stepX;
        const baseY = ey(equities[i]);
        const ddY = ey(equities[i] * (1 + d[1] / 100));
        if (i === 0) ctx.moveTo(x, baseY);
        else ctx.lineTo(x, baseY);
    });
    for (let i = data.length - 1; i >= 0; i--) {
        ctx.lineTo(padL + i * stepX, ey(100)); // baseline
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(239,83,80,0.15)';
    ctx.fill();

    // Equity line (green)
    ctx.beginPath();
    equities.forEach((v, i) => {
        const x = padL + i * stepX, y = ey(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#26a69a'; ctx.lineWidth = 1.5; ctx.stroke();

    // Start/end labels
    ctx.fillStyle = '#9b9bb0'; ctx.font = '10px -apple-system,sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Start', padL + 4, padT + 14);
    ctx.textAlign = 'right';
    const finalVal = equities[equities.length - 1];
    const retPct = ((finalVal / equities[0]) - 1) * 100;
    ctx.fillStyle = retPct >= 0 ? '#26a69a' : '#ef5350';
    ctx.fillText(`$${finalVal.toFixed(1)} (${retPct >= 0 ? '+' : ''}${retPct.toFixed(1)}%)`, w - padR - 4, padT + 14);
}

// ── Feature Importance Bar Chart ──
function drawFeatureChart(canvas, features) {
    // features: [{name, importance, rank, description}, ...] (sorted by importance desc)
    if (!features || !features.length) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0e0e16'; ctx.fillRect(0, 0, w, h);

    const top = features.slice(0, 20);
    const padL = 110, padR = 50, padT = 8, padB = 8;
    const barH = Math.min(18, (h - padT - padB) / top.length - 2);
    const maxImp = top[0].importance || 1;

    top.forEach((f, i) => {
        const y = padT + i * (barH + 3);
        const barW = ((f.importance / maxImp) * (w - padL - padR));

        // Bar
        const green = Math.round(166 * (f.importance / maxImp));
        ctx.fillStyle = `rgba(38,${100 + green},154,${0.4 + 0.6 * (f.importance / maxImp)})`;
        ctx.fillRect(padL, y, barW, barH);

        // Rank + Name
        ctx.fillStyle = '#6b6b80'; ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'right';
        ctx.fillText(`#${f.rank} ${f.name}`, padL - 6, y + barH - 4);

        // Value
        ctx.fillStyle = '#9b9bb0'; ctx.textAlign = 'left';
        ctx.fillText(`${(f.importance * 100).toFixed(1)}%`, padL + barW + 4, y + barH - 4);
    });
}

// ── Coin Heatmap (Treemap-style) ──
function drawCoinHeatmap(canvas, coins) {
    // coins: [{symbol, baseAsset, priceChangePercent, quoteVolume, direction}, ...]
    if (!coins || !coins.length) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0e0e16'; ctx.fillRect(0, 0, w, h);

    // Sort by volume (largest first) for treemap
    const sorted = [...coins].sort((a, b) => (b.quoteVolume || 0) - (a.quoteVolume || 0));
    const n = sorted.length;

    // Simple grid layout (close to square)
    const cols = Math.ceil(Math.sqrt(n * w / h));
    const rows = Math.ceil(n / cols);
    const cellW = w / cols;
    const cellH = h / rows;

    sorted.forEach((coin, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = col * cellW;
        const y = row * cellH;
        const chg = parseFloat(coin.priceChangePercent) || 0;

        // Color based on change %
        let r, g, b;
        if (chg >= 0) {
            const intensity = Math.min(1, chg / 10); // 10% = max green
            r = Math.round(14 + (38 - 14) * intensity);
            g = Math.round(14 + (166 - 14) * intensity);
            b = Math.round(22 + (154 - 22) * intensity);
        } else {
            const intensity = Math.min(1, Math.abs(chg) / 10);
            r = Math.round(14 + (239 - 14) * intensity);
            g = Math.round(14 + (83 - 14) * intensity);
            b = Math.round(22 + (80 - 22) * intensity);
        }

        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

        // Text
        const name = coin.baseAsset || coin.symbol.replace('USDT', '');
        ctx.font = `bold ${Math.min(14, cellW / 4)}px -apple-system,sans-serif`;
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.fillText(name, x + cellW / 2, y + cellH / 2 - 2);

        ctx.font = `${Math.min(10, cellW / 6)}px -apple-system,sans-serif`;
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.fillText(`${chg >= 0 ? '+' : ''}${chg.toFixed(1)}%`, x + cellW / 2, y + cellH / 2 + 12);
    });
}

// ── Monte Carlo Paths ──
function drawMonteCarlo(canvas, paths, percentiles) {
    if (!paths || !paths.length) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0e0e16'; ctx.fillRect(0, 0, w, h);

    const padL = 55, padR = 10, padT = 10, padB = 24;
    const cw = w - padL - padR, ch = h - padT - padB;

    // Find min/max across all paths
    let minV = Infinity, maxV = -Infinity;
    paths.forEach(p => p.forEach(v => { if (v < minV) minV = v; if (v > maxV) maxV = v; }));
    const pad = (maxV - minV) * 0.05;
    minV -= pad; maxV += pad;
    const rangeV = maxV - minV || 1;
    const steps = paths[0].length;
    const stepX = cw / (steps - 1);

    const vy = v => padT + ch - ((v - minV) / rangeV) * ch;

    // Grid
    ctx.strokeStyle = '#1c1c2e'; ctx.lineWidth = 0.5;
    ctx.font = '9px -apple-system,sans-serif'; ctx.fillStyle = '#6b6b80'; ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const v = minV + rangeV * i / 4;
        const y = vy(v);
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
        ctx.fillText('$' + v.toFixed(0), padL - 4, y + 3);
    }

    // Draw all sample paths (faint)
    paths.forEach(path => {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(100,100,150,0.15)';
        ctx.lineWidth = 1;
        path.forEach((v, i) => {
            const x = padL + i * stepX, y = vy(v);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
    });

    // Starting balance line
    const startY = vy(paths[0][0]);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#555'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, startY); ctx.lineTo(w - padR, startY); ctx.stroke();
    ctx.setLineDash([]);

    // Percentile labels on right
    if (percentiles) {
        const labels = [
            { key: 'p95', color: '#26a69a', label: 'P95' },
            { key: 'p50', color: '#f7b924', label: 'P50' },
            { key: 'p5', color: '#ef5350', label: 'P5' },
        ];
        labels.forEach(({ key, color, label }) => {
            const val = percentiles[key];
            if (val == null) return;
            const y = vy(val);
            ctx.fillStyle = color;
            ctx.font = 'bold 10px -apple-system,sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`${label}: $${val.toFixed(0)}`, w - padR - 4, y - 3);
            ctx.beginPath();
            ctx.setLineDash([2, 3]);
            ctx.strokeStyle = color; ctx.lineWidth = 1;
            ctx.moveTo(padL, y); ctx.lineTo(w - padR, y);
            ctx.stroke();
            ctx.setLineDash([]);
        });
    }
}

// ── Seasonality Bar Chart ──
function drawSeasonalityChart(canvas, data, type) {
    // data: {key: {avg_return, win_rate, count}, ...}
    if (!data || !Object.keys(data).length) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0e0e16'; ctx.fillRect(0, 0, w, h);

    const keys = Object.keys(data);
    const values = keys.map(k => data[k].avg_return);
    const maxAbs = Math.max(...values.map(Math.abs)) || 0.01;

    const padL = 30, padR = 10, padT = 12, padB = 20;
    const barW = (w - padL - padR) / keys.length;
    const midY = padT + (h - padT - padB) / 2;

    // Zero line
    ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, midY); ctx.lineTo(w - padR, midY); ctx.stroke();

    keys.forEach((key, i) => {
        const val = data[key].avg_return;
        const x = padL + i * barW;
        const barH = (val / maxAbs) * ((h - padT - padB) / 2 - 4);
        const y = val >= 0 ? midY - barH : midY;
        const color = val >= 0 ? '#26a69a' : '#ef5350';

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.7;
        ctx.fillRect(x + 2, y, barW - 4, Math.abs(barH));
        ctx.globalAlpha = 1;

        // Label
        ctx.fillStyle = '#9b9bb0';
        ctx.font = '8px -apple-system,sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(type === 'hourly' ? `${key}h` : key, x + barW / 2, h - 6);

        // Value on top
        ctx.fillStyle = color;
        ctx.font = '7px -apple-system,sans-serif';
        const textY = val >= 0 ? y - 2 : y + Math.abs(barH) + 8;
        ctx.fillText(`${val >= 0 ? '+' : ''}${val.toFixed(3)}%`, x + barW / 2, textY);
    });

    // Title
    ctx.fillStyle = '#6b6b80';
    ctx.font = '10px -apple-system,sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(type === 'hourly' ? 'Avg Return by Hour' : 'Avg Return by Day', padL, padT);
}

// ── Correlation Heatmap ──
function drawCorrelationHeatmap(canvas, matrix, symbols) {
    if (!matrix || !symbols || !symbols.length) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0e0e16'; ctx.fillRect(0, 0, w, h);

    const n = symbols.length;
    const padL = 60, padT = 40;
    const cellW = Math.min(40, (w - padL) / n);
    const cellH = Math.min(40, (h - padT) / n);

    // Column headers
    ctx.font = '9px -apple-system,sans-serif'; ctx.textAlign = 'center';
    symbols.forEach((s, i) => {
        ctx.save();
        ctx.translate(padL + i * cellW + cellW / 2, padT - 6);
        ctx.rotate(-Math.PI / 4);
        ctx.fillStyle = '#9b9bb0';
        ctx.fillText(s.replace('USDT', ''), 0, 0);
        ctx.restore();
    });

    // Row headers + cells
    symbols.forEach((row, ri) => {
        ctx.fillStyle = '#9b9bb0'; ctx.textAlign = 'right';
        ctx.fillText(row.replace('USDT', ''), padL - 4, padT + ri * cellH + cellH / 2 + 3);

        symbols.forEach((col, ci) => {
            const corr = matrix[row]?.[col] ?? 0;
            const x = padL + ci * cellW, y = padT + ri * cellH;

            // Color: -1=red, 0=dark, +1=green
            if (corr >= 0) {
                ctx.fillStyle = `rgba(38,166,154,${corr * 0.8})`;
            } else {
                ctx.fillStyle = `rgba(239,83,80,${Math.abs(corr) * 0.8})`;
            }
            ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

            // Value text
            if (cellW > 25) {
                ctx.fillStyle = Math.abs(corr) > 0.5 ? '#fff' : '#888';
                ctx.font = '8px -apple-system,sans-serif'; ctx.textAlign = 'center';
                ctx.fillText(corr.toFixed(2), x + cellW / 2, y + cellH / 2 + 3);
            }
        });
    });
}
