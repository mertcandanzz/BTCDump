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
