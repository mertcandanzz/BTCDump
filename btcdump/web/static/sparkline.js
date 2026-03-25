/**
 * Canvas-based sparkline renderer for mini charts.
 * Usage: drawSparkline(canvasElement, [price1, price2, ...], {width, height})
 */
function drawSparkline(canvas, prices, opts) {
    opts = opts || {};
    const ctx = canvas.getContext('2d');
    const w = opts.width || canvas.clientWidth || 120;
    const h = opts.height || canvas.clientHeight || 30;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    if (!prices || prices.length < 2) return;

    const isUp = prices[prices.length - 1] >= prices[0];
    const lineColor = opts.lineColor || (isUp ? '#00d4aa' : '#ff4466');
    const fillColor = opts.fillColor || (isUp ? 'rgba(0,212,170,0.15)' : 'rgba(255,68,102,0.15)');

    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min || 1;
    const pad = 2;
    const stepX = (w - pad * 2) / (prices.length - 1);

    ctx.beginPath();
    prices.forEach((price, i) => {
        const x = pad + i * stepX;
        const y = h - pad - ((price - min) / range) * (h - pad * 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.strokeStyle = lineColor;
    ctx.lineWidth = opts.lineWidth || 1.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Fill gradient under curve
    ctx.lineTo(pad + (prices.length - 1) * stepX, h);
    ctx.lineTo(pad, h);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
}
