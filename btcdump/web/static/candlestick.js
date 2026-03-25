/**
 * TradingView-quality interactive candlestick chart.
 * Features: Zoom (scroll), Pan (drag), Crosshair, OHLCV Tooltip,
 * EMA overlays, Volume histogram, Current price tag, Ruler tool.
 */
const TVChart = (() => {
    const C = {
        bg:'#0e0e16', green:'#26a69a', red:'#ef5350',
        greenVol:'rgba(38,166,154,0.3)', redVol:'rgba(239,83,80,0.3)',
        grid:'#1c1c2e', text:'#6b6b80', textLight:'#9b9bb0',
        crosshair:'#555570', ema9:'#f7b924', ema21:'#7b68ee',
        tooltip:'#1e1e32', tooltipBorder:'#333350',
        ruler:'#ff9800',
    };

    let _allCandles=[], _canvas=null, _ctx=null;
    let _viewStart=0, _viewEnd=0; // visible candle range
    let _mouseX=-1, _mouseY=-1, _hoverIdx=-1;
    let _dragging=false, _dragStartX=0, _dragStartView=0;
    let _rulerActive=false, _rulerStart=null, _rulerEnd=null;
    let _srLevels=[]; // support/resistance levels
    let _w=0, _h=0, _dpr=1;
    const PAD_R=72, PAD_T=8, PAD_B=26;

    function setSRLevels(levels) {
        _srLevels = levels || [];
        if (_canvas) render();
    }

    function init(canvas, candles) {
        _canvas = canvas;
        _allCandles = candles || [];
        _dpr = window.devicePixelRatio || 1;
        _w = canvas.clientWidth;
        _h = canvas.clientHeight;
        canvas.width = _w * _dpr;
        canvas.height = _h * _dpr;

        // Default view: last 80 candles or all
        _viewEnd = _allCandles.length;
        _viewStart = Math.max(0, _viewEnd - 80);

        if (!canvas._tvEvents) {
            canvas.addEventListener('wheel', onWheel, {passive:false});
            canvas.addEventListener('mousedown', onMouseDown);
            canvas.addEventListener('mousemove', onMouseMove);
            canvas.addEventListener('mouseup', onMouseUp);
            canvas.addEventListener('mouseleave', onMouseLeave);
            canvas.addEventListener('dblclick', onDblClick);
            canvas._tvEvents = true;
        }
        render();
    }

    function render() {
        if (!_canvas || !_allCandles.length) return;
        const ctx = _canvas.getContext('2d');
        _ctx = ctx;
        _w = _canvas.clientWidth;
        _h = _canvas.clientHeight;
        _canvas.width = _w * _dpr;
        _canvas.height = _h * _dpr;
        ctx.scale(_dpr, _dpr);

        const candles = _allCandles.slice(_viewStart, _viewEnd);
        if (candles.length < 2) return;

        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, _w, _h);

        const chartW = _w - PAD_R;
        const priceH = (_h - PAD_T - PAD_B) * 0.78;
        const volH = (_h - PAD_T - PAD_B) * 0.18;
        const volTop = PAD_T + priceH + 4;
        const gap = chartW / candles.length;
        const candleW = Math.max(1, Math.min(gap * 0.7, 20));

        let minP=Infinity, maxP=-Infinity, maxV=0;
        candles.forEach(c => { if(c.l<minP) minP=c.l; if(c.h>maxP) maxP=c.h; if(c.v>maxV) maxV=c.v; });
        const pad = (maxP-minP)*0.06||1;
        minP-=pad; maxP+=pad;
        const range = maxP-minP;

        const py = p => PAD_T + priceH - ((p-minP)/range)*priceH;

        // Grid
        ctx.strokeStyle = C.grid; ctx.lineWidth = 0.5;
        const step = niceNum(range/5);
        const gStart = Math.ceil(minP/step)*step;
        ctx.font = '10px -apple-system,sans-serif';
        ctx.textAlign = 'left';
        for (let p=gStart; p<=maxP; p+=step) {
            const y = Math.round(py(p))+0.5;
            ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(chartW,y); ctx.stroke();
            ctx.fillStyle = C.text;
            ctx.fillText(fmtP(p), chartW+6, y+3);
        }

        // Time axis + vertical grid
        ctx.textAlign = 'center';
        const tStep = Math.max(1, Math.floor(candles.length/6));
        for (let i=tStep; i<candles.length; i+=tStep) {
            const x = i*gap + gap/2;
            const d = new Date(candles[i].t);
            ctx.fillStyle = C.text;
            ctx.fillText(`${d.toLocaleString('en',{month:'short'})} ${d.getDate()}`, x, _h-12);
            ctx.fillStyle = C.textLight;
            ctx.fillText(`${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`, x, _h-2);
            ctx.strokeStyle = '#14141f'; ctx.lineWidth=0.5;
            ctx.beginPath(); ctx.moveTo(Math.round(x)+0.5, PAD_T); ctx.lineTo(Math.round(x)+0.5, PAD_T+priceH); ctx.stroke();
        }

        // Volume
        candles.forEach((c,i) => {
            const x = i*gap+(gap-candleW)/2;
            ctx.fillStyle = c.c>=c.o ? C.greenVol : C.redVol;
            const vh = (c.v/(maxV||1))*volH;
            ctx.fillRect(x, volTop+volH-vh, candleW, vh);
        });

        // EMA
        const ema9 = calcEMA(candles.map(c=>c.c),9);
        const ema21 = calcEMA(candles.map(c=>c.c),21);
        drawLine(ctx,candles,ema9,py,gap,C.ema9,1.2);
        drawLine(ctx,candles,ema21,py,gap,C.ema21,1.2);

        // Support / Resistance lines
        if (_srLevels.length) {
            _srLevels.forEach(sr => {
                if (sr.price < minP || sr.price > maxP) return;
                const y = Math.round(py(sr.price)) + 0.5;
                const isSup = sr.type === 'support';
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = isSup ? 'rgba(38,166,154,0.6)' : 'rgba(239,83,80,0.6)';
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(chartW, y); ctx.stroke();
                ctx.setLineDash([]);
                // Label
                ctx.font = '9px -apple-system,sans-serif';
                ctx.fillStyle = isSup ? 'rgba(38,166,154,0.8)' : 'rgba(239,83,80,0.8)';
                ctx.textAlign = 'left';
                const label = `${isSup ? 'S' : 'R'} $${fmtP(sr.price)} (${sr.touches}x)`;
                ctx.fillText(label, 4, y - 3);
            });
        }

        // Candles
        candles.forEach((c,i) => {
            const cx = i*gap+gap/2;
            const bx = cx-candleW/2;
            const up = c.c>=c.o;
            const color = up ? C.green : C.red;
            const bt = py(Math.max(c.o,c.c));
            const bb = py(Math.min(c.o,c.c));
            const bh = Math.max(1, bb-bt);
            ctx.strokeStyle=color; ctx.lineWidth=1;
            ctx.beginPath(); ctx.moveTo(Math.round(cx)+0.5,Math.round(py(c.h))); ctx.lineTo(Math.round(cx)+0.5,Math.round(py(c.l))); ctx.stroke();
            ctx.fillStyle=color;
            ctx.fillRect(Math.round(bx),Math.round(bt),Math.round(candleW),Math.max(1,Math.round(bh)));
        });

        // Current price line + tag
        const last=candles[candles.length-1], prev=candles[candles.length-2];
        const lastUp = last.c>=prev.c;
        const cpY = Math.round(py(last.c));
        ctx.setLineDash([2,3]); ctx.strokeStyle=lastUp?'rgba(38,166,154,0.5)':'rgba(239,83,80,0.5)'; ctx.lineWidth=1;
        ctx.beginPath(); ctx.moveTo(0,cpY+0.5); ctx.lineTo(chartW,cpY+0.5); ctx.stroke(); ctx.setLineDash([]);
        // Tag
        const tc = lastUp?C.green:C.red;
        ctx.fillStyle=tc;
        ctx.beginPath(); ctx.moveTo(chartW,cpY); ctx.lineTo(chartW+5,cpY-9); ctx.lineTo(_w-2,cpY-9); ctx.lineTo(_w-2,cpY+9); ctx.lineTo(chartW+5,cpY+9); ctx.closePath(); ctx.fill();
        ctx.fillStyle='#fff'; ctx.font='bold 10px -apple-system,sans-serif'; ctx.textAlign='center';
        ctx.fillText(fmtP(last.c), chartW+PAD_R/2, cpY+4);

        // Legend
        ctx.font='10px -apple-system,sans-serif'; ctx.textAlign='left';
        ctx.fillStyle=C.ema9; ctx.fillText('EMA 9',8,PAD_T+14);
        ctx.fillStyle=C.ema21; ctx.fillText('EMA 21',56,PAD_T+14);
        // Zoom hint
        ctx.fillStyle=C.text; ctx.textAlign='right';
        ctx.fillText(`${candles.length} candles | Scroll:zoom  Drag:pan  DblClick:ruler`, chartW-4, PAD_T+14);

        // Ruler
        if (_rulerActive && _rulerStart && _rulerEnd) {
            const s = _rulerStart, e = _rulerEnd;
            ctx.setLineDash([4,4]); ctx.strokeStyle=C.ruler; ctx.lineWidth=1.5;
            ctx.beginPath(); ctx.moveTo(s.x,s.y); ctx.lineTo(e.x,e.y); ctx.stroke(); ctx.setLineDash([]);
            // Price diff label
            const p1 = minP + (1-(s.y-PAD_T)/priceH)*range;
            const p2 = minP + (1-(e.y-PAD_T)/priceH)*range;
            const diff = p2-p1;
            const pct = (diff/p1)*100;
            const mx=(s.x+e.x)/2, my=(s.y+e.y)/2;
            ctx.fillStyle=C.ruler; ctx.font='bold 11px -apple-system,sans-serif'; ctx.textAlign='center';
            ctx.fillText(`${diff>=0?'+':''}${fmtP(diff)} (${pct>=0?'+':''}${pct.toFixed(2)}%)`, mx, my-8);
        }

        // Crosshair
        if (_hoverIdx>=0 && _hoverIdx<candles.length && !_dragging) {
            const c = candles[_hoverIdx];
            const x = Math.round(_hoverIdx*gap+gap/2)+0.5;
            ctx.strokeStyle=C.crosshair; ctx.lineWidth=0.5; ctx.setLineDash([3,3]);
            ctx.beginPath(); ctx.moveTo(x,PAD_T); ctx.lineTo(x,_h-PAD_B); ctx.stroke();
            if (_mouseY>PAD_T && _mouseY<PAD_T+priceH) {
                ctx.beginPath(); ctx.moveTo(0,Math.round(_mouseY)+0.5); ctx.lineTo(chartW,Math.round(_mouseY)+0.5); ctx.stroke();
                const hp = minP+(1-(_mouseY-PAD_T)/priceH)*range;
                ctx.setLineDash([]); ctx.fillStyle='#333350'; ctx.fillRect(chartW,_mouseY-9,PAD_R,18);
                ctx.fillStyle='#ddd'; ctx.font='10px -apple-system,sans-serif'; ctx.textAlign='center';
                ctx.fillText(fmtP(hp), chartW+PAD_R/2, _mouseY+4);
            }
            ctx.setLineDash([]);

            // Tooltip
            const up = c.c>=c.o;
            const tw=155, th=95;
            let tx=x+14; if(tx+tw>chartW) tx=x-tw-14;
            let ty=PAD_T+24;
            ctx.shadowColor='rgba(0,0,0,0.4)'; ctx.shadowBlur=8;
            ctx.fillStyle=C.tooltip; rrect(ctx,tx,ty,tw,th,4); ctx.fill();
            ctx.shadowBlur=0; ctx.strokeStyle=C.tooltipBorder; ctx.lineWidth=1; rrect(ctx,tx,ty,tw,th,4); ctx.stroke();
            ctx.font='10px -apple-system,sans-serif'; ctx.textAlign='left';
            const dt=new Date(c.t);
            ctx.fillStyle=C.textLight; ctx.fillText(`${dt.toLocaleString('en',{month:'short'})} ${dt.getDate()}, ${String(dt.getHours()).padStart(2,'0')}:${String(dt.getMinutes()).padStart(2,'0')}`,tx+8,ty+14);
            [['O',c.o,up],['H',c.h,true],['L',c.l,false],['C',c.c,up]].forEach(([l,v,u],ri) => {
                ctx.fillStyle=C.text; ctx.fillText(l,tx+8,ty+30+ri*15);
                ctx.fillStyle=u?C.green:C.red; ctx.fillText(fmtP(v),tx+24,ty+30+ri*15);
            });
            ctx.fillStyle=C.text; ctx.fillText('Vol',tx+95,ty+30);
            ctx.fillStyle=C.textLight; ctx.fillText(fmtVol(c.v),tx+95,ty+45);
            // Change from previous
            if(_hoverIdx>0) {
                const pc = candles[_hoverIdx-1].c;
                const chg = ((c.c-pc)/pc)*100;
                ctx.fillStyle=chg>=0?C.green:C.red;
                ctx.fillText(`${chg>=0?'+':''}${chg.toFixed(2)}%`,tx+95,ty+60);
            }
        }
    }

    // ── Events ──
    function onWheel(e) {
        e.preventDefault();
        const zoomFactor = e.deltaY > 0 ? 1.15 : 0.87;
        const total = _viewEnd - _viewStart;
        const newTotal = Math.max(10, Math.min(_allCandles.length, Math.round(total * zoomFactor)));
        // Zoom centered on mouse position
        const mouseRatio = _mouseX / (_w - PAD_R);
        const center = _viewStart + total * mouseRatio;
        _viewStart = Math.max(0, Math.round(center - newTotal * mouseRatio));
        _viewEnd = Math.min(_allCandles.length, _viewStart + newTotal);
        if (_viewEnd - _viewStart < 10) _viewEnd = Math.min(_allCandles.length, _viewStart + 10);
        render();
    }
    function onMouseDown(e) {
        const rect = _canvas.getBoundingClientRect();
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        if (_rulerActive) {
            _rulerStart = {x, y}; _rulerEnd = {x, y};
        } else {
            _dragging = true; _dragStartX = x; _dragStartView = _viewStart;
            _canvas.style.cursor = 'grabbing';
        }
    }
    function onMouseMove(e) {
        const rect = _canvas.getBoundingClientRect();
        _mouseX = e.clientX - rect.left;
        _mouseY = e.clientY - rect.top;
        const chartW = _w - PAD_R;
        const gap = chartW / (_viewEnd - _viewStart);
        _hoverIdx = Math.max(0, Math.min(_viewEnd-_viewStart-1, Math.floor(_mouseX / gap)));

        if (_dragging) {
            const dx = _mouseX - _dragStartX;
            const candleShift = Math.round(-dx / gap);
            let ns = _dragStartView + candleShift;
            const total = _viewEnd - _viewStart;
            ns = Math.max(0, Math.min(_allCandles.length - total, ns));
            _viewStart = ns; _viewEnd = ns + total;
        }
        if (_rulerActive && _rulerStart) {
            _rulerEnd = {x: _mouseX, y: _mouseY};
        }
        render();
    }
    function onMouseUp() {
        _dragging = false;
        if (_rulerActive && _rulerStart) { _rulerActive = false; }
        _canvas.style.cursor = 'crosshair';
    }
    function onMouseLeave() {
        _hoverIdx=-1; _mouseX=-1; _mouseY=-1; _dragging=false;
        _canvas.style.cursor='crosshair';
        render();
    }
    function onDblClick(e) {
        // Toggle ruler mode
        _rulerActive = !_rulerActive;
        _rulerStart = null; _rulerEnd = null;
        _canvas.style.cursor = _rulerActive ? 'crosshair' : 'crosshair';
        render();
    }

    // ── Helpers ──
    function drawLine(ctx,candles,vals,py,gap,color,w) {
        ctx.strokeStyle=color; ctx.lineWidth=w; ctx.beginPath();
        let s=false;
        vals.forEach((v,i) => { if(v==null) return; const x=i*gap+gap/2,y=py(v); if(!s){ctx.moveTo(x,y);s=true;}else ctx.lineTo(x,y); });
        ctx.stroke();
    }
    function calcEMA(d,p) {
        const k=2/(p+1), r=new Array(d.length).fill(null);
        let s=0;
        for(let i=0;i<d.length;i++) { if(i<p-1){s+=d[i];continue;} if(i===p-1){s+=d[i];r[i]=s/p;continue;} r[i]=d[i]*k+r[i-1]*(1-k); }
        return r;
    }
    function niceNum(v) {
        const e=Math.floor(Math.log10(Math.abs(v)||1)),f=v/Math.pow(10,e);
        return (f<=1.5?1:f<=3?2:f<=7?5:10)*Math.pow(10,e);
    }
    function rrect(ctx,x,y,w,h,r) {
        ctx.beginPath(); ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.quadraticCurveTo(x+w,y,x+w,y+r);
        ctx.lineTo(x+w,y+h-r); ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
        ctx.lineTo(x+r,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r);
        ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y); ctx.closePath();
    }
    function fmtP(p) { if(p>=10000) return p.toLocaleString(undefined,{maximumFractionDigits:0}); if(p>=1000) return p.toLocaleString(undefined,{maximumFractionDigits:1}); if(p>=1) return p.toFixed(2); if(p>=0.01) return p.toFixed(4); return p.toFixed(6); }
    function fmtVol(v) { if(v>=1e9) return (v/1e9).toFixed(1)+'B'; if(v>=1e6) return (v/1e6).toFixed(1)+'M'; if(v>=1e3) return (v/1e3).toFixed(1)+'K'; return v.toFixed(0); }

    return { init, render, setSRLevels };
})();

function drawCandlestick(canvas, candles) {
    TVChart.init(canvas, candles);
}
