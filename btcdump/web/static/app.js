// BTCDump v4.1 - Professional UX

const PROVIDERS = {
    openai:{name:'OpenAI',color:'#10a37f'}, claude:{name:'Claude',color:'#d4a574'},
    grok:{name:'Grok',color:'#1da1f2'}, gemini:{name:'Gemini',color:'#8e75ff'},
};

// ── State ──
let ws=null, providerStatus={}, currentStreamBuffers={};
let currentMode='single', activeSymbol='BTCUSDT', signalFilter='all';
let watchlist=JSON.parse(localStorage.getItem('btcdump_wl'))||['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT'];
let watchlistData={}, sortCol='quoteVolume', sortDir='desc', expandedRow=null;
let fcOpen=false, autoRefreshId=null, autoRefreshSec=0;

// ── WebSocket ──
function connectWS() {
    const p = location.protocol==='https:'?'wss:':'ws:';
    ws = new WebSocket(`${p}//${location.host}/ws`);
    ws.onopen = () => { document.getElementById('wsStatus').classList.add('connected'); };
    ws.onclose = () => { document.getElementById('wsStatus').classList.remove('connected'); setTimeout(connectWS,2000); };
    ws.onmessage = e => {
        const m = JSON.parse(e.data);
        switch(m.type) {
            case 'signal_data': updateSignalUI(m.data); toast('Signal updated','success'); break;
            case 'chat_chunk': handleChatChunk(m); break;
            case 'discussion_start': addDiscMsg('user',m.question); break;
            case 'discussion_chunk': handleDiscChunk(m); break;
            case 'discussion_complete': addDiscMsg('system','Discussion complete'); document.getElementById('discussionInput').disabled=false; break;
            case 'coin_selected': onCoinSelected(m); break;
            case 'watchlist_update': watchlistData[m.data.symbol]={...watchlistData[m.data.symbol],...m.data}; renderCompareGrid(); break;
            case 'watchlist_progress': document.getElementById('compareProgress').textContent=`${m.data.completed}/${m.data.total}`; if(m.data.completed>=m.data.total) setTimeout(()=>document.getElementById('compareProgress').textContent='',2000); break;
            case 'backtest_progress': onBacktestProgress(m.data); break;
            case 'backtest_complete': onBacktestComplete(m.data); break;
            case 'live_price': onLivePrice(m.data); break;
            case 'alert_triggered': onAlertTriggered(m.alert); break;
            case 'status': addDiscMsg('system',m.message); break;
            case 'error': toast(m.message,'error'); break;
        }
    };
}

// ── Toast Notifications ──
function toast(msg, type='info') {
    const c=document.getElementById('toastContainer');
    const t=document.createElement('div');
    t.className=`toast ${type}`; t.textContent=msg;
    c.appendChild(t);
    setTimeout(()=>t.remove(), 3000);
}

// ── Command Palette (Ctrl+K) ──
function openCmdPalette() { document.getElementById('cmdPalette').style.display='flex'; document.getElementById('cmdInput').value=''; document.getElementById('cmdInput').focus(); onCmdSearch(''); }
function closeCmdPalette() { document.getElementById('cmdPalette').style.display='none'; }
async function onCmdSearch(q) {
    const el=document.getElementById('cmdResults');
    try {
        const r=await fetch(`/api/coins?q=${encodeURIComponent(q||'')}&limit=12`);
        const j=await r.json();
        if(!j.ok) return;
        el.innerHTML = j.coins.map(c => {
            const cls=c.priceChangePercent>=0?'up':'down';
            return `<div class="coin-search-item" onclick="selectCoin('${c.symbol}');closeCmdPalette()">
                <span class="coin-search-sym">${c.baseAsset}</span>
                <span class="coin-search-price">$${fmtP(c.lastPrice)}</span>
                <span class="coin-search-change ${cls}">${c.priceChangePercent>=0?'+':''}${c.priceChangePercent.toFixed(2)}%</span>
            </div>`;
        }).join('');
    } catch(e){}
}

// ── Keyboard Shortcuts ──
document.addEventListener('keydown', e => {
    if(e.ctrlKey && e.key==='k') { e.preventDefault(); openCmdPalette(); }
    if(e.key==='Escape') { closeCmdPalette(); closeSettings(); }
    if(document.activeElement.tagName==='INPUT'||document.activeElement.tagName==='SELECT') return;
    if(e.key==='s'||e.key==='S') switchMode('single');
    if(e.key==='c'||e.key==='C') switchMode('compare');
    if(e.key==='r'||e.key==='R') refreshSignal();
});

// ── Mode Switch ──
function switchMode(mode) {
    currentMode=mode;
    document.getElementById('modeSingle').classList.toggle('active',mode==='single');
    document.getElementById('modeCompare').classList.toggle('active',mode==='compare');
    document.getElementById('singleSignalView').style.display=mode==='single'?'':'none';
    document.getElementById('watchlistManager').style.display=mode==='compare'?'':'none';
    document.getElementById('singleModeContent').style.display=mode==='single'?'':'none';
    document.getElementById('compareModeContent').style.display=mode==='compare'?'':'none';
    document.getElementById('mainLayout').classList.toggle('compare-mode',mode==='compare');
    updateFcContext();
    if(mode==='compare'){renderWatchlistMgr();loadWatchlistOverview();}
}

// ── Timeframe Pills ──
document.getElementById('tfPills').addEventListener('click', async e => {
    const btn=e.target.closest('.tf-pill');
    if(!btn) return;
    document.querySelectorAll('.tf-pill').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    const tf=btn.dataset.tf;
    try{await fetch('/api/interval',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({interval:tf})})}catch(e){}
    updateFcContext();
    loadCandlestickChart();
    toast(`Timeframe: ${tf}`,'info');
});

// ── Auto Refresh ──
function toggleAutoRefresh() {
    const intervals=[0,30,60,300];
    const idx=(intervals.indexOf(autoRefreshSec)+1)%intervals.length;
    autoRefreshSec=intervals[idx];
    if(autoRefreshId){clearInterval(autoRefreshId);autoRefreshId=null;}
    const badge=document.getElementById('autoRefreshBadge');
    if(autoRefreshSec>0){
        autoRefreshId=setInterval(refreshSignal,autoRefreshSec*1000);
        badge.style.display='';
        toast(`Auto-refresh: ${autoRefreshSec}s`,'info');
    } else {
        badge.style.display='none';
        toast('Auto-refresh off','info');
    }
}

// ── Left Panel Collapse ──
function toggleLeftPanel(){ document.getElementById('mainLayout').classList.toggle('collapsed'); }

// ── Watchlist Search ──
function onWatchlistSearch(q) {
    const d=document.getElementById('watchlistSearchDrop');
    if(!q){d.style.display='none';return;}
    setTimeout(async()=>{try{const r=await fetch(`/api/coins?q=${encodeURIComponent(q)}&limit=10`);const j=await r.json();if(j.ok)renderCoinDrop(j.coins,'watchlistSearchDrop','watchlist')}catch(e){}},200);
}
function renderCoinDrop(coins,dropId,mode) {
    const d=document.getElementById(dropId);
    if(!coins.length){d.style.display='none';return;}
    d.innerHTML=coins.map(c=>{const cls=c.priceChangePercent>=0?'up':'down';const act=mode==='watchlist'?`addToWatchlist('${c.symbol}')`:`selectCoin('${c.symbol}')`;
        return `<div class="coin-search-item" onclick="${act};this.parentElement.style.display='none'"><span class="coin-search-sym">${c.baseAsset}</span><span class="coin-search-price">$${fmtP(c.lastPrice)}</span><span class="coin-search-change ${cls}">${c.priceChangePercent>=0?'+':''}${c.priceChangePercent.toFixed(2)}%</span></div>`;
    }).join('');
    d.style.display='';
}
document.addEventListener('click',e=>{if(!e.target.closest('.watchlist-search')){const d=document.getElementById('watchlistSearchDrop');if(d)d.style.display='none';}});

// ── Coin Selection ──
async function selectCoin(sym) {
    activeSymbol=sym;
    const d=sym.replace('USDT','/USDT');
    document.getElementById('coinLabel').textContent=d;
    document.getElementById('chartTitle').textContent=d;
    document.getElementById('discussionCoinBadge').textContent=d;
    document.getElementById('signalDirection').textContent='Computing...';
    updateFcContext();
    if(currentMode!=='single') switchMode('single');
    loadCandlestickChart(sym);
    if(ws&&ws.readyState===1) ws.send(JSON.stringify({type:'select_coin',symbol:sym}));
    try{const r=await fetch(`/api/coin/${sym}/signal`);const j=await r.json();if(j.ok&&j.data?.current_price) updateSignalUI(j.data);}catch(e){}
}
function onCoinSelected(m) {
    activeSymbol=m.symbol;
    const d=m.symbol.replace('USDT','/USDT');
    document.getElementById('coinLabel').textContent=d;
    document.getElementById('chartTitle').textContent=d;
    if(m.signal_data) updateSignalUI(m.signal_data);
    loadCandlestickChart(m.symbol);
    updateFcContext();
}

// ── Watchlist ──
function addToWatchlist(s){if(watchlist.includes(s)||watchlist.length>=15)return;watchlist.push(s);saveWL();renderWatchlistMgr();syncWL();}
function removeFromWatchlist(s){watchlist=watchlist.filter(x=>x!==s);saveWL();renderWatchlistMgr();syncWL();}
function saveWL(){localStorage.setItem('btcdump_wl',JSON.stringify(watchlist));}
async function syncWL(){try{await fetch('/api/watchlist',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbols:watchlist})})}catch(e){}}
function renderWatchlistMgr(){
    document.getElementById('watchlistCount').textContent=`${watchlist.length}/15`;
    document.getElementById('watchlistItems').innerHTML=watchlist.map(s=>`<div class="watchlist-item"><span class="sym">${s.replace('USDT','')}</span><span style="color:var(--text-muted);font-size:9px">${s}</span><button class="remove-btn" onclick="removeFromWatchlist('${s}')">&times;</button></div>`).join('');
}

// ── Ticker Strip ──
async function loadTickerStrip(){try{const r=await fetch('/api/coins?limit=25');const j=await r.json();if(!j.ok)return;document.getElementById('tickerStrip').innerHTML=j.coins.slice(0,20).map(c=>`<div class="ticker-badge" onclick="selectCoin('${c.symbol}')"><span class="sym">${c.baseAsset}</span><span class="price">$${fmtP(c.lastPrice)}</span><span class="change ${c.priceChangePercent>=0?'up':'down'}">${c.priceChangePercent>=0?'+':''}${c.priceChangePercent.toFixed(1)}%</span></div>`).join('');}catch(e){}}

// ── Signal UI ──
function updateSignalUI(d) {
    if(!d?.current_price)return;
    document.getElementById('coinLabel').textContent=(d.symbol||activeSymbol).replace('USDT','/USDT');
    document.getElementById('currentPrice').textContent='$'+fmtP(d.current_price);
    const ch=document.getElementById('priceChange');
    ch.textContent=`${d.change_pct>=0?'+':''}${d.change_pct.toFixed(2)}%`;
    ch.className='signal-change '+(d.change_pct>=0?'up':'down');
    document.getElementById('predPrice').textContent='$'+fmtP(d.predicted_price)+` (${d.change_pct>=0?'+':''}${d.change_pct.toFixed(2)}%)`;

    // Signal hero
    const hero=document.getElementById('signalHero');
    const dir=d.direction||'--';
    hero.className='signal-hero '+(dir.includes('BUY')?'buy':dir.includes('SELL')?'sell':'hold');
    document.getElementById('signalDirection').textContent=dir;
    document.getElementById('signalConf').textContent=d.confidence.toFixed(0)+'%';
    const confFill=document.getElementById('confBarFill');
    confFill.style.width=d.confidence+'%';
    confFill.style.background=d.confidence>70?'var(--green)':d.confidence>40?'var(--yellow)':'var(--red)';

    document.getElementById('signalRR').textContent=d.risk_reward.toFixed(2);

    // RSI Gauge
    drawRSIGauge(document.getElementById('rsiGauge'), d.rsi||50);
    document.getElementById('indRSI').textContent=d.rsi;

    document.getElementById('indMACD').textContent=d.macd_bullish?'Bull':'Bear';
    document.getElementById('indMACD').style.color=d.macd_bullish?'var(--green)':'var(--red)';
    document.getElementById('indStoch').textContent=d.stoch_k;
    document.getElementById('indADX').textContent=d.adx;
    document.getElementById('indATR').textContent='$'+d.atr;
    document.getElementById('indVol').textContent=d.volume_ratio+'x';
    document.getElementById('modelMAPE').textContent=d.mape+'%';

    const wE=document.getElementById('weightsContent');wE.innerHTML='';
    if(d.weights){
        const modelInfo = {
            xgb: {color:'var(--green)', full:'XGBoost', desc:'Gradient boosted trees. Best at capturing complex non-linear patterns. Fast, handles missing data well. Often the most accurate on tabular data.'},
            rf: {color:'var(--yellow)', full:'Random Forest', desc:'Ensemble of decision trees with random sampling. Very robust, resistant to overfitting. Good at identifying stable trends.'},
            gb: {color:'var(--red)', full:'Gradient Boosting', desc:'Sequential tree boosting (sklearn). Slower but precise. Good at reducing bias. Captures subtle price patterns.'},
        };
        for(const[n,w] of Object.entries(d.weights)){
            const mi = modelInfo[n]||{color:'var(--accent)',full:n,desc:''};
            wE.innerHTML+=`<div class="weight-bar-wrap">
                <div class="weight-bar">
                    <span class="weight-bar-label" title="${mi.full}">${n}</span>
                    <div class="weight-bar-track"><div class="weight-bar-fill" style="width:${(w*100).toFixed(0)}%;background:${mi.color}"></div></div>
                    <span class="weight-bar-value">${(w*100).toFixed(1)}%</span>
                </div>
                <div class="weight-bar-desc">${mi.full}: ${mi.desc} ${w>0.35?'<strong>Leading model</strong> - highest weight means best accuracy on recent data.':''}</div>
            </div>`;
        }
    }

    const rE=document.getElementById('reasonsContent');rE.innerHTML='';
    (d.reasons||[]).forEach(r=>{const li=document.createElement('li');li.textContent=r;rE.appendChild(li);});
    document.getElementById('lastUpdated').textContent='Updated '+new Date().toLocaleTimeString();
    updateFcContext();
    loadSLTP(d.symbol || activeSymbol);
}

// RSI Semi-circle Gauge
function drawRSIGauge(canvas,value) {
    const ctx=canvas.getContext('2d');
    const dpr=window.devicePixelRatio||1;
    const w=100,h=60;
    canvas.width=w*dpr;canvas.height=h*dpr;
    canvas.style.width=w+'px';canvas.style.height=h+'px';
    ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);

    const cx=w/2,cy=h-4,r=38;
    // Background arc
    ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,0);
    ctx.strokeStyle='#1c1c2e';ctx.lineWidth=6;ctx.lineCap='round';ctx.stroke();
    // Colored arc
    const ratio=Math.max(0,Math.min(1,(value||0)/100));
    const angle=Math.PI+ratio*Math.PI;
    const grad=ctx.createLinearGradient(0,0,w,0);
    grad.addColorStop(0,'#26a69a');grad.addColorStop(0.5,'#f7b924');grad.addColorStop(1,'#ef5350');
    ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,angle);
    ctx.strokeStyle=grad;ctx.lineWidth=6;ctx.lineCap='round';ctx.stroke();
    // Needle dot
    const nx=cx+r*Math.cos(angle);const ny=cy+r*Math.sin(angle);
    ctx.beginPath();ctx.arc(nx,ny,4,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
}

async function refreshSignal(){const b=document.getElementById('refreshBtn');b.textContent='...';b.disabled=true;try{const r=await fetch('/api/signal');const j=await r.json();if(j.ok)updateSignalUI(j.data);}catch(e){}b.textContent='Refresh';b.disabled=false;}

// ── Chart ──
async function loadCandlestickChart(sym){
    sym=sym||activeSymbol;
    try{const r=await fetch(`/api/coin/${sym}/ohlcv?limit=300`);const j=await r.json();
        if(j.ok&&j.candles?.length>1){drawCandlestick(document.getElementById('candlestickChart'),j.candles);
            const last=j.candles[j.candles.length-1],prev=j.candles[j.candles.length-2];
            const pe=document.getElementById('chartPrice');pe.textContent='$'+fmtP(last.c);pe.className='chart-live-price '+(last.c>=prev.c?'up':'down');
            loadSRLevels(sym);
        }
    }catch(e){}
}
window.addEventListener('resize',()=>{if(currentMode==='single')loadCandlestickChart();});

// ── Compare Grid ──
async function loadWatchlistOverview(){document.getElementById('compareProgress').textContent='Loading...';try{const r=await fetch('/api/watchlist/overview');const j=await r.json();if(j.ok){j.coins.forEach(c=>watchlistData[c.symbol]=c);renderCompareGrid();}}catch(e){}document.getElementById('compareProgress').textContent='';}
function refreshWatchlist(){if(ws&&ws.readyState===1){ws.send(JSON.stringify({type:'refresh_watchlist'}));document.getElementById('compareProgress').textContent='Computing...';}}

function setSignalFilter(f){signalFilter=f;document.querySelectorAll('.filter-pill').forEach(b=>b.classList.toggle('active',b.textContent.toLowerCase().includes(f==='all'?'all':f)));renderCompareGrid();}

function renderCompareGrid(){
    let coins=watchlist.map(s=>watchlistData[s]).filter(Boolean);
    if(signalFilter==='buy') coins=coins.filter(c=>c.direction&&(c.direction.includes('BUY')));
    if(signalFilter==='sell') coins=coins.filter(c=>c.direction&&(c.direction.includes('SELL')));
    coins.sort((a,b)=>{let va=a[sortCol]??0,vb=b[sortCol]??0;if(typeof va==='string'){va=va.toLowerCase();vb=(vb||'').toLowerCase();}return sortDir==='asc'?(va>vb?1:-1):(va<vb?1:-1);});
    const ar=c=>sortCol===c?(sortDir==='asc'?' ▲':' ▼'):'';
    const th=(c,l)=>`<th onclick="sortG('${c}')">${l}${ar(c)}</th>`;
    let h=`<table class="comparison-table"><thead><tr>${th('baseAsset','Coin')}<th>Chart</th>${th('lastPrice','Price')}${th('priceChangePercent','24h%')}${th('quoteVolume','Vol')}${th('direction','Signal')}${th('confidence','Conf')}${th('change_pct','AI%')}${th('rsi','RSI')}${th('macd_bullish','MACD')}${th('stoch_k','Stoch')}${th('adx','ADX')}${th('risk_reward','R/R')}${th('model_agreement','Agree')}</tr></thead><tbody>`;
    coins.forEach(c=>{
        const rdy=c.status==='ready';const chg=c.priceChangePercent||0;
        const sig=rdy?c.direction||'--':'Pending';const sc=sigCls(rdy?c.direction:null);
        const heatCls=rdy?heatClass(c.direction):'';
        h+=`<tr class="${heatCls}" onclick="toggleDetail('${c.symbol}')">
            <td><strong>${c.baseAsset||c.symbol.replace('USDT','')}</strong></td>
            <td class="sparkline-cell"><canvas class="sparkline-canvas" data-symbol="${c.symbol}" width="100" height="26"></canvas></td>
            <td>$${fmtP(c.lastPrice)}</td>
            <td class="change-cell ${chg>=0?'up':'down'}">${chg>=0?'+':''}${chg.toFixed(2)}%</td>
            <td>${fmtVol(c.quoteVolume)}</td>
            <td><span class="signal-badge ${sc}">${sig}</span></td>
            <td>${rdy?c.confidence?.toFixed(0)+'%':'--'}</td>
            <td>${rdy&&c.change_pct!=null?(c.change_pct>=0?'+':'')+c.change_pct.toFixed(2)+'%':'--'}</td>
            <td style="color:${rsiCol(c.rsi)}">${rdy?c.rsi:'--'}</td>
            <td>${rdy?(c.macd_bullish?'<span style="color:var(--green)">Bull</span>':'<span style="color:var(--red)">Bear</span>'):'--'}</td>
            <td>${rdy?c.stoch_k?.toFixed(0):'--'}</td><td>${rdy?c.adx?.toFixed(0):'--'}</td>
            <td>${rdy?c.risk_reward?.toFixed(2):'--'}</td><td>${rdy?(c.model_agreement*100)?.toFixed(0)+'%':'--'}</td></tr>`;
        if(expandedRow===c.symbol&&rdy){
            h+=`<tr class="compare-detail-row"><td colspan="14"><div class="compare-detail">
                <div class="cd-item"><span class="cd-label">Predicted</span><span class="cd-value">$${fmtP(c.predicted_price)}</span></div>
                <div class="cd-item"><span class="cd-label">ATR</span><span class="cd-value">$${c.atr}</span></div>
                <div class="cd-item"><span class="cd-label">MAPE</span><span class="cd-value">${c.mape}%</span></div>
                <div class="cd-item"><span class="cd-label">Vol Ratio</span><span class="cd-value">${c.volume_ratio}x</span></div>
                <div class="cd-item"><span class="cd-label">Confluence</span><span class="cd-value">${c.indicator_confluence}/5</span></div>
                <div class="cd-item"><span class="cd-label">MACD Val</span><span class="cd-value">${c.macd_val}</span></div>
                ${c.reasons?.length?`<div class="cd-reasons">${c.reasons.join(' | ')}</div>`:''}
                <div style="grid-column:1/-1;margin-top:6px"><button class="btn btn-primary btn-sm" onclick="event.stopPropagation();drillInto('${c.symbol}')">Full Analysis</button></div>
            </div></td></tr>`;
        }
    });
    h+='</tbody></table>';
    document.getElementById('comparisonGrid').innerHTML=h;
    requestAnimationFrame(()=>{document.querySelectorAll('.sparkline-canvas').forEach(cv=>{const d=watchlistData[cv.dataset.symbol];if(d?.mini_chart?.length>1)drawSparkline(cv,d.mini_chart,{width:100,height:26});});});
}
function toggleDetail(s){expandedRow=expandedRow===s?null:s;renderCompareGrid();}
function sortG(c){if(sortCol===c)sortDir=sortDir==='asc'?'desc':'asc';else{sortCol=c;sortDir='desc';}renderCompareGrid();}
function drillInto(s){selectCoin(s);switchMode('single');}
function sigCls(d){if(!d)return'pending';const u=d.toUpperCase();return u==='STRONG BUY'?'strong-buy':u==='BUY'?'buy':u==='STRONG SELL'?'strong-sell':u==='SELL'?'sell':u==='HOLD'?'hold':'pending';}
function heatClass(d){if(!d)return'';const u=d.toUpperCase();return u==='STRONG BUY'?'heat-strong-buy':u==='BUY'?'heat-buy':u==='SELL'?'heat-sell':u==='STRONG SELL'?'heat-strong-sell':'';}
function rsiCol(v){if(!v)return'var(--text-dim)';if(v<30)return'var(--green)';if(v>70)return'var(--red)';return'var(--yellow)';}
function fmtVol(v){if(!v)return'--';if(v>=1e9)return'$'+(v/1e9).toFixed(1)+'B';if(v>=1e6)return'$'+(v/1e6).toFixed(0)+'M';return'$'+(v/1e3).toFixed(0)+'K';}

// ── Discussion ──
function addDiscMsg(type,text){const el=document.getElementById('discussionMessages');if(type==='user')el.innerHTML+=`<div class="msg msg-user"><div class="msg-body"><div class="msg-content">${esc(text)}</div></div><div class="msg-avatar" style="background:var(--accent)">U</div></div>`;else el.innerHTML+=`<div class="msg msg-system"><div class="msg-body"><div class="msg-content">${esc(text)}</div></div></div>`;el.scrollTop=el.scrollHeight;}
function handleDiscChunk(m){const el=document.getElementById('discussionMessages');const key=`disc-${m.provider}-r${m.round}`;if(m.content?.startsWith('__ROUND_START__')){el.innerHTML+=`<div class="round-divider">Round ${m.content.replace('__ROUND_START__','')}</div>`;el.scrollTop=el.scrollHeight;return;}if(!document.getElementById(key)&&!m.done){const p=PROVIDERS[m.provider]||{name:m.provider,color:'#888'};el.innerHTML+=`<div class="msg" id="${key}"><div class="msg-avatar" style="background:${p.color}">${p.name[0]}</div><div class="msg-body"><div class="msg-header"><span class="msg-name" style="color:${p.color}">${p.name}</span><span class="msg-round">R${m.round}</span></div><div class="msg-content" id="${key}-c"></div></div></div>`;currentStreamBuffers[key]='';}if(!m.done&&m.content){currentStreamBuffers[key]=(currentStreamBuffers[key]||'')+m.content;const ce=document.getElementById(`${key}-c`);if(ce)ce.textContent=currentStreamBuffers[key];}if(m.done)delete currentStreamBuffers[key];el.scrollTop=el.scrollHeight;}
function sendDiscussion(){const inp=document.getElementById('discussionInput');const msg=inp.value.trim();if(!msg||!ws||ws.readyState!==1)return;ws.send(JSON.stringify({type:'discussion',message:msg,rounds:parseInt(document.getElementById('roundsSelect').value)||3}));inp.value='';inp.disabled=true;setTimeout(()=>inp.disabled=false,500);}
function quickAsk(q){document.getElementById('discussionInput').value=q;sendDiscussion();}

// ── Floating Chat ──
function toggleFloatingChat(){fcOpen=!fcOpen;document.getElementById('floatingChat').style.display=fcOpen?'flex':'none';if(fcOpen)populateModelSelect();}
function minimizeChat(){document.getElementById('floatingChat').classList.toggle('minimized');}
function maximizeChat(){document.getElementById('floatingChat').classList.toggle('maximized');}
function populateModelSelect(){const sel=document.getElementById('fcModelSelect');sel.innerHTML='';for(const[pn,info]of Object.entries(providerStatus)){if(!info.enabled||!info.has_key)continue;const p=PROVIDERS[pn]||{name:pn};(info.models||[]).forEach(m=>{const opt=document.createElement('option');opt.value=`${pn}::${m}`;opt.textContent=`${p.name} - ${m}`;if(m===info.model)opt.selected=true;sel.appendChild(opt);});}if(!sel.options.length)sel.innerHTML='<option disabled>No active models</option>';}
function updateFcContext(){const el=document.getElementById('fcContext');if(!el)return;const tf=document.querySelector('.tf-pill.active')?.dataset.tf||'1h';if(currentMode==='compare')el.textContent=`Compare ${watchlist.length} coins`;else el.textContent=`${activeSymbol.replace('USDT','/USDT')} ${tf}`;}
function fcQuick(q){document.getElementById('fcInput').value=q;sendFloatingChat();}
function sendFloatingChat(){
    const inp=document.getElementById('fcInput');const msg=inp.value.trim();if(!msg||!ws||ws.readyState!==1)return;
    const sel=document.getElementById('fcModelSelect').value;if(!sel)return;
    const[provider]=sel.split('::');
    if(msg.startsWith('@all ')){addFcMsg('user',msg.substring(5));ws.send(JSON.stringify({type:'discussion',message:msg.substring(5),rounds:3}));inp.value='';return;}
    addFcMsg('user',msg);
    const botId=`fc-bot-${Date.now()}`;const pName=PROVIDERS[provider]?.name||provider;const mEl=document.getElementById('fcMessages');
    mEl.innerHTML+=`<div class="fc-msg"><div class="fc-msg-model-name" style="color:${PROVIDERS[provider]?.color||'#888'}">${pName}</div><div class="fc-msg-bot" id="${botId}"><span class="typing-dots"><span></span><span></span><span></span></span></div></div>`;
    mEl.scrollTop=mEl.scrollHeight;
    currentStreamBuffers[`chat-${provider}`]={id:botId,text:''};
    const tf=document.querySelector('.tf-pill.active')?.dataset.tf||'1h';
    ws.send(JSON.stringify({type:'chat',provider,message:msg,context:{mode:currentMode,symbol:activeSymbol,interval:tf}}));
    inp.value='';
}
function addFcMsg(type,text){const el=document.getElementById('fcMessages');el.innerHTML+=`<div class="fc-msg fc-msg-${type}">${esc(text)}</div>`;el.scrollTop=el.scrollHeight;}
function handleChatChunk(m){const buf=currentStreamBuffers[`chat-${m.provider}`];if(!buf)return;if(!m.done&&m.content){buf.text+=m.content;const e=document.getElementById(buf.id);if(e)e.textContent=buf.text;}if(m.done)delete currentStreamBuffers[`chat-${m.provider}`];const fc=document.getElementById('fcMessages');if(fc)fc.scrollTop=fc.scrollHeight;}
function startResizeChat(e){e.preventDefault();const fc=document.getElementById('floatingChat');const startW=fc.offsetWidth,startH=fc.offsetHeight,startX=e.clientX,startY=e.clientY;
    function onMove(ev){fc.style.width=Math.max(300,startW-(ev.clientX-startX))+'px';fc.style.height=Math.max(300,startH-(ev.clientY-startY))+'px';}
    function onUp(){document.removeEventListener('mousemove',onMove);document.removeEventListener('mouseup',onUp);}
    document.addEventListener('mousemove',onMove);document.addEventListener('mouseup',onUp);
}

// ── Settings ──
let showApiKeys=false;
function openSettings(){document.getElementById('settingsModal').style.display='flex';buildSettingsForm();}
function closeSettings(){document.getElementById('settingsModal').style.display='none';}
function buildSettingsForm(){const c=document.getElementById('settingsForms');const iT=showApiKeys?'text':'password';
    let h=`<div class="settings-top-controls"><label class="settings-toggle" onclick="toggleShowKeys()"><div class="toggle-switch ${showApiKeys?'active':''}" id="toggle-show-keys"></div><span>${showApiKeys?'Hide':'Show'} Keys</span></label></div>`;
    for(const[pn,info]of Object.entries(providerStatus)){const p=PROVIDERS[pn]||{name:pn,color:'#888'};
        h+=`<div class="settings-provider"><div class="settings-provider-header"><div class="settings-provider-dot" style="background:${p.color}"></div><span class="settings-provider-name">${p.name}</span><div class="settings-toggle" onclick="toggleProvider('${pn}')"><div class="toggle-switch ${info.enabled?'active':''}" id="toggle-${pn}"></div></div></div><div class="settings-field"><label>API Key</label><input type="${iT}" id="key-${pn}" value="${info.has_key?'••••••••':''}" placeholder="Enter API key..." onfocus="if(this.value==='••••••••')this.value=''"></div><div class="settings-field"><label>Model</label><select id="model-${pn}">${(info.models||[]).map(m=>`<option value="${m}" ${m===info.model?'selected':''}>${m}</option>`).join('')}</select></div></div>`;
    }h+=`<button class="btn btn-primary settings-save-all" onclick="saveAllProviders()">Save All Settings</button>`;c.innerHTML=h;}
function toggleShowKeys(){showApiKeys=!showApiKeys;buildSettingsForm();}
function toggleProvider(p){document.getElementById(`toggle-${p}`).classList.toggle('active');}
async function saveAllProviders(){const btn=document.querySelector('.settings-save-all');btn.textContent='Saving...';btn.disabled=true;
    for(const pn of Object.keys(providerStatus)){const k=document.getElementById(`key-${pn}`)?.value,m=document.getElementById(`model-${pn}`)?.value,en=document.getElementById(`toggle-${pn}`)?.classList.contains('active');const body={provider:pn,model:m,enabled:en};if(k&&k!=='••••••••')body.api_key=k;
        try{const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(j.ok)providerStatus=j.status;}catch(e){}}
    if(fcOpen)populateModelSelect();
    btn.textContent='Saved!';btn.style.background='var(--green)';btn.style.color='#000';toast('Settings saved','success');
    setTimeout(()=>{btn.textContent='Save All Settings';btn.disabled=false;btn.style.background='';btn.style.color='';},1500);}

// ── Utilities ──
function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML;}

// Open TradingView to verify indicator
function openVerify(indicator) {
    const sym = activeSymbol.replace('USDT','');
    const url = `https://www.tradingview.com/chart/?symbol=BINANCE:${sym}USDT`;
    window.open(url, '_blank');
    toast(`Opening TradingView for ${sym}/USDT - check ${indicator} there`, 'info');
}

// Toggle ensemble info panel
function toggleEnsembleInfo(e) {
    e.stopPropagation();
    const el = document.getElementById('ensembleInfo');
    el.style.display = el.style.display === 'none' ? '' : 'none';
}
function fmtP(p){if(p==null)return'--';p=parseFloat(p);if(p>=1000)return p.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});if(p>=1)return p.toFixed(2);if(p>=0.01)return p.toFixed(4);return p.toFixed(6);}

// ── Center Tab Switching ──
function switchCenterTab(tab) {
    document.querySelectorAll('.center-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(d => d.style.display = 'none');
    document.querySelector(`.center-tab[onclick*="${tab}"]`).classList.add('active');
    document.getElementById('tab' + tab.charAt(0).toUpperCase() + tab.slice(1)).style.display = '';
    if (tab === 'features') loadFeatureImportance();
    if (tab === 'portfolio') loadPortfolio();
}

// ── Feature Importance ──
async function loadFeatureImportance() {
    try {
        const r = await fetch('/api/feature-importance');
        const j = await r.json();
        if (j.ok) {
            drawFeatureChart(document.getElementById('featureChartCanvas'), j.features);
        } else {
            toast(j.error || 'No feature data', 'error');
        }
    } catch (e) { toast('Feature importance load failed', 'error'); }
}

// ── Backtest ──
function runBacktest() {
    if (!ws || ws.readyState !== 1) return;
    const retrain = parseInt(document.getElementById('btRetrain').value) || 50;
    document.getElementById('btRunBtn').disabled = true;
    document.getElementById('btProgress').style.display = 'flex';
    document.getElementById('btMetrics').style.display = 'none';
    document.getElementById('btProgressFill').style.width = '0%';
    document.getElementById('btProgressText').textContent = '0%';
    ws.send(JSON.stringify({ type: 'run_backtest', symbol: activeSymbol, retrain_every: retrain }));
    toast('Backtest started...', 'info');
}

function onBacktestProgress(data) {
    document.getElementById('btProgress').style.display = 'flex';
    document.getElementById('btProgressFill').style.width = data.pct + '%';
    document.getElementById('btProgressText').textContent = data.pct + '%';
}

function onBacktestComplete(data) {
    document.getElementById('btRunBtn').disabled = false;
    document.getElementById('btProgress').style.display = 'none';
    document.getElementById('btMetrics').style.display = 'flex';
    toast('Backtest complete!', 'success');

    // Metrics
    document.getElementById('btWinRate').textContent = (data.win_rate * 100).toFixed(1) + '%';
    document.getElementById('btWinRate').style.color = data.win_rate > 0.5 ? 'var(--green)' : 'var(--red)';
    document.getElementById('btPF').textContent = data.profit_factor.toFixed(2);
    document.getElementById('btPF').style.color = data.profit_factor > 1 ? 'var(--green)' : 'var(--red)';
    document.getElementById('btSharpe').textContent = data.sharpe_ratio.toFixed(2);
    document.getElementById('btMaxDD').textContent = data.max_drawdown_pct.toFixed(1) + '%';
    document.getElementById('btMaxDD').style.color = 'var(--red)';
    document.getElementById('btReturn').textContent = (data.total_return_pct >= 0 ? '+' : '') + data.total_return_pct.toFixed(1) + '%';
    document.getElementById('btReturn').style.color = data.total_return_pct >= 0 ? 'var(--green)' : 'var(--red)';

    // Equity curve
    if (data.equity_curve?.length) {
        drawEquityCurve(document.getElementById('equityCurveCanvas'), data.equity_curve);
    }

    // Signal accuracy
    if (data.signal_accuracy) {
        const acc = document.getElementById('btAccuracy');
        acc.style.display = '';
        acc.innerHTML = '<strong>Signal Accuracy:</strong> ' +
            Object.entries(data.signal_accuracy).map(([k, v]) =>
                `<span style="color:${k.includes('BUY')?'var(--green)':k.includes('SELL')?'var(--red)':'var(--yellow)'}">${k}: ${(v*100).toFixed(0)}%</span>`
            ).join(' | ');
    }

    // Optimal thresholds
    if (data.optimal_thresholds?.profit_factor) {
        const th = document.getElementById('btThresholds');
        th.style.display = '';
        th.innerHTML = `<strong>Optimal Thresholds:</strong> Buy: ${data.optimal_thresholds.buy_threshold}%, Sell: ${data.optimal_thresholds.sell_threshold}% (PF: ${data.optimal_thresholds.profit_factor}) <button class="btn btn-sm btn-primary" onclick="toast('Thresholds applied','success')">Apply</button>`;
    }
}

// ── Live Price ──
function onLivePrice(tick) {
    // Update ticker strip
    const badges = document.querySelectorAll('.ticker-badge');
    badges.forEach(b => {
        if (b.textContent.includes(tick.symbol.replace('USDT',''))) {
            const priceEl = b.querySelector('.price');
            const chgEl = b.querySelector('.change');
            if (priceEl) priceEl.textContent = '$' + fmtP(tick.price);
            if (chgEl) {
                chgEl.textContent = `${tick.change_pct>=0?'+':''}${tick.change_pct.toFixed(1)}%`;
                chgEl.className = `change ${tick.change_pct>=0?'up':'down'}`;
            }
            b.style.background = 'var(--accent-dim)';
            setTimeout(() => b.style.background = '', 300);
        }
    });
    // Update active coin price
    if (tick.symbol === activeSymbol) {
        document.getElementById('currentPrice').textContent = '$' + fmtP(tick.price);
    }
    // Update compare grid cell
    if (watchlistData[tick.symbol]) {
        watchlistData[tick.symbol].lastPrice = tick.price;
        watchlistData[tick.symbol].priceChangePercent = tick.change_pct;
    }
}

// ── Alerts ──
function onAlertTriggered(alert) {
    toast(`Alert: ${alert.symbol} ${alert.condition.replace('_',' ')} $${fmtP(alert.value)}`, 'success');
}

// ── Paper Trading ──
async function paperTrade(side) {
    const size = 10;
    try {
        const r = await fetch('/api/paper/open', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({symbol: activeSymbol, side, size_pct: size})
        });
        const j = await r.json();
        if (j.ok) toast(`Paper ${side} ${activeSymbol} opened`, 'success');
        else toast(j.error, 'error');
    } catch(e) { toast('Trade failed', 'error'); }
}

async function paperClose(symbol) {
    try {
        const r = await fetch('/api/paper/close', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({symbol})
        });
        const j = await r.json();
        if (j.ok) toast(`Closed ${symbol}: ${j.pnl>=0?'+':''}$${j.pnl} (${j.pnl_pct}%)`, j.pnl>=0?'success':'error');
        else toast(j.error, 'error');
    } catch(e) {}
}

async function loadPortfolio() {
    try {
        const r = await fetch('/api/paper/portfolio');
        const j = await r.json();
        if (!j.ok) return;
        const el = document.getElementById('portfolioContent');
        if (!el) return;
        const pnlColor = j.total_pnl >= 0 ? 'var(--green)' : 'var(--red)';
        let h = `<div style="padding:10px 14px;border-bottom:1px solid var(--border)">
            <div style="display:flex;justify-content:space-between;font-size:12px">
                <span>Balance: <strong>$${j.balance.toLocaleString()}</strong></span>
                <span>Total: <strong>$${j.total_value.toLocaleString()}</strong></span>
                <span style="color:${pnlColor}">P&L: ${j.total_pnl>=0?'+':''}$${j.total_pnl.toFixed(2)} (${j.total_pnl_pct.toFixed(1)}%)</span>
                <span>Win Rate: ${(j.win_rate*100).toFixed(0)}%</span>
            </div>
        </div>`;
        if (j.positions.length) {
            h += '<div style="padding:6px 14px;font-size:11px">';
            j.positions.forEach(p => {
                const pc = p.pnl >= 0 ? 'var(--green)' : 'var(--red)';
                h += `<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--border)">
                    <span><strong>${p.symbol.replace('USDT','')}</strong> ${p.side.toUpperCase()}</span>
                    <span>$${fmtP(p.entry_price)} → $${fmtP(p.current_price)}</span>
                    <span style="color:${pc}">${p.pnl>=0?'+':''}$${p.pnl.toFixed(2)} (${p.pnl_pct.toFixed(1)}%)</span>
                    <button class="btn btn-sm" onclick="paperClose('${p.symbol}');setTimeout(loadPortfolio,500)">Close</button>
                </div>`;
            });
            h += '</div>';
        }
        h += `<div style="padding:8px 14px;display:flex;gap:6px">
            <button class="btn btn-primary btn-sm" onclick="paperTrade('long')">Buy ${activeSymbol.replace('USDT','')}</button>
            <button class="btn btn-sm" style="border-color:var(--red);color:var(--red)" onclick="paperTrade('short')">Short ${activeSymbol.replace('USDT','')}</button>
            <button class="btn btn-sm btn-ghost" onclick="fetch('/api/paper/reset',{method:'POST'});setTimeout(loadPortfolio,300)">Reset</button>
        </div>`;
        el.innerHTML = h;
    } catch(e) {}
}

// ── SL/TP Calculator ──
async function loadSLTP(sym) {
    try {
        const r = await fetch(`/api/coin/${sym || activeSymbol}/sl-tp`);
        const j = await r.json();
        if (!j.ok) return;
        const sec = document.getElementById('sltpSection');
        sec.style.display = '';
        document.getElementById('sltpTP').textContent = '$' + fmtP(j.take_profit);
        document.getElementById('sltpSL').textContent = '$' + fmtP(j.stop_loss);
        document.getElementById('sltpTPpct').textContent = '+' + j.tp_distance_pct + '%';
        document.getElementById('sltpSLpct').textContent = '-' + j.sl_distance_pct + '%';
        document.getElementById('sltpRR').textContent = j.risk_reward.toFixed(1);
        // S/R context
        let sr = '';
        if (j.nearest_support) sr += `S: $${fmtP(j.nearest_support)}`;
        if (j.nearest_resistance) sr += `${sr ? ' | ' : ''}R: $${fmtP(j.nearest_resistance)}`;
        if (sr) sr = `Nearest S/R: ${sr}`;
        document.getElementById('sltpSR').textContent = sr;
    } catch(e) {}
}

// ── Multi-Timeframe UI ──
async function loadMultiTF() {
    const el = document.getElementById('mtfContent');
    el.innerHTML = '<div style="padding:4px 14px;font-size:10px;color:var(--text-muted)">Loading 4 timeframes...</div>';
    try {
        const r = await fetch(`/api/coin/${activeSymbol}/multi-tf`);
        const j = await r.json();
        if (!j.ok) { el.innerHTML = `<div style="padding:4px 14px;font-size:10px;color:var(--red)">${j.error}</div>`; return; }

        let h = '';
        const tfs = ['15m', '1h', '4h', '1d'];
        for (const tf of tfs) {
            const d = j.timeframes[tf];
            if (!d) continue;
            const dir = d.direction || 'ERROR';
            const cls = dir.includes('BUY') ? 'buy' : dir.includes('SELL') ? 'sell' : dir === 'HOLD' ? 'hold' : 'error';
            const confColor = d.confidence > 60 ? 'var(--green)' : d.confidence > 40 ? 'var(--yellow)' : 'var(--red)';
            h += `<div class="mtf-row" onclick="document.querySelector('.tf-pill[data-tf=\\'${tf}\\']')?.click()">
                <span class="mtf-tf">${tf}</span>
                <span class="mtf-badge ${cls}">${dir}</span>
                <div class="mtf-conf"><div class="mtf-conf-fill" style="width:${d.confidence || 0}%;background:${confColor}"></div></div>
                <span class="mtf-rsi" style="color:${rsiCol(d.rsi)}">${d.rsi || '--'}</span>
            </div>`;
        }

        // Alignment badge
        const aCls = j.alignment || 'neutral';
        h += `<div class="mtf-alignment ${aCls}">${j.alignment_pct}% ${aCls.charAt(0).toUpperCase() + aCls.slice(1)}</div>`;

        el.innerHTML = h;
    } catch(e) { el.innerHTML = '<div style="padding:4px 14px;font-size:10px;color:var(--red)">Failed</div>'; }
}

// ── Watchlist Presets ──
const PRESETS = {
    top10: ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","TRXUSDT","LINKUSDT"],
    defi: ["UNIUSDT","AAVEUSDT","MKRUSDT","CRVUSDT","SUSHIUSDT","COMPUSDT"],
    layer2: ["ARBUSDT","OPUSDT","MATICUSDT","STRKUSDT","IMXUSDT"],
    meme: ["DOGEUSDT","SHIBUSDT","PEPEUSDT","FLOKIUSDT","BONKUSDT","WIFUSDT"],
    ai: ["FETUSDT","RENDERUSDT","AXSUSDT","SANDUSDT","MANAUSDT"],
};

function applyPreset(key) {
    if (!key) return;
    // Check custom presets first
    const custom = JSON.parse(localStorage.getItem('btcdump_presets') || '{}');
    const coins = custom[key] || PRESETS[key];
    if (!coins) return;
    watchlist = [...coins];
    saveWL(); renderWatchlistMgr(); syncWL();
    toast(`Preset applied: ${coins.length} coins`, 'success');
    document.getElementById('presetSelect').value = '';
    loadWatchlistOverview();
}

function saveCustomPreset() {
    const name = prompt('Preset name:');
    if (!name) return;
    const custom = JSON.parse(localStorage.getItem('btcdump_presets') || '{}');
    custom[name] = [...watchlist];
    localStorage.setItem('btcdump_presets', JSON.stringify(custom));
    // Add to select
    const sel = document.getElementById('presetSelect');
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    sel.appendChild(opt);
    toast(`Preset "${name}" saved`, 'success');
}

// Load custom presets into dropdown on init
function loadCustomPresets() {
    const custom = JSON.parse(localStorage.getItem('btcdump_presets') || '{}');
    const sel = document.getElementById('presetSelect');
    if (!sel) return;
    for (const name of Object.keys(custom)) {
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name;
        sel.appendChild(opt);
    }
}

// ── Compare Tab Switching ──
function switchCompareTab(tab) {
    document.querySelectorAll('.compare-tabs .filter-pill').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById('compareGridContent').style.display = tab === 'grid' ? '' : 'none';
    document.getElementById('compareCorrelationContent').style.display = tab === 'correlation' ? '' : 'none';
    document.getElementById('compareScannerContent').style.display = tab === 'scanner' ? '' : 'none';
    if (tab === 'correlation') loadCorrelation();
}

// ── Market Scanner ──
async function runScanner() {
    const condition = document.getElementById('scannerCondition').value;
    const statusEl = document.getElementById('scannerStatus');
    const resultsEl = document.getElementById('scannerResults');
    statusEl.textContent = 'Scanning top 60 coins...';
    resultsEl.innerHTML = '';

    try {
        const r = await fetch(`/api/scanner?condition=${condition}&limit=20`);
        const j = await r.json();
        if (!j.ok) { statusEl.textContent = j.error; return; }

        statusEl.textContent = `Found ${j.results.length} matches`;

        if (!j.results.length) {
            resultsEl.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted)">No coins match this condition right now</div>';
            return;
        }

        let h = '<table class="comparison-table"><thead><tr><th>Coin</th><th>Price</th><th>24h%</th><th>RSI</th><th>ADX</th><th>Vol</th><th>MACD</th><th>Action</th></tr></thead><tbody>';
        j.results.forEach(c => {
            const chg = c.priceChangePercent || 0;
            h += `<tr>
                <td><strong>${c.baseAsset}</strong></td>
                <td>$${fmtP(c.lastPrice)}</td>
                <td class="change-cell ${chg >= 0 ? 'up' : 'down'}">${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%</td>
                <td style="color:${rsiCol(c.rsi)}">${c.rsi}</td>
                <td>${c.adx}</td>
                <td>${c.volume_ratio}x</td>
                <td>${c.macd_bullish ? '<span style="color:var(--green)">Bull</span>' : '<span style="color:var(--red)">Bear</span>'}</td>
                <td><button class="btn btn-sm btn-primary" onclick="selectCoin('${c.symbol}');switchMode('single')">Analyze</button></td>
            </tr>`;
        });
        h += '</tbody></table>';
        resultsEl.innerHTML = h;
    } catch(e) {
        statusEl.textContent = 'Scanner failed';
    }
}

// ── Correlation Matrix ──
async function loadCorrelation() {
    try {
        const r = await fetch('/api/correlation');
        const j = await r.json();
        if (j.ok && j.symbols?.length) {
            drawCorrelationHeatmap(document.getElementById('correlationCanvas'), j.matrix, j.symbols);
        } else {
            toast(j.error || 'Need 2+ coins in watchlist', 'error');
        }
    } catch(e) { toast('Correlation load failed', 'error'); }
}

// ── S/R Levels ──
async function loadSRLevels(sym) {
    try {
        const r = await fetch(`/api/coin/${sym || activeSymbol}/sr-levels`);
        const j = await r.json();
        if (j.ok && j.levels?.length) {
            TVChart.setSRLevels(j.levels);
        }
    } catch(e) {}
}

// ── Fear & Greed ──
async function loadFearGreed() {
    try {
        const r = await fetch('/api/fear-greed');
        const j = await r.json();
        if (!j.ok && !j.value) return;
        const badge = document.getElementById('fngBadge');
        const valEl = document.getElementById('fngValue');
        const clsEl = document.getElementById('fngClass');
        valEl.textContent = j.value;
        clsEl.textContent = j.classification;
        // Color class
        badge.className = 'fng-badge';
        if (j.value <= 25) badge.classList.add('extreme-fear');
        else if (j.value <= 40) badge.classList.add('fear');
        else if (j.value <= 60) badge.classList.add('neutral');
        else if (j.value <= 75) badge.classList.add('greed');
        else badge.classList.add('extreme-greed');
    } catch(e) {}
}

// ── Anomaly Detection ──
async function checkAnomalies(sym) {
    try {
        const r = await fetch(`/api/coin/${sym || activeSymbol}/anomalies`);
        const j = await r.json();
        if (j.ok && (j.volume_anomaly || j.price_anomaly)) {
            toast(`Anomaly: ${j.description}`, 'error');
        }
    } catch(e) {}
}

// ── Init ──
async function init(){
    try{const r=await fetch('/api/providers');providerStatus=await r.json();}catch(e){}
    try{const r=await fetch('/api/signal/cached');const j=await r.json();if(j.ok&&j.data?.current_price)updateSignalUI(j.data);}catch(e){}
    await syncWL(); loadTickerStrip(); loadCandlestickChart(); updateFcContext(); connectWS();
    loadFearGreed(); loadCustomPresets(); checkAnomalies();
    addDiscMsg('system','BTCDump v5.0 | Ctrl+K: search | S/C: mode | R: refresh | Live prices active');
}
init();
