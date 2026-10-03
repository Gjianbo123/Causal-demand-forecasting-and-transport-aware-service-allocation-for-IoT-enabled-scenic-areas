/* Original, deterministic cartographic illustration. Layout is schematic, not surveyed GIS. */
(function () {
  'use strict';
  const positions = [
    [280,228],[393,194],[325,278],[192,296],
    [617,165],[725,216],[808,302],[664,283],
    [794,465],[747,549],[614,553],[650,436],
    [300,437],[387,514],[207,554],[178,451],
    [258,362],[803,372],[645,605],[359,603],
    [96,371],[926,379],[847,612],[165,638]
  ];
  window.scenicPositions = positions;
  const namesZh = ['云岭观景台','听风台','云溪茶亭','古木栈道','青岚峰','竹海书院','观瀑台','林间驿站','峡谷观景台','溪谷营地','水岸剧场','湖畔码头','古村戏台','文化展馆','花溪步道','古桥遗址','西区服务站','东区服务站','南区服务站','古村服务站','西门','东门','南门','古村入口'];
  const namesEn = ['Cloud ridge','Wind terrace','Tea pavilion','Forest walk','Green peak','Bamboo grove','Falls','Rest stop','Valley view','Camp','Lakeside stage','Lakeside pier','Village','Culture hall','Flower trail','Old bridge','West hub','East hub','South hub','Village hub','West gate','East gate','South gate','Village gate'];
  const nodeId = i => i < 16 ? 'S'+(i+1) : i < 20 ? 'H'+(i-15) : 'G'+(i-19);
  const esc = s => String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const lake = 'M379 288 C408 267 435 281 460 273 C483 256 512 270 524 295 C543 318 570 300 594 322 C607 339 590 350 610 371 C639 393 630 415 602 427 C577 437 570 460 542 456 C514 453 500 467 476 450 C459 436 445 449 425 430 C410 414 389 413 377 391 C365 368 344 363 355 340 C369 321 354 305 379 288 Z';
  const peaks = [[130,98,83,55,1],[330,78,98,57,1.3],[527,73,94,59,1.1],[734,79,108,76,1.2],[897,134,64,106,1.2],[65,260,66,120,.9],[923,463,84,149,1.1],[98,520,82,72,.95],[430,590,58,61,.6],[504,664,120,48,.9]];
  function altitude(x,y) {
    let h=.11*Math.sin(x/75+y/127)+.07*Math.sin(x/38-y/96);
    peaks.forEach(p=>{h+=p[4]*Math.exp(-((x-p[0])**2/(2*p[2]**2)+(y-p[1])**2/(2*p[3]**2)));});
    return h;
  }
  function contour(threshold) {
    let path=''; const step=13;
    for(let y=0;y<690;y+=step) for(let x=0;x<1000;x+=step) {
      const corners=[[x,y],[x+step,y],[x+step,y+step],[x,y+step]];
      const h=corners.map(p=>altitude(...p)); const cuts=[];
      for(let j=0;j<4;j++) {const k=(j+1)%4;if((h[j]<threshold)!==(h[k]<threshold)) {
        const t=(threshold-h[j])/(h[k]-h[j]);cuts.push([corners[j][0]+t*(corners[k][0]-corners[j][0]),corners[j][1]+t*(corners[k][1]-corners[j][1])]);
      }}
      for(let j=0;j+1<cuts.length;j+=2) path+=`M${cuts[j][0].toFixed(1)},${cuts[j][1].toFixed(1)}L${cuts[j+1][0].toFixed(1)},${cuts[j+1][1].toFixed(1)}`;
    }
    return path;
  }
  function baseMap() {
    let s=`<defs>
      <linearGradient id="land" x2=".8" y2="1"><stop stop-color="#102a2d"/><stop offset=".5" stop-color="#163b37"/><stop offset="1" stop-color="#0a2228"/></linearGradient>
      <linearGradient id="water" x2=".6" y2="1"><stop stop-color="#166d88"/><stop offset=".48" stop-color="#144e68"/><stop offset="1" stop-color="#123348"/></linearGradient>
      <linearGradient id="ridge" x2=".3" y2="1"><stop stop-color="#5f7465" stop-opacity=".8"/><stop offset=".5" stop-color="#24473d"/><stop offset="1" stop-color="#102b2c"/></linearGradient>
      <radialGradient id="hill"><stop stop-color="#779376" stop-opacity=".28"/><stop offset=".55" stop-color="#41694e" stop-opacity=".2"/><stop offset="1" stop-color="#081d24" stop-opacity="0"/></radialGradient>
      <radialGradient id="hot"><stop stop-color="#ffc86c" stop-opacity=".42"/><stop offset=".45" stop-color="#f89b42" stop-opacity=".2"/><stop offset="1" stop-color="#e6643c" stop-opacity="0"/></radialGradient>
      <radialGradient id="edgefade"><stop offset=".5" stop-color="#001722" stop-opacity="0"/><stop offset="1" stop-color="#00101b" stop-opacity=".65"/></radialGradient>
      <filter id="mapGlow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3"/></filter>
      <marker id="moveArrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="#63e2df"/></marker>
      <clipPath id="waterClip"><path d="${lake}"/></clipPath>
      <g id="pine"><path d="M0-8L-5 1H-3L-7 6H7L3 1H5Z" fill="#285447"/><path d="M0-8L0 6H7L3 1H5Z" fill="#143e35"/><path d="M0 5V9" stroke="#567060" stroke-width="1"/></g>
      <g id="deciduous"><circle cy="-2" r="5" fill="#2b5848"/><circle cx="-2" cy="-4" r="3" fill="#406b50"/><path d="M0 3V8" stroke="#566851"/></g>
    </defs><rect width="1000" height="690" fill="url(#land)"/>`;
    // Broad landforms use fixed geometry; contours below arise from a deterministic height field.
    s+=`<path d="M0 169L60 101L100 125L166 23L225 111L286 63L343 9L413 110L468 52L521 7L587 116L645 67L702 16L773 119L825 34L889 111L947 63L1000 159V0H0Z" fill="#061d28" opacity=".6"/>`;
    peaks.forEach((p,i)=>{
      s+=`<ellipse cx="${p[0]}" cy="${p[1]}" rx="${p[2]*1.5}" ry="${p[3]*1.5}" fill="url(#hill)"/>`;
      if(i<5)s+=`<path d="M${p[0]-p[2]} ${p[1]+p[3]}Q${p[0]-15} ${p[1]-30} ${p[0]} ${p[1]-p[3]*.66}Q${p[0]+28} ${p[1]+13} ${p[0]+p[2]} ${p[1]+p[3]}Z" fill="url(#ridge)" opacity=".52"/><path d="M${p[0]} ${p[1]-p[3]*.66}l-9 32 19 19 -11 35" fill="none" stroke="#9dafa1" stroke-opacity=".22" stroke-width="2"/>`;
    });
    for(let k=0;k<16;k++)s+=`<path d="${contour(.17+k*.1)}" fill="none" stroke="${k%4===0?'#7e9b83':'#5d826f'}" stroke-opacity="${k%4===0?'.25':'.15'}" stroke-width="${k%4===0?'1.15':'.65'}"/>`;
    // Forest clusters are kept away from the central water basin.
    let seed=71831;const rng=()=>{seed=(seed*16807)%2147483647;return(seed-1)/2147483646;};
    for(let i=0;i<630;i++) {
      const x=20+rng()*960,y=25+rng()*640;
      if(((x-495)/165)**2+((y-363)/113)**2<1.05)continue;
      if(positions.some(p=>Math.hypot(x-p[0],y-p[1])<24))continue;
      const scale=(.6+rng()*.55).toFixed(2);
      s+=`<use href="#${i%4?'pine':'deciduous'}" transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) scale(${scale})" opacity="${(.35+rng()*.38).toFixed(2)}"/>`;
    }
    s+=`<path d="M929 22C878 86 878 150 847 189C825 214 837 247 828 279" fill="none" stroke="#15374b" stroke-width="16"/><path d="M929 22C878 86 878 150 847 189C825 214 837 247 828 279" fill="none" stroke="#3d6d7c" stroke-width="4" opacity=".65"/>`;
    s+=`<path d="${lake}" fill="#061d29" stroke="#82a98d" stroke-opacity=".22" stroke-width="15"/><path d="${lake}" fill="url(#water)" stroke="#70aca9" stroke-opacity=".65" stroke-width="2"/>`;
    s+='<g clip-path="url(#waterClip)" fill="none" stroke="#66b9c3" stroke-opacity=".13">';
    for(let y=283;y<466;y+=10)s+=`<path d="M340 ${y}Q425 ${y-8} 485 ${y}T652 ${y-2}" stroke-width="1"/>`;
    s+='</g>';
    s+=`<path d="M476 347q20-20 37 0l14 15-8 12-38 0-11-13Z" fill="#274f40" stroke="#719c7c" stroke-opacity=".65"/><path d="M487 352h21v12h-21Z" fill="#6c8170"/><path d="M482 352l15-10 16 10Z" fill="#b2aa82"/><path d="M499 374v44" stroke="#9ca184" stroke-width="4"/><text x="493" y="411" text-anchor="middle" fill="#94c7cc" font-size="21" letter-spacing="4" id="lake-name"></text>`;
    // Perimeter service road is a schematic contextual layer.
    const road='M96 371C64 420 109 490 116 544S160 623 205 638C257 649 310 647 359 645S539 647 645 642S810 674 875 611C948 541 934 459 926 379C912 295 892 242 845 190C788 130 712 112 635 105C525 72 348 89 260 104C165 117 111 230 96 371Z';
    s+=`<path d="${road}" fill="none" stroke="#062438" stroke-width="13"/><path d="${road}" fill="none" stroke="#38778f" stroke-width="4"/><path d="${road}" fill="none" stroke="#7fc8df" stroke-opacity=".6" stroke-width="1.5" stroke-dasharray="6 11"/>`;
    // District perimeters reinforce a GIS layer appearance without asserting surveyed polygons.
    s+=`<g fill="none" stroke-width="1" stroke-dasharray="5 7" opacity=".5"><path d="M137 148Q301 78 440 171L420 254 331 336 150 335Z" stroke="#63b2ac"/><path d="M548 138Q724 95 851 244L839 329 661 330 537 256Z" stroke="#63b2ac"/><path d="M670 409L858 421 847 568 601 592 563 499Z" stroke="#729cae"/><path d="M143 409L297 407 426 491 403 579 172 600Z" stroke="#c0a96b"/></g>`;
    s+='<rect width="1000" height="690" fill="url(#edgefade)" pointer-events="none"/>';
    return s;
  }
  let base;
  const hubPaths={
    '16-17':'M258 362C304 340 323 319 339 290C367 232 511 226 556 262C621 314 726 318 803 372',
    '17-18':'M803 372C845 400 859 476 805 536S709 570 645 605',
    '18-19':'M645 605C552 574 461 581 359 603',
    '16-19':'M258 362C240 401 277 433 306 469S354 555 359 603'
  };
  function edgePath(i,j) {
    const key=[i,j].sort((a,b)=>a-b).join('-');if(hubPaths[key])return hubPaths[key];
    const a=positions[i],b=positions[j];const mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2;
    const bend=((i*11+j*7)%3-1)*15;
    return `M${a[0]} ${a[1]}Q${mx+bend} ${my-bend} ${b[0]} ${b[1]}`;
  }
  window.scenicMapMarkup=function(frame,nodeIndex,language,visible) {
    const zh=language==='zh',v=Object.assign({routes:true,heat:true,resources:true},visible||{});
    if(!base)base=baseMap();let s=base;
    const queues=(frame&&frame.queue_by_node)||new Array(24).fill(0);
    s+=`<g font-family="Microsoft YaHei,Arial,sans-serif">`;
    if(v.routes) {
      const edges=(window.scenicReplayData&&window.scenicReplayData.edges)||[];
      s+='<g fill="none" stroke-linecap="round">';
      edges.forEach(e=>{if(!positions[e[0]]||!positions[e[1]])return;const d=edgePath(e[0],e[1]);s+=`<path d="${d}" stroke="#081f27" stroke-width="8"/><path d="${d}" stroke="#c8b681" stroke-width="2.5" stroke-dasharray="4 5" opacity=".9"/>`;});
      s+='</g>';
    }
    if(v.heat)queues.forEach((q,i)=>{if(q>=20){const r=30+Math.min(q,80)*.42;s+=`<circle cx="${positions[i][0]}" cy="${positions[i][1]}" r="${r}" fill="url(#hot)" pointer-events="none"/>`;}});
    const districtLabels=zh?['云岭观景区','竹海休闲区','峡谷游览区','古村文化区']:['CLOUD RIDGE','BAMBOO GROVE','VALLEY TRAIL','HERITAGE VILLAGE'];
    [[430,134],[711,126],[680,505],[281,499]].forEach((p,i)=>{
      const w=zh?128:(i===3?190:162);s+=`<g transform="translate(${p[0]} ${p[1]})"><rect x="${-w/2}" y="-19" width="${w}" height="29" rx="4" fill="#092f39" fill-opacity=".91" stroke="#438482" stroke-opacity=".7"/><text text-anchor="middle" y="1" fill="#a4dcd2" font-size="${zh?18:15}" font-weight="600" letter-spacing="1">${districtLabels[i]}</text></g>`;
    });
    s+=`<text x="495" y="404" text-anchor="middle" fill="#98c3c8" font-size="${zh?19:15}" letter-spacing="${zh?4:2}">${zh?'青岚湖':'QINGLAN LAKE'}</text>`;
    if(v.resources&&frame&&frame.moves) {
      s+='<g fill="none" stroke="#63e2df" stroke-width="2" marker-end="url(#moveArrow)" opacity=".92">';
      frame.moves.forEach((m,k)=>{const a=positions[m.origin],b=positions[m.destination];if(!a||!b)return;const dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy)||1;const bend=Math.min(65,len*.2)*(k%2?-1:1);const cx=(a[0]+b[0])/2-dy/len*bend,cy=(a[1]+b[1])/2+dx/len*bend;s+=`<path d="M${a[0]} ${a[1]}Q${cx.toFixed(1)} ${cy.toFixed(1)} ${(b[0]-dx/len*16).toFixed(1)} ${(b[1]-dy/len*16).toFixed(1)}" stroke-dasharray="7 5"/>`;});s+='</g>';
    }
    positions.forEach((p,i)=>{
      const selected=i===nodeIndex,q=queues[i]||0,service=i<16,hub=i>=16&&i<20;
      const color=selected?'#75f7ee':q>=40?'#f6b15a':service?'#d5bd83':hub?'#6ad3ba':'#7dbde7';
      const name=zh?namesZh[i]:namesEn[i],label=`${nodeId(i)} ${name}`;
      const w=zh?(name.length*16+34):Math.min(178,name.length*8+35);
      const ly=i===23?-29:22;
      s+=`<g data-node="${i}" role="button" tabindex="0" aria-label="${esc(label)}" style="cursor:pointer" transform="translate(${p[0]} ${p[1]})"><title>${esc(label)} | ${zh?'排队':'Queue'}: ${q}</title>`;
      if(selected)s+=`<circle r="28" fill="#61eeed" opacity=".08"/><circle r="22" fill="none" stroke="#76f1e5" stroke-width="1.6"/><circle r="26" fill="none" stroke="#76f1e5" stroke-opacity=".4" stroke-dasharray="3 6"/>`;
      s+=`<circle r="15" fill="#021c28" opacity=".8"/>`;
      if(service)s+=`<circle r="10" fill="#34453a" stroke="${color}" stroke-width="1.7"/><path d="M0-7L2-2 7-2 3 1 4 6 0 3-4 6-3 1-7-2-2-2Z" fill="${color}"/>`;
      else if(hub)s+=`<rect x="-11" y="-11" width="22" height="22" rx="5" fill="#155b59" stroke="${color}" stroke-width="1.5"/><path d="M-6-4H6M-4-4V5M4-4V5M-7 5H7M0-8V-4" stroke="#ccf0db" stroke-width="2"/>`;
      else s+=`<rect x="-12" y="-11" width="24" height="22" rx="4" fill="#174869" stroke="${color}" stroke-width="1.5"/><path d="M-6 7V-5H6V7M-2-5V5M-7-8H7" stroke="#cce8f4" stroke-width="2" fill="none"/>`;
      s+=`<rect x="${-w/2}" y="${ly-13}" width="${w}" height="23" rx="3" fill="${selected?'#124b51':'#092733'}" fill-opacity=".94" stroke="${color}" stroke-opacity="${selected?'.9':'.42'}"/><text x="0" y="${ly+3}" text-anchor="middle" font-size="15" fill="${selected?'#e3fff8':'#d4e3dc'}">${esc(label)}</text>`;
      if(service&&q>=20)s+=`<g transform="translate(18 -17)"><rect x="-9" y="-11" width="29" height="20" rx="9" fill="${q>=40?'#855321':'#284b46'}" stroke="${color}" stroke-opacity=".65"/><text x="5" y="3" fill="#ffdf9b" text-anchor="middle" font-size="13" font-weight="700">${q}</text></g>`;
      s+='</g>';
    });
    s+=`<g transform="translate(39 49)"><path d="M0-25L-9 4 0-1 9 4Z" fill="#c6e5e2"/><path d="M0-25V-1L9 4Z" fill="#5d949a"/><text y="-32" text-anchor="middle" fill="#cce6e3" font-size="16" font-weight="600">N</text></g>`;
    s+='</g>';return s;
  };
})();
