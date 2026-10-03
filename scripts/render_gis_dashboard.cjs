const fs=require('fs'),path=require('path'),assert=require('assert');
const {pathToFileURL}=require('url'),{chromium}=require('playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'output/improvement_v4_gis');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const page=await browser.newPage({viewport:{width:1680,height:1000},deviceScaleFactor:3});
 const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 const url=pathToFileURL(path.join(out,'web_dashboard/index.html')).href;
 await page.goto(url);await page.waitForFunction(()=>window.dashboardReady===true);
 const checks=[];
 for(const language of ['zh','en']){
  if(await page.evaluate(()=>window.dashboardState.lang)!==language)await page.locator('#language').click();
  for(const view of ['overview','crowd','tickets','traffic','emergency','service']){
   await page.locator(`[data-view="${view}"]`).click();
   const r=await page.evaluate(()=>({lang:window.dashboardState.lang,view:window.dashboardState.view,inflow:Number(document.querySelector('#inflow').textContent),queue:Number(document.querySelector('#queue').textContent),nodes:document.querySelectorAll('#scenic-map [data-node]').length,
    overflow:[...document.querySelectorAll('.panel-title,.kpi,.topbar,.bottom,.activity,.node-popup')].filter(e=>e.scrollWidth>e.clientWidth+1).map(e=>e.className)}));
   assert.equal(r.lang,language);assert.equal(r.view,view);assert.equal(r.inflow,302);assert.equal(r.queue,323);assert.equal(r.nodes,24);assert.deepEqual(r.overflow,[]);checks.push(r);
  }
 }
 await page.locator('[data-view="overview"]').click();
 for(const [t,n] of [[0,2],[48,15],[71,20],[36,0]]){
  await page.locator('#step').fill(String(t));await page.locator('#step').dispatchEvent('input');await page.locator('#node').selectOption(String(n));
  const r=await page.evaluate(()=>({state:window.dashboardState,queue:Number(document.querySelector('#queue').textContent),expected:window.currentReplayState.queued}));
  assert.equal(r.state.step,t);assert.equal(r.state.node,n);assert.equal(r.queue,r.expected);if(t===71)assert((await page.locator('.node-popup').innerText()).includes('N/A'));
 }
 await page.locator('#scenic-map [data-node="8"]').click();assert.equal(await page.locator('#node').inputValue(),'8');
 for(const layer of ['routes','heat','resources']){
  const before=await page.locator('#scenic-map').innerHTML();await page.locator(`[data-layer="${layer}"]`).uncheck();
  assert.notEqual(await page.locator('#scenic-map').innerHTML(),before);await page.locator(`[data-layer="${layer}"]`).check();
 }
 await page.locator('#zoom-in').click();assert((await page.evaluate(()=>window.dashboardState.zoom))>1);await page.locator('#reset-map').click();
 await page.locator('[data-event="0"]').click();await page.locator('#advance-event').click();assert(await page.locator('#modal').innerText().then(s=>s.includes('In progress')));
 await page.locator('#advance-event').click();assert(await page.locator('#advance-event').isDisabled());await page.locator('#close-modal').click();
 await page.locator('#play').click();await page.waitForFunction(()=>window.dashboardState.step>=37,{},{timeout:5000});await page.locator('#play').click();
 await page.locator('#step').fill('36');await page.locator('#step').dispatchEvent('input');
 const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#export').click()]);await download.saveAs(path.join(out,'export_verified.json'));
 const exported=JSON.parse(fs.readFileSync(path.join(out,'export_verified.json'),'utf8'));assert.equal(exported.simulation_record.queued,323);assert.equal(exported.illustrative_scenario.incidents[0].status,2);
 // Fresh state for manuscript screenshots; scenario interactions are not presented as recorded results.
 await page.goto(url);await page.waitForFunction(()=>window.dashboardReady);
 const geometry=await page.evaluate(()=>{
  const q=s=>document.querySelector(s).getBoundingClientRect(),overlap=(a,b)=>Math.max(0,Math.min(a.right,b.right)-Math.max(a.left,b.left))*Math.max(0,Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top));
  const clip=[];for(const s of ['.weather-foot','.table-wrap','.node-popup','.map-legend']){const e=q(s),p=q(s==='.weather-foot'?'.weather-panel':s==='.table-wrap'?'.dispatch-panel':'.map-panel');if(e.bottom>p.bottom+1||e.right>p.right+1)clip.push(s);}
  return {width:q('#dashboard').width,height:q('#dashboard').height,transferRows:document.querySelectorAll('#transfers tr').length,transferClientHeight:document.querySelector('.table-wrap').clientHeight,transferScrollHeight:document.querySelector('.table-wrap').scrollHeight,clip,headerOverlap:overlap(q('.brand'),q('.nav'))};
 });
 assert.equal(geometry.transferRows,5);assert(geometry.transferScrollHeight<=geometry.transferClientHeight+1);assert.deepEqual(geometry.clip,[]);assert.equal(geometry.headerOverlap,0);
 await page.locator('#dashboard').screenshot({path:path.join(out,'Scenic_Command_Overview_ZH.png')});
 await page.locator('[data-view="emergency"]').click();await page.locator('#dashboard').screenshot({path:path.join(out,'Scenic_Command_Emergency_ZH.png')});
 await page.locator('[data-view="overview"]').click();await page.locator('#language').click();
 await page.locator('#dashboard').screenshot({path:path.join(out,'Scenic_Command_Overview_EN.png')});
 const englishGeometry=await page.evaluate(()=>{const a=document.querySelector('.brand h1').getBoundingClientRect(),b=document.querySelector('.nav').getBoundingClientRect();return {brandRight:a.right,navLeft:b.left};});assert(englishGeometry.brandRight<=englishGeometry.navLeft);
 await page.setViewportSize({width:1366,height:768});await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth&&document.documentElement.scrollHeight<=innerHeight,{},{timeout:4000});const fit=await page.evaluate(()=>({w:document.documentElement.scrollWidth,h:document.documentElement.scrollHeight,vw:innerWidth,vh:innerHeight}));assert(fit.w<=fit.vw&&fit.h<=fit.vh);
 assert.deepEqual(errors,[]);assert(!requests.some(r=>!r.startsWith('file:')&&!r.startsWith('blob:')));
 fs.writeFileSync(path.join(out,'BROWSER_QA.json'),JSON.stringify({status:'PASS',views:checks.length,checks,geometry,englishGeometry,scaledViewport:fit,errors,externalNetworkRequests:0,exportSeparatesSimulationAndExamples:true,incidentWorkflowVerified:true,layerControlsVerified:true,timeNodePlayAndZoomVerified:true,screenshotPixels:[5040,3000]},null,2));
 await browser.close();console.log(JSON.stringify({status:'PASS',views:checks.length,geometry}));
})().catch(e=>{console.error(e);process.exit(1)});
