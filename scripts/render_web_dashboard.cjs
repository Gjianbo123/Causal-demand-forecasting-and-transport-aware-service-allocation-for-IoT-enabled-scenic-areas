const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'output/improvement_v4_web');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const page=await browser.newPage({viewport:{width:1280,height:1400},deviceScaleFactor:3});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const requests=[];page.on('request',r=>requests.push(r.url()));
 await page.goto(pathToFileURL(path.join(out,'web_dashboard/index.html')).href);
 await page.waitForFunction(()=>window.dashboardReady===true);
 const check=async(t,node)=>{
  await page.locator('#step').fill(String(t));await page.locator('#step').dispatchEvent('input');
  await page.locator('#node').selectOption(String(node));
  return page.evaluate(()=>({t:window.currentReplayState.t,node:document.querySelector('#node').value,
   inflow:Number(document.querySelector('#inflow').textContent),queue:Number(document.querySelector('#queue').textContent),
   rows:document.querySelectorAll('#transfers tr').length,expectedRows:window.currentReplayState.moves.length,
   ledgerRows:document.querySelectorAll('#ledger tr').length,graphNodes:document.querySelectorAll('#graph [data-node]').length}));
 };
 const checks=[];for(const [t,n]of [[0,1],[48,15],[71,20],[36,0]]){
  const r=await check(t,n);if(r.t!==t||r.node!==String(n)||r.rows!==r.expectedRows||r.ledgerRows!==4||r.graphNodes!==24)throw Error('UI state mismatch');checks.push(r);
 }
 const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#export').click()]);
 await download.saveAs(path.join(out,'web_dashboard/export_snapshot_t36.json'));
 const exported=JSON.parse(fs.readFileSync(path.join(out,'web_dashboard/export_snapshot_t36.json'),'utf8'));
 if(exported.frame.queued!==323||exported.frame.stationary!==80||exported.frame.in_transit!==2)throw Error('Export mismatch');
 const metrics=await page.evaluate(()=>({width:document.querySelector('#dashboard').getBoundingClientRect().width,height:document.querySelector('#dashboard').getBoundingClientRect().height,
  transferScrollHeight:document.querySelector('.row-scroll').scrollHeight,transferClientHeight:document.querySelector('.row-scroll').clientHeight,
  overflow:[...document.querySelectorAll('.card-head,.kpi,.legend,.toolbar,.replay,.footer')].filter(e=>e.scrollWidth>e.clientWidth+1).map(e=>e.className)}));
 if(errors.length||metrics.overflow.length||metrics.transferScrollHeight>metrics.transferClientHeight+1)throw Error(JSON.stringify({errors,metrics}));
 if(requests.some(x=>!x.startsWith('file:')&&!x.startsWith('blob:')))throw Error('Unexpected network request');
 await page.locator('#dashboard').screenshot({path:path.join(out,'Web_dashboard.png')});
 fs.writeFileSync(path.join(out,'BROWSER_QA.json'),JSON.stringify({status:'PASS',checks,metrics,errors,requests,deviceScaleFactor:3,exportVerified:true},null,2));
 await browser.close();console.log(JSON.stringify({status:'PASS',metrics,png:path.join(out,'Web_dashboard.png')}));
})().catch(e=>{console.error(e);process.exit(1)});
