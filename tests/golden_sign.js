// Verbatim port of the ORIGINAL TypeScript signing code (core.ts request()) - independent oracle.
const crypto = require('crypto');
const md5 = s => crypto.createHash('md5').update(s).digest('hex');
const FAKE = { device_platform:"web", biz_id:"3", app_id:"3001", version_code:"22201", uuid:null, device_id:null,
  os_name:"Mac", browser_name:"chrome", device_memory:8, cpu_core_num:11, browser_language:"zh-CN",
  browser_platform:"MacIntel", user_id:null, screen_width:1920, screen_height:1080, unix:null, lang:"zh", token:null };
const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = cases.map(c => {
  const timestamp = c.ts;
  const unix = `${c.ts * 1000}`;
  const userData = Object.assign({}, FAKE);
  userData.uuid = c.user_id; userData.device_id = c.device_id || undefined; userData.user_id = c.user_id;
  userData.unix = unix; userData.token = c.token;
  let queryStr = "";
  for (let key in userData) { if (userData[key] === undefined) continue; queryStr += `&${key}=${userData[key]}`; }
  queryStr = queryStr.substring(1);
  const dataJson = JSON.stringify(c.data || {});
  const fullUri = `${c.uri}${c.uri.lastIndexOf("?") != -1 ? "&" : "?"}${queryStr}`;
  const yy = md5(`${encodeURIComponent(fullUri)}_${dataJson}${md5(unix)}ooui`);
  const signature = md5(`${timestamp}${c.token}${dataJson}`);
  return { fullUri, dataJson, yy, signature };
});
process.stdout.write(JSON.stringify(out));
