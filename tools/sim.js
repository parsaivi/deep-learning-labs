// load common math + a lesson's @sim region into this scope
const fs=require("fs");
module.exports=function(dir,file){const c=fs.readFileSync(dir+"/common.jsx","utf8");const l=fs.readFileSync(dir+"/"+file,"utf8");
return (c.slice(c.indexOf("// @math-start"),c.indexOf("// @math-end"))+"\n"+l.slice(l.indexOf("// @sim-start"),l.indexOf("// @sim-end"))).replace(/^const /gm,"var ");};
