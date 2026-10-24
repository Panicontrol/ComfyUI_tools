// Editor-side display for nodes that report text: after a run the node draws
// its report below its widgets. Drawn on the canvas rather than in a widget,
// so nothing is added to the saved workflow. The same text is each node's
// STRING output, so it can also go to Preview Any when this script is missing.

import { app } from "../../scripts/app.js";

const NODES = new Set(["ToolsLatentInfo", "ToolsShotFinalize"]);
const LINE = 16;
const PAD = 10;
const TOP = 56; // below the title and one socket row, when there are no widgets

function textTop(node) {
  let top = TOP;
  for (const w of node.widgets ?? []) {
    if (w.last_y === undefined) continue;
    const h = w.computeSize ? w.computeSize(node.size[0])[1] : 20;
    top = Math.max(top, w.last_y + h + LINE);
  }
  return top;
}

app.registerExtension({
  name: "comfyui_tools.textReport",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.has(nodeData.name)) return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      this.reportLines = (message?.text ?? []).join("\n").split("\n");
      const width = Math.max(...this.reportLines.map((l) => l.length)) * 6.6 + 2 * PAD;
      const height = textTop(this) + this.reportLines.length * LINE + PAD;
      this.setSize([Math.max(this.size[0], width), Math.max(this.size[1], height)]);
      app.graph.setDirtyCanvas(true, true);
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      onDrawForeground?.apply(this, arguments);
      if (this.flags.collapsed || !this.reportLines) return;
      const top = textTop(this);
      ctx.save();
      ctx.font = "12px monospace";
      ctx.textAlign = "left";
      this.reportLines.forEach((line, i) => {
        const bad = line.includes("!") || line.includes("NaN") || line.includes("empty") || line.includes("not found");
        ctx.fillStyle = bad ? "#f88" : "#ccc";
        ctx.fillText(line, PAD, top + i * LINE);
      });
      ctx.restore();
    };
  },
});
