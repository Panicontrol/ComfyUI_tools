// Editor-side display for "Latent Info (tools)": after a run the node draws
// what it found under its inputs. Drawn on the canvas rather than in a widget,
// so nothing is added to the saved workflow. The same text is the node's info
// output, so it can also go to Preview Any when this script is missing.

import { app } from "../../scripts/app.js";

const NODE_NAME = "ToolsLatentInfo";
const LINE = 16;
const PAD = 10;
const TOP = 56; // below the title and the latent socket rows

app.registerExtension({
  name: "comfyui_tools.latentInfo",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      this.latentInfo = (message?.text ?? []).join("\n").split("\n");
      const width = Math.max(...this.latentInfo.map((l) => l.length)) * 6.6 + 2 * PAD;
      this.setSize([Math.max(this.size[0], width), Math.max(this.size[1], TOP + this.latentInfo.length * LINE + PAD)]);
      app.graph.setDirtyCanvas(true, true);
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      onDrawForeground?.apply(this, arguments);
      if (this.flags.collapsed || !this.latentInfo) return;
      ctx.save();
      ctx.font = "12px monospace";
      ctx.textAlign = "left";
      this.latentInfo.forEach((line, i) => {
        ctx.fillStyle = line.includes("!") || line.includes("NaN") || line.includes("empty") ? "#f88" : "#ccc";
        ctx.fillText(line, PAD, TOP + i * LINE);
      });
      ctx.restore();
    };
  },
});
