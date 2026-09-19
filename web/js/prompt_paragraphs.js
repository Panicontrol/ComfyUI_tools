// Show only the cells the "Prompt Paragraphs (tools)" node actually uses.
//
// Every cell (text_1 .. text_N) is declared on the Python side, so the node
// works with this script missing or broken -- it just shows all of its cells.
// Hiding is cosmetic: the values stay in the widgets, which is why raising the
// paragraph count again brings the text back.

import { app } from "../../scripts/app.js";

const NODE_NAME = "ToolsPromptParagraphs";
const HIDDEN_TYPE = "converted-widget";

function isCell(widget) {
    return /^text_\d+$/.test(widget?.name ?? "");
}

function cellIndex(widget) {
    return parseInt(widget.name.slice("text_".length), 10);
}

function setVisible(widget, visible) {
    if (widget.origType === undefined) {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
    }

    widget.type = visible ? widget.origType : HIDDEN_TYPE;
    widget.computeSize = visible ? widget.origComputeSize : () => [0, -4];
    // newer frontends drive DOM widgets through this flag
    widget.hidden = !visible;

    const element = widget.element ?? widget.inputEl;
    if (element?.style) {
        element.style.display = visible ? "" : "none";
    }
}

function applyCount(node) {
    const counter = node.widgets?.find((w) => w.name === "paragraphs");
    if (!counter) {
        return;
    }

    const count = Math.max(1, Number(counter.value) || 1);
    let changed = false;
    for (const widget of node.widgets) {
        if (!isCell(widget)) {
            continue;
        }
        const visible = cellIndex(widget) <= count;
        if (widget.hidden === !visible) {
            continue;  // already in the right state
        }
        setVisible(widget, visible);
        changed = true;
    }

    if (changed) {
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), size[1]]);
        node.graph?.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "comfyui_tools.PromptParagraphs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const counter = this.widgets?.find((w) => w.name === "paragraphs");
            if (counter) {
                const callback = counter.callback;
                counter.callback = (...args) => {
                    const value = callback?.apply(counter, args);
                    applyCount(this);
                    return value;
                };
            }
            requestAnimationFrame(() => applyCount(this));
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => applyCount(this));
            return result;
        };
    },
});
