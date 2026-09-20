// Editor-side behaviour for "Prompt Paragraphs (tools)":
//
//   * cells above the paragraph count are hidden (widget.hidden, the facade
//     over the frontend's widget visibility component)
//   * every visible cell is given a fixed height that follows its own text,
//     or a fixed line count when the cellLines node property is set
//     (right click the node -> "Cell height")
//
// Both are cosmetic. The cells are declared server side, so the node keeps
// working with this script missing, and hidden cells keep their text.

import { app } from "../../scripts/app.js";

const NODE_NAME = "ToolsPromptParagraphs";
const CELL = /^text_(\d+)$/;

const MIN_LINES = 2;       // a cell never collapses below this
const MAX_AUTO_LINES = 20; // past this a growing cell scrolls instead
const FALLBACK_LINE_HEIGHT = 20;
const WIDGET_CHROME = 4;   // the layout adds 4px to every fixed-height widget

function cellIndex(widget) {
    const match = CELL.exec(widget?.name ?? "");
    return match ? Number(match[1]) : 0;
}

function textareaOf(widget) {
    const element = widget?.element ?? widget?.inputEl ?? null;
    if (!element) {
        return null;
    }
    return element.tagName === "TEXTAREA" ? element : element.querySelector?.("textarea") ?? null;
}

function widgetValue(node, name, fallback) {
    const widget = node.widgets?.find((w) => w.name === name);
    const value = Number(widget?.value);
    return Number.isFinite(value) ? value : fallback;
}

/**
 * Lines per cell, 0 meaning "fit the text".
 *
 * This lives in node.properties rather than in a widget: widget values are
 * restored by position from a saved workflow, so adding a widget among the
 * existing ones shifts every value after it.
 */
function cellLines(node) {
    const value = Number(node.properties?.cellLines);
    return Number.isFinite(value) && value > 0 ? Math.min(Math.round(value), 40) : 0;
}

/** Height in pixels the textarea needs for `lines` lines, or for its text. */
function contentHeight(element, lines) {
    const style = getComputedStyle(element);
    const lineHeight = parseFloat(style.lineHeight) || FALLBACK_LINE_HEIGHT;
    const borders = (parseFloat(style.borderTopWidth) || 0) + (parseFloat(style.borderBottomWidth) || 0);
    const padding = (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0);

    if (lines > 0) {
        return lines * lineHeight + padding + borders;
    }

    // scrollHeight only reports the text height while the element is not
    // already stretched to it, so collapse it for the measurement
    const previous = element.style.height;
    element.style.height = "0px";
    const text = element.scrollHeight + borders;
    element.style.height = previous;

    const min = MIN_LINES * lineHeight + padding + borders;
    const max = MAX_AUTO_LINES * lineHeight + padding + borders;
    return Math.min(Math.max(text, min), max);
}

function sizeCell(node, widget, lines) {
    const element = textareaOf(widget);
    if (!element) {
        return false;
    }

    // the frontend gives the element (widget height - 2 * margin), so the
    // widget has to ask for that much more than the text needs
    const margin = widget.margin ?? 10;
    const height = Math.round(contentHeight(element, lines)) + 2 * margin - WIDGET_CHROME;
    if (widget.toolsHeight === height) {
        return false;
    }

    widget.toolsHeight = height;
    widget.computeSize = () => [0, height];
    return true;
}

function setHidden(widget, hidden) {
    if (widget.hidden === hidden) {
        return false;
    }

    widget.hidden = hidden;
    if (hidden) {
        widget.computeSize = () => [0, -WIDGET_CHROME];
        widget.toolsHeight = undefined;
    }

    // older frontends that do not act on widget.hidden
    const element = widget.element ?? widget.inputEl;
    if (element?.style) {
        element.style.display = hidden ? "none" : "";
    }
    return true;
}

function apply(node) {
    if (!node.widgets?.length) {
        return;
    }

    const count = Math.max(1, widgetValue(node, "paragraphs", 1));
    const lines = cellLines(node);

    let changed = false;
    let pending = false;
    for (const widget of node.widgets) {
        const index = cellIndex(widget);
        if (!index) {
            continue;
        }
        const hidden = index > count;
        changed = setHidden(widget, hidden) || changed;
        if (!hidden) {
            changed = sizeCell(node, widget, lines) || changed;
            pending = pending || !textareaOf(widget);
        }
    }

    if (pending && (node.toolsRetries ?? 0) < 10) {
        // the DOM widgets are not in the document yet (a freshly loaded
        // workflow); measure again on the next frame
        node.toolsRetries = (node.toolsRetries ?? 0) + 1;
        schedule(node);
    } else {
        node.toolsRetries = 0;
    }

    if (!changed) {
        return;
    }

    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.graph?.setDirtyCanvas(true, true);
}

function schedule(node) {
    if (node.toolsPending) {
        return;
    }
    node.toolsPending = true;
    requestAnimationFrame(() => {
        node.toolsPending = false;
        apply(node);
    });
}

/** Re-measure a cell while it is typed in, and when the node is re-laid out. */
function watch(node) {
    for (const widget of node.widgets ?? []) {
        if (!cellIndex(widget)) {
            continue;
        }
        const element = textareaOf(widget);
        if (!element || element.dataset.toolsWatched === "1") {
            continue;
        }
        element.dataset.toolsWatched = "1";
        element.addEventListener("input", () => schedule(node));
    }
}

function follow(node, name) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (!widget || widget.toolsFollowed) {
        return;
    }
    widget.toolsFollowed = true;

    const callback = widget.callback;
    widget.callback = (...args) => {
        const value = callback?.apply(widget, args);
        schedule(node);
        return value;
    };
}

function attach(node) {
    if (node.properties.cellLines === undefined) {
        node.properties.cellLines = 0;
    }
    follow(node, "paragraphs");
    watch(node);
    schedule(node);
}

app.registerExtension({
    name: "comfyui_tools.PromptParagraphs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const result = getExtraMenuOptions?.apply(this, arguments);
            const node = this;
            options.push({
                content: `Cell height (${cellLines(node) || "fit text"})`,
                callback: () => {
                    canvas.prompt(
                        "Lines per cell (0 = fit the text)",
                        cellLines(node),
                        (value) => {
                            const lines = Number(value);
                            node.properties.cellLines = Number.isFinite(lines)
                                ? Math.max(0, Math.min(Math.round(lines), 40))
                                : 0;
                            for (const widget of node.widgets ?? []) {
                                if (cellIndex(widget)) {
                                    widget.toolsHeight = undefined;
                                }
                            }
                            schedule(node);
                        },
                        {},
                    );
                },
            });
            return result;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            attach(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            attach(this);
            return result;
        };

        // a narrower node wraps the text differently, so re-measure on resize
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            const result = onResize?.apply(this, arguments);
            if (this.toolsWidth !== this.size[0]) {
                this.toolsWidth = this.size[0];
                schedule(this);
            }
            return result;
        };
    },
});
