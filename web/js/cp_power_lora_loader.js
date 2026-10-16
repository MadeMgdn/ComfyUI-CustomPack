import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_NAME = "ComfyUI.CustomPack.RgthreePowerLoraLoader";
const TARGET_NODE_NAME = "Power Lora Loader (rgthree)";
const CATEGORY = "Custom pack/Imported/Rgthree";
const PROPERTY_SHOW_STRENGTHS = "Show Strengths";
const PROPERTY_MATCH = "Match";
const MODE_SINGLE = "Single Strength";
const MODE_SEPARATE = "Separate Model & Clip";
const WIDGET_HEIGHT = LiteGraph.NODE_WIDGET_HEIGHT || 20;
const ROW_HEIGHT = Math.max(24, WIDGET_HEIGHT);
const MIN_NODE_WIDTH = 340;
const STRENGTH_WIDTH = 72;

let loraNamesPromise = null;

function markDirty(node) {
    if (typeof node?.setDirtyCanvas === "function") {
        node.setDirtyCanvas(true, true);
        return;
    }
    app.graph?.setDirtyCanvas?.(true, true);
}

function pointInBounds(pos, bounds) {
    return (
        pos[0] >= bounds[0] &&
        pos[0] <= bounds[0] + bounds[2] &&
        pos[1] >= bounds[1] &&
        pos[1] <= bounds[1] + bounds[3]
    );
}

function fitString(ctx, value, maxWidth) {
    const text = String(value || "None");
    if (ctx.measureText(text).width <= maxWidth) {
        return text;
    }

    const suffix = "...";
    let low = 0;
    let high = text.length;
    while (low < high) {
        const middle = Math.ceil((low + high) / 2);
        if (ctx.measureText(text.slice(0, middle) + suffix).width <= maxWidth) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return text.slice(0, low) + suffix;
}

function normalizeStrength(value, fallback = 1) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}

function normalizeRowValue(value) {
    const strength = normalizeStrength(value?.strength, 1);
    return {
        on: value?.on !== false,
        lora: typeof value?.lora === "string" ? value.lora : "",
        strength,
        strengthTwo: value?.strengthTwo == null
            ? null
            : normalizeStrength(value.strengthTwo, strength),
    };
}

function isSeparateMode(node) {
    return node.properties?.[PROPERTY_SHOW_STRENGTHS] === MODE_SEPARATE;
}

function getLoraWidgets(node) {
    return (node.widgets || []).filter((widget) => widget.__cpPowerLoraRow);
}

function ensureProperties(node) {
    node.properties = node.properties || {};
    if (node.properties[PROPERTY_SHOW_STRENGTHS] === undefined) {
        node.properties[PROPERTY_SHOW_STRENGTHS] = MODE_SINGLE;
    }
    if (node.properties[PROPERTY_MATCH] === undefined) {
        node.properties[PROPERTY_MATCH] = "";
    }

    try {
        node.addProperty?.(
            PROPERTY_SHOW_STRENGTHS,
            node.properties[PROPERTY_SHOW_STRENGTHS],
            "combo",
            { values: [MODE_SINGLE, MODE_SEPARATE] }
        );
        node.addProperty?.(PROPERTY_MATCH, node.properties[PROPERTY_MATCH], "string");
    } catch (_error) {
        // Some ComfyUI builds reject a property that already exists.
    }
}

async function requestLoraNames(force = false) {
    if (force) {
        loraNamesPromise = null;
    }
    if (loraNamesPromise) {
        return loraNamesPromise;
    }

    loraNamesPromise = (async () => {
        try {
            const response = await api.fetchApi("/cp_rgthree_power_lora_loader/loras", {
                cache: "no-store",
            });
            if (response.ok) {
                const names = await response.json();
                if (Array.isArray(names)) {
                    return names;
                }
            }
        } catch (error) {
            console.warn("[CustomPack] Power Lora Loader route is unavailable", error);
        }

        try {
            const response = await api.fetchApi("/object_info/LoraLoader", { cache: "no-store" });
            if (!response.ok) {
                return [];
            }
            const data = await response.json();
            const definition = data?.LoraLoader || data;
            const names = definition?.input?.required?.lora_name?.[0];
            return Array.isArray(names) ? names : [];
        } catch (error) {
            console.warn("[CustomPack] Failed to load LoRA names", error);
            return [];
        }
    })();

    return loraNamesPromise;
}

function getFilteredLoraNames(node, names) {
    const pattern = String(node.properties?.[PROPERTY_MATCH] || "").trim();
    if (!pattern) {
        return names;
    }

    try {
        const expression = new RegExp(pattern);
        return names.filter((name) => expression.test(name));
    } catch (error) {
        console.warn(`[CustomPack] Invalid Power Lora Loader Match expression: ${pattern}`, error);
        return names;
    }
}

async function showLoraChooser(node, event, onChoose) {
    const names = getFilteredLoraNames(node, await requestLoraNames());
    const options = names.length
        ? names.map((name) => ({
            content: name,
            callback: () => onChoose(name),
        }))
        : [{ content: "No matching LoRAs", disabled: true }];

    new LiteGraph.ContextMenu(options, {
        event,
        node,
        title: "Select LoRA",
    });
}

function updateNodeSize(node) {
    const computed = node.computeSize?.() || node.size || [MIN_NODE_WIDTH, 100];
    const width = Math.max(node.size?.[0] || 0, computed[0] || 0, MIN_NODE_WIDTH);
    const height = Math.max(computed[1] || 0, 80);
    node.setSize?.([width, height]);
    markDirty(node);
}

function drawWidgetBackground(ctx, x, y, width, height, enabled = true) {
    ctx.save();
    ctx.globalAlpha = enabled ? 0.95 : 0.45;
    ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#222";
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#555";
    ctx.beginPath();
    ctx.rect(x, y, width, height);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
}

function drawStrength(ctx, bounds, value, enabled) {
    const [x, y, width, height] = bounds;
    const sideWidth = 16;
    drawWidgetBackground(ctx, x, y, width, height, enabled);

    ctx.save();
    ctx.globalAlpha = enabled ? 1 : 0.45;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#DDD";
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillText("-", x + sideWidth / 2, y + height / 2);
    ctx.fillText(normalizeStrength(value).toFixed(2), x + width / 2, y + height / 2);
    ctx.fillText("+", x + width - sideWidth / 2, y + height / 2);
    ctx.restore();

    return {
        decrement: [x, y, sideWidth, height],
        value: [x + sideWidth, y, width - sideWidth * 2, height],
        increment: [x + width - sideWidth, y, sideWidth, height],
    };
}

function promptForStrength(node, widget, property, event) {
    const current = normalizeStrength(widget.value[property], 1);
    const applyValue = (input) => {
        if (input === null || input === undefined || input === "") {
            return;
        }
        const value = Number(input);
        if (Number.isFinite(value)) {
            widget.value[property] = value;
            markDirty(node);
        }
    };

    if (typeof app.canvas?.prompt === "function") {
        app.canvas.prompt("LoRA strength", String(current), applyValue, event);
    } else {
        applyValue(window.prompt("LoRA strength", String(current)));
    }
}

function changeStrength(node, widget, property, amount) {
    const value = normalizeStrength(widget.value[property], widget.value.strength);
    widget.value[property] = Math.round((value + amount) * 100) / 100;
    markDirty(node);
}

function moveLoraWidget(node, widget, direction) {
    const rows = getLoraWidgets(node);
    const rowIndex = rows.indexOf(widget);
    const other = rows[rowIndex + direction];
    if (!other) {
        return;
    }

    const widgetIndex = node.widgets.indexOf(widget);
    const otherIndex = node.widgets.indexOf(other);
    node.widgets[widgetIndex] = other;
    node.widgets[otherIndex] = widget;
    markDirty(node);
}

function removeLoraWidget(node, widget) {
    const index = node.widgets.indexOf(widget);
    if (index >= 0) {
        node.widgets.splice(index, 1);
        widget.onRemove?.();
        updateNodeSize(node);
    }
}

function showRowMenu(node, widget, event) {
    const rows = getLoraWidgets(node);
    const index = rows.indexOf(widget);
    new LiteGraph.ContextMenu([
        {
            content: widget.value.on ? "Disable" : "Enable",
            callback: () => {
                widget.value.on = !widget.value.on;
                markDirty(node);
            },
        },
        null,
        {
            content: "Move Up",
            disabled: index <= 0,
            callback: () => moveLoraWidget(node, widget, -1),
        },
        {
            content: "Move Down",
            disabled: index < 0 || index >= rows.length - 1,
            callback: () => moveLoraWidget(node, widget, 1),
        },
        {
            content: "Remove",
            callback: () => removeLoraWidget(node, widget),
        },
    ], {
        event,
        node,
        title: widget.value.lora || "LoRA",
    });
}

function createHeaderWidget() {
    return {
        name: "power_lora_header",
        type: "custom",
        value: null,
        serialize: false,
        options: { serialize: false },
        __cpPowerLoraWidget: true,
        draw(ctx, node, width, y, height) {
            const rows = getLoraWidgets(node);
            const separate = isSeparateMode(node);
            const margin = 8;
            const gap = 4;
            const right = width - margin;
            const firstStrengthX = separate
                ? right - STRENGTH_WIDTH * 2 - gap
                : right - STRENGTH_WIDTH;

            this.__toggleBounds = [margin, y + 2, 18, height - 4];
            ctx.save();
            ctx.globalAlpha = rows.length ? 0.65 : 0.3;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#DDD";
            ctx.textBaseline = "middle";
            ctx.textAlign = "left";
            ctx.fillText("Toggle All", margin + 24, y + height / 2);
            ctx.textAlign = "center";
            if (separate) {
                ctx.fillText("Model", firstStrengthX + STRENGTH_WIDTH / 2, y + height / 2);
                ctx.fillText("CLIP", right - STRENGTH_WIDTH / 2, y + height / 2);
            } else {
                ctx.fillText("Strength", right - STRENGTH_WIDTH / 2, y + height / 2);
            }

            const allOn = rows.length > 0 && rows.every((row) => row.value.on);
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#777";
            ctx.fillStyle = allOn ? "#7FA3C8" : (LiteGraph.WIDGET_BGCOLOR || "#222");
            ctx.beginPath();
            ctx.arc(margin + 9, y + height / 2, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        },
        mouse(event, pos, node) {
            if (event.type !== "pointerdown" || !pointInBounds(pos, this.__toggleBounds || [])) {
                return true;
            }
            const rows = getLoraWidgets(node);
            const nextState = !rows.length || !rows.every((row) => row.value.on);
            rows.forEach((row) => {
                row.value.on = nextState;
            });
            markDirty(node);
            return true;
        },
        computeSize(width) {
            return [width, WIDGET_HEIGHT];
        },
        serializeValue() {
            return undefined;
        },
    };
}

function createLoraWidget(node, name, initialValue) {
    const widget = {
        name,
        type: "custom",
        value: normalizeRowValue(initialValue),
        __cpPowerLoraWidget: true,
        __cpPowerLoraRow: true,
        draw(ctx, currentNode, width, y, height) {
            const margin = 8;
            const gap = 4;
            const separate = isSeparateMode(currentNode);
            const right = width - margin;
            const firstStrengthX = separate
                ? right - STRENGTH_WIDTH * 2 - gap
                : right - STRENGTH_WIDTH;
            const toggleX = margin;
            const menuX = toggleX + 20;
            const nameX = menuX + 20;
            const nameWidth = Math.max(20, firstStrengthX - gap - nameX);
            const boxY = y + 2;
            const boxHeight = height - 4;

            drawWidgetBackground(ctx, margin, boxY, width - margin * 2, boxHeight, this.value.on);

            this.__toggleBounds = [toggleX, boxY, 18, boxHeight];
            this.__menuBounds = [menuX, boxY, 18, boxHeight];
            this.__loraBounds = [nameX, boxY, nameWidth, boxHeight];

            ctx.save();
            ctx.globalAlpha = this.value.on ? 1 : 0.42;
            ctx.fillStyle = this.value.on ? "#7FA3C8" : (LiteGraph.WIDGET_BGCOLOR || "#222");
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#777";
            ctx.beginPath();
            ctx.arc(toggleX + 9, y + height / 2, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#DDD";
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            ctx.fillText("...", menuX + 9, y + height / 2 - 2);
            ctx.textAlign = "left";
            ctx.fillText(
                fitString(ctx, this.value.lora || "Select LoRA", nameWidth - 4),
                nameX + 2,
                y + height / 2
            );
            ctx.restore();

            this.__strengthAreas = {
                strength: drawStrength(
                    ctx,
                    [firstStrengthX, boxY, STRENGTH_WIDTH, boxHeight],
                    this.value.strength,
                    this.value.on
                ),
            };

            if (separate) {
                if (this.value.strengthTwo == null) {
                    this.value.strengthTwo = this.value.strength;
                }
                this.__strengthAreas.strengthTwo = drawStrength(
                    ctx,
                    [right - STRENGTH_WIDTH, boxY, STRENGTH_WIDTH, boxHeight],
                    this.value.strengthTwo,
                    this.value.on
                );
            }
        },
        mouse(event, pos, currentNode) {
            if (event.type === "pointerup") {
                const pendingAction = this.__pendingClickAction;
                this.__pendingClickAction = null;

                if (pendingAction === "menu" && pointInBounds(pos, this.__menuBounds || [])) {
                    showRowMenu(currentNode, this, event);
                } else if (
                    pendingAction === "lora"
                    && pointInBounds(pos, this.__loraBounds || [])
                ) {
                    showLoraChooser(currentNode, event, (lora) => {
                        this.value.lora = lora;
                        markDirty(currentNode);
                    });
                }
                return true;
            }

            if (event.type !== "pointerdown") {
                if (event.type === "pointercancel") {
                    this.__pendingClickAction = null;
                }
                return true;
            }

            this.__pendingClickAction = null;
            if (pointInBounds(pos, this.__toggleBounds || [])) {
                this.value.on = !this.value.on;
                markDirty(currentNode);
                return true;
            }
            if (pointInBounds(pos, this.__menuBounds || [])) {
                this.__pendingClickAction = "menu";
                return true;
            }
            if (pointInBounds(pos, this.__loraBounds || [])) {
                this.__pendingClickAction = "lora";
                return true;
            }

            for (const [property, areas] of Object.entries(this.__strengthAreas || {})) {
                if (pointInBounds(pos, areas.decrement)) {
                    changeStrength(currentNode, this, property, -0.05);
                    return true;
                }
                if (pointInBounds(pos, areas.increment)) {
                    changeStrength(currentNode, this, property, 0.05);
                    return true;
                }
                if (pointInBounds(pos, areas.value)) {
                    promptForStrength(currentNode, this, property, event);
                    return true;
                }
            }
            return true;
        },
        computeSize(width) {
            return [width, ROW_HEIGHT];
        },
        serializeValue(currentNode) {
            const result = {
                on: !!this.value.on,
                lora: String(this.value.lora || ""),
                strength: normalizeStrength(this.value.strength, 1),
            };
            if (isSeparateMode(currentNode)) {
                result.strengthTwo = normalizeStrength(
                    this.value.strengthTwo,
                    result.strength
                );
            }
            return result;
        },
    };

    if (isSeparateMode(node) && widget.value.strengthTwo == null) {
        widget.value.strengthTwo = widget.value.strength;
    }
    return widget;
}

function addLoraWidget(node, value) {
    node.__cpPowerLoraCounter = (node.__cpPowerLoraCounter || 0) + 1;
    const widget = createLoraWidget(node, `lora_${node.__cpPowerLoraCounter}`, value);
    const addButtonIndex = (node.widgets || []).findIndex((item) => item.__cpPowerLoraAdd);
    node.addCustomWidget(widget);

    if (addButtonIndex >= 0) {
        const currentIndex = node.widgets.indexOf(widget);
        node.widgets.splice(addButtonIndex, 0, node.widgets.splice(currentIndex, 1)[0]);
    }
    updateNodeSize(node);
    return widget;
}

function createAddWidget() {
    return {
        name: "power_lora_add",
        type: "custom",
        value: null,
        serialize: false,
        options: { serialize: false },
        __cpPowerLoraWidget: true,
        __cpPowerLoraAdd: true,
        draw(ctx, _node, width, y, height) {
            const margin = 8;
            this.__bounds = [margin, y + 2, width - margin * 2, height - 4];
            drawWidgetBackground(ctx, ...this.__bounds, true);
            ctx.save();
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#DDD";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("+ Add LoRA", width / 2, y + height / 2);
            ctx.restore();
        },
        mouse(event, pos, node) {
            if (event.type === "pointerup") {
                const shouldOpen = this.__pendingClick && pointInBounds(pos, this.__bounds || []);
                this.__pendingClick = false;
                if (shouldOpen) {
                    showLoraChooser(node, event, (lora) => addLoraWidget(node, {
                        on: true,
                        lora,
                        strength: 1,
                        strengthTwo: isSeparateMode(node) ? 1 : null,
                    }));
                }
                return true;
            }

            if (event.type === "pointerdown") {
                this.__pendingClick = pointInBounds(pos, this.__bounds || []);
            } else if (event.type === "pointercancel") {
                this.__pendingClick = false;
            }
            return true;
        },
        computeSize(width) {
            return [width, WIDGET_HEIGHT];
        },
        serializeValue() {
            return undefined;
        },
    };
}

function clearPowerLoraWidgets(node) {
    for (let index = (node.widgets || []).length - 1; index >= 0; index--) {
        const widget = node.widgets[index];
        if (widget.__cpPowerLoraWidget) {
            node.widgets.splice(index, 1);
            widget.onRemove?.();
        }
    }
}

function setupNode(node, values = null) {
    ensureProperties(node);
    const currentValues = values || getLoraWidgets(node).map((widget) => widget.value);

    clearPowerLoraWidgets(node);
    node.__cpPowerLoraCounter = 0;
    node.serialize_widgets = true;
    node.category = CATEGORY;
    node.addCustomWidget(createHeaderWidget());
    for (const value of currentValues) {
        addLoraWidget(node, value);
    }
    node.addCustomWidget(createAddWidget());
    node.__cpPowerLoraSetup = true;
    updateNodeSize(node);
}

function syncStrengthMode(node) {
    const separate = isSeparateMode(node);
    for (const widget of getLoraWidgets(node)) {
        if (separate && widget.value.strengthTwo == null) {
            widget.value.strengthTwo = widget.value.strength;
        } else if (!separate) {
            widget.value.strengthTwo = null;
        }
    }
    markDirty(node);
}

app.registerExtension({
    name: EXTENSION_NAME,

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== TARGET_NODE_NAME) {
            return;
        }

        nodeType.category = CATEGORY;
        nodeType[`@${PROPERTY_SHOW_STRENGTHS}`] = {
            type: "combo",
            values: [MODE_SINGLE, MODE_SEPARATE],
        };
        nodeType[`@${PROPERTY_MATCH}`] = { type: "string" };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            setupNode(this, []);
            return result;
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info, ...rest) {
            const values = Array.isArray(info?.widgets_values)
                ? info.widgets_values.filter((value) => (
                    value && typeof value === "object" && typeof value.lora === "string"
                ))
                : [];
            const result = originalConfigure?.apply(this, [info, ...rest]);
            setupNode(this, values);
            return result;
        };

        const originalOnPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function (name, value, previousValue) {
            const result = originalOnPropertyChanged?.apply(this, [name, value, previousValue]);
            if (name === PROPERTY_SHOW_STRENGTHS && value !== previousValue) {
                syncStrengthMode(this);
            }
            return result;
        };

        const originalOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (serialized, ...rest) {
            const result = originalOnSerialize?.apply(this, [serialized, ...rest]);
            serialized.widgets_values = getLoraWidgets(this).map((widget) => (
                widget.serializeValue(this)
            ));
            return result;
        };

        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const result = originalGetExtraMenuOptions?.apply(this, [canvas, options]) || options || [];
            result.push(null, {
                content: "Refresh LoRA list",
                callback: () => requestLoraNames(true),
            });
            return result;
        };
    },

    loadedGraphNode(node) {
        if (node?.type === TARGET_NODE_NAME && !node.__cpPowerLoraSetup) {
            setupNode(node);
        }
    },
});
