import { app } from "/scripts/app.js";

const EXTENSION_NAME = "ComfyUI.CustomPack.RgthreeFastGroups";
const TARGET_NODE_CONFIG = {
    "Fast Groups Bypasser (rgthree)": {
        modeOff: 4,
        actionOff: "Bypass all",
    },
    "Fast Groups Muter (rgthree)": {
        modeOff: LiteGraph.NEVER,
        actionOff: "Mute all",
    },
};
const TARGET_NODE_NAMES = new Set(Object.keys(TARGET_NODE_CONFIG));
const CATEGORY = "Custom pack/Imported/Rgthree";
const MODE_ON = LiteGraph.ALWAYS;
const WIDGET_HEIGHT = LiteGraph.NODE_WIDGET_HEIGHT || 20;

const PROPERTY_SORT = "sort";
const PROPERTY_SORT_CUSTOM_ALPHA = "customSortAlphabet";
const PROPERTY_MATCH_COLORS = "matchColors";
const PROPERTY_MATCH_TITLE = "matchTitle";
const PROPERTY_SHOW_NAV = "showNav";
const PROPERTY_SHOW_ALL_GRAPHS = "showAllGraphs";
const PROPERTY_RESTRICTION = "toggleRestriction";

const fastGroupNodes = new Set();
let refreshTimer = null;

function markDirty(node, foreground = true, background = false) {
    if (typeof node?.setDirtyCanvas === "function") {
        node.setDirtyCanvas(foreground, background);
        return;
    }
    node?.graph?.setDirtyCanvas?.(foreground, background);
    app.graph?.setDirtyCanvas?.(foreground, background);
}

function ensureProperties(node) {
    node.properties = node.properties || {};

    const defaults = {
        [PROPERTY_MATCH_COLORS]: "",
        [PROPERTY_MATCH_TITLE]: "",
        [PROPERTY_SHOW_NAV]: true,
        [PROPERTY_SHOW_ALL_GRAPHS]: true,
        [PROPERTY_SORT]: "position",
        [PROPERTY_SORT_CUSTOM_ALPHA]: "",
        [PROPERTY_RESTRICTION]: "default",
    };

    for (const [key, value] of Object.entries(defaults)) {
        if (node.properties[key] === undefined) {
            node.properties[key] = value;
        }
    }

    try {
        node.addProperty?.(PROPERTY_MATCH_COLORS, node.properties[PROPERTY_MATCH_COLORS], "string");
        node.addProperty?.(PROPERTY_MATCH_TITLE, node.properties[PROPERTY_MATCH_TITLE], "string");
        node.addProperty?.(PROPERTY_SHOW_NAV, node.properties[PROPERTY_SHOW_NAV], "boolean");
        node.addProperty?.(PROPERTY_SHOW_ALL_GRAPHS, node.properties[PROPERTY_SHOW_ALL_GRAPHS], "boolean");
        node.addProperty?.(PROPERTY_SORT, node.properties[PROPERTY_SORT], "combo", {
            values: ["position", "alphanumeric", "custom alphabet"],
        });
        node.addProperty?.(PROPERTY_SORT_CUSTOM_ALPHA, node.properties[PROPERTY_SORT_CUSTOM_ALPHA], "string");
        node.addProperty?.(PROPERTY_RESTRICTION, node.properties[PROPERTY_RESTRICTION], "combo", {
            values: ["default", "max one", "always one"],
        });
    } catch (_error) {
        // Some ComfyUI builds throw on duplicate addProperty calls.
    }
}

function normalizeHexColor(color) {
    if (!color) {
        return "";
    }

    let normalized = String(color).trim().toLowerCase();
    if (LGraphCanvas.node_colors?.[normalized]) {
        normalized = LGraphCanvas.node_colors[normalized].groupcolor || normalized;
    }

    normalized = normalized.replace("#", "");
    if (normalized.length === 3) {
        normalized = normalized.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
    }

    return normalized ? `#${normalized}` : "";
}

function getGroupBounds(group) {
    if (Array.isArray(group?._bounding)) {
        return group._bounding;
    }

    const pos = group?._pos || group?.pos || [0, 0];
    const size = group?._size || group?.size || [0, 0];
    return [pos[0], pos[1], size[0], size[1]];
}

function getNodeBounds(node) {
    if (typeof node?.getBounding === "function") {
        const bounds = node.getBounding();
        if (bounds && (bounds[2] || bounds[3])) {
            return bounds;
        }
    }

    const pos = node?.pos || [0, 0];
    const size = node?.size || [0, 0];
    return [pos[0], pos[1], size[0], size[1]];
}

function getGroupGraph(group) {
    return group?.graph || app.canvas?.getCurrentGraph?.() || app.graph;
}

function recomputeGroupNodes(group) {
    try {
        group.recomputeInsideNodes?.();
    } catch (_error) {
        // Fall through to the local computation.
    }

    const graph = getGroupGraph(group);
    const nodes = graph?.nodes || graph?._nodes || app.graph?._nodes || [];
    const groupBounds = getGroupBounds(group);
    const inside = [];

    for (const node of nodes) {
        if (!node || TARGET_NODE_NAMES.has(node.type)) {
            continue;
        }

        const bounds = getNodeBounds(node);
        const centerX = bounds[0] + bounds[2] * 0.5;
        const centerY = bounds[1] + bounds[3] * 0.5;

        if (
            centerX >= groupBounds[0] &&
            centerX < groupBounds[0] + groupBounds[2] &&
            centerY >= groupBounds[1] &&
            centerY < groupBounds[1] + groupBounds[3]
        ) {
            inside.push(node);
        }
    }

    group.__cpFastGroupsNodes = inside;
    if (group._children instanceof Set) {
        group._children.clear();
        for (const node of inside) {
            group._children.add(node);
        }
    }
    if (Array.isArray(group.nodes)) {
        group.nodes.length = 0;
        group.nodes.push(...inside);
    }
    group.rgthree_hasAnyActiveNode = inside.some((node) => node.mode === MODE_ON);
    return inside;
}

function getGroupNodes(group) {
    if (Array.isArray(group.__cpFastGroupsNodes)) {
        return group.__cpFastGroupsNodes;
    }
    return recomputeGroupNodes(group);
}

function getAllGroups(showAllGraphs = true) {
    const canvas = app.canvas;
    const currentGraph = canvas?.getCurrentGraph?.() || app.graph;
    const groups = [];

    if (showAllGraphs) {
        groups.push(...(app.graph?._groups || app.graph?.groups || []));
        const subgraphs = app.graph?.subgraphs?.values?.();
        if (subgraphs) {
            let graph;
            while ((graph = subgraphs.next().value)) {
                groups.push(...(graph.groups || graph._groups || []));
            }
        }
    } else {
        groups.push(...(currentGraph?._groups || currentGraph?.groups || []));
    }

    return groups.filter(Boolean);
}

function sortGroups(groups, node) {
    const sort = node.properties?.[PROPERTY_SORT] || "position";
    const sorted = [...groups];

    if (sort === "alphanumeric") {
        sorted.sort((a, b) => String(a.title || "").localeCompare(String(b.title || "")));
        return sorted;
    }

    if (sort === "custom alphabet") {
        const rawAlphabet = String(node.properties?.[PROPERTY_SORT_CUSTOM_ALPHA] || "").replace(/\n/g, "").trim();
        const alphabet = rawAlphabet
            ? (rawAlphabet.includes(",") ? rawAlphabet.split(",") : rawAlphabet.split(""))
                .map((entry) => entry.trim().toLowerCase())
                .filter(Boolean)
            : [];

        if (alphabet.length) {
            sorted.sort((a, b) => {
                const aTitle = String(a.title || "").toLowerCase();
                const bTitle = String(b.title || "").toLowerCase();
                const aIndex = alphabet.findIndex((entry) => aTitle.startsWith(entry));
                const bIndex = alphabet.findIndex((entry) => bTitle.startsWith(entry));

                if (aIndex > -1 && bIndex > -1) {
                    return aIndex === bIndex
                        ? aTitle.localeCompare(bTitle)
                        : aIndex - bIndex;
                }
                if (aIndex > -1) {
                    return -1;
                }
                if (bIndex > -1) {
                    return 1;
                }
                return aTitle.localeCompare(bTitle);
            });
            return sorted;
        }
    }

    sorted.sort((a, b) => {
        const aPos = a._pos || a.pos || [0, 0];
        const bPos = b._pos || b.pos || [0, 0];
        const aRow = Math.floor(aPos[1] / 30);
        const bRow = Math.floor(bPos[1] / 30);
        return aRow === bRow
            ? Math.floor(aPos[0] / 30) - Math.floor(bPos[0] / 30)
            : aRow - bRow;
    });
    return sorted;
}

function shouldShowGroup(group, node) {
    const colorFilters = String(node.properties?.[PROPERTY_MATCH_COLORS] || "")
        .split(",")
        .map(normalizeHexColor)
        .filter(Boolean);

    if (colorFilters.length) {
        const groupColor = normalizeHexColor(group.color);
        if (!groupColor || !colorFilters.includes(groupColor)) {
            return false;
        }
    }

    const titleFilter = String(node.properties?.[PROPERTY_MATCH_TITLE] || "").trim();
    if (titleFilter) {
        try {
            if (!new RegExp(titleFilter, "i").test(String(group.title || ""))) {
                return false;
            }
        } catch (error) {
            console.warn("[CustomPack] Invalid Fast Groups title filter", error);
            return false;
        }
    }

    return true;
}

function getVisibleGroups(node) {
    const groups = getAllGroups(node.properties?.[PROPERTY_SHOW_ALL_GRAPHS] !== false);
    for (const group of groups) {
        recomputeGroupNodes(group);
    }
    return sortGroups(groups.filter((group) => shouldShowGroup(group, node)), node);
}

function changeModeOfNodes(nodes, mode) {
    const stack = [...nodes];
    while (stack.length) {
        const node = stack.pop();
        if (!node) {
            continue;
        }

        node.mode = mode;
        if (node.isSubgraphNode?.() && node.subgraph?.nodes) {
            stack.push(...node.subgraph.nodes);
        }
    }
}

function drawWidgetBackground(ctx, width, y, height) {
    const margin = 15;
    const lowQuality = (app.canvas?.ds?.scale || 1) <= 0.5;
    const radius = lowQuality ? 0 : height * 0.5;

    ctx.save();
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
        ctx.roundRect(margin, y, width - margin * 2, height, radius);
    } else {
        ctx.rect(margin, y, width - margin * 2, height);
    }
    ctx.fill();
    if (!lowQuality) {
        ctx.stroke();
    }
    ctx.restore();

    return { margin, lowQuality };
}

function fitString(ctx, text, maxWidth) {
    const value = String(text || "");
    if (ctx.measureText(value).width <= maxWidth) {
        return value;
    }

    const ellipsis = "...";
    let current = value;
    while (current.length > 0 && ctx.measureText(current + ellipsis).width > maxWidth) {
        current = current.slice(0, -1);
    }
    return current + ellipsis;
}

class FastGroupToggleWidget {
    constructor(group, node) {
        this.type = "custom";
        this.name = "CP_FAST_GROUP_TOGGLE";
        this.group = group;
        this.node = node;
        this.label = "";
        this.value = { toggled: false };
        this.options = { on: "yes", off: "no" };
    }

    get toggled() {
        return !!this.value.toggled;
    }

    set toggled(value) {
        this.value.toggled = !!value;
    }

    doModeChange(force = null, skipOtherNodeCheck = false) {
        const nodes = recomputeGroupNodes(this.group);
        const hasAnyActiveNodes = nodes.some((node) => node.mode === MODE_ON);
        let nextValue = force !== null && force !== undefined ? !!force : !hasAnyActiveNodes;
        const restriction = this.node.properties?.[PROPERTY_RESTRICTION] || "default";

        if (!skipOtherNodeCheck) {
            if (nextValue && restriction.includes(" one")) {
                for (const widget of this.node.widgets || []) {
                    if (widget !== this && typeof widget.doModeChange === "function") {
                        widget.doModeChange(false, true);
                    }
                }
            } else if (!nextValue && restriction === "always one") {
                nextValue = (this.node.widgets || []).every(
                    (widget) => widget === this || !widget.toggled
                );
            }
        }

        const modeOff = this.node.__cpFastGroupsModeOff
            ?? TARGET_NODE_CONFIG[this.node.type]?.modeOff
            ?? 4;
        changeModeOfNodes(nodes, nextValue ? MODE_ON : modeOff);
        this.group.rgthree_hasAnyActiveNode = nextValue;
        this.toggled = nextValue;
        this.group.graph?.setDirtyCanvas?.(true, false);
        markDirty(this.node, true, false);
    }

    toggle(value = null) {
        const nextValue = value === null || value === undefined ? !this.toggled : !!value;
        if (nextValue !== this.toggled) {
            this.doModeChange(nextValue);
        }
    }

    draw(ctx, node, width, y, height) {
        const { margin, lowQuality } = drawWidgetBackground(ctx, width, y, height);
        const showNav = node.properties?.[PROPERTY_SHOW_NAV] !== false;
        let currentX = width - margin;

        ctx.save();

        if (showNav) {
            currentX -= 12;
            const midY = y + height * 0.5;
            ctx.fillStyle = ctx.strokeStyle = "#89A";
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(currentX, midY);
            ctx.lineTo(currentX - 7, midY + 6);
            ctx.lineTo(currentX - 7, midY + 3);
            ctx.lineTo(currentX - 14, midY + 3);
            ctx.lineTo(currentX - 14, midY - 3);
            ctx.lineTo(currentX - 7, midY - 3);
            ctx.lineTo(currentX - 7, midY - 6);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            currentX -= 26;
        }

        const toggleRadius = height * 0.36;
        currentX -= 8;
        ctx.fillStyle = this.toggled ? "#89A" : "#333";
        ctx.beginPath();
        ctx.arc(currentX - toggleRadius, y + height * 0.5, toggleRadius, 0, Math.PI * 2);
        ctx.fill();
        currentX -= toggleRadius * 2 + 5;

        if (!lowQuality) {
            ctx.textBaseline = "alphabetic";
            ctx.textAlign = "right";
            ctx.fillStyle = this.toggled ? LiteGraph.WIDGET_TEXT_COLOR : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
            const stateLabel = this.toggled ? this.options.on : this.options.off;
            ctx.fillText(stateLabel, currentX, y + height * 0.7);
            currentX -= Math.max(
                ctx.measureText(this.options.on).width,
                ctx.measureText(this.options.off).width
            ) + 8;

            ctx.textAlign = "left";
            const labelWidth = Math.max(10, currentX - margin - 10);
            ctx.fillText(fitString(ctx, this.label, labelWidth), margin + 10, y + height * 0.7);
        }

        ctx.restore();
    }

    mouse(event, pos, node) {
        if (event.type !== "pointerdown") {
            return true;
        }

        const showNav = node.properties?.[PROPERTY_SHOW_NAV] !== false;
        if (showNav && pos[0] >= node.size[0] - 44) {
            const canvas = app.canvas;
            const lowQuality = (canvas?.ds?.scale || 1) <= 0.5;
            if (!lowQuality) {
                canvas.centerOnNode?.(this.group);
                const groupSize = this.group._size || this.group.size || [1, 1];
                const zoomCurrent = canvas.ds?.scale || 1;
                const zoomX = canvas.canvas.width / Math.max(groupSize[0], 1) - 0.02;
                const zoomY = canvas.canvas.height / Math.max(groupSize[1], 1) - 0.02;
                canvas.setZoom?.(Math.min(zoomCurrent, zoomX, zoomY), [
                    canvas.canvas.width / 2,
                    canvas.canvas.height / 2,
                ]);
                canvas.setDirty?.(true, true);
            }
        } else {
            this.toggle();
        }

        return true;
    }

    computeSize(width) {
        return [width, WIDGET_HEIGHT];
    }

    serializeValue() {
        return this.value;
    }
}

function removeWidget(node, widgetOrIndex) {
    const index = typeof widgetOrIndex === "number"
        ? widgetOrIndex
        : node.widgets.indexOf(widgetOrIndex);

    if (index >= 0) {
        const [widget] = node.widgets.splice(index, 1);
        widget?.onRemove?.();
    }
}

function refreshWidgets(node) {
    ensureProperties(node);

    const previousSize = [...(node.size || [260, 0])];
    const groups = getVisibleGroups(node);
    let widgetStructureChanged = false;
    let index = 0;

    for (const group of groups) {
        const label = `Enable ${group.title || "Group"}`;
        let widget = (node.widgets || []).find((candidate) => candidate.group === group);
        let dirty = false;

        if (!widget) {
            widget = new FastGroupToggleWidget(group, node);
            node.addCustomWidget(widget);
            widgetStructureChanged = true;
            dirty = true;
        }

        if (widget.label !== label) {
            widget.label = label;
            dirty = true;
        }

        const toggled = !!group.rgthree_hasAnyActiveNode;
        if (widget.toggled !== toggled) {
            widget.toggled = toggled;
            dirty = true;
        }

        if (node.widgets[index] !== widget) {
            const oldIndex = node.widgets.indexOf(widget);
            if (oldIndex >= 0) {
                node.widgets.splice(index, 0, node.widgets.splice(oldIndex, 1)[0]);
                dirty = true;
            }
        }

        if (dirty) {
            markDirty(node, true, false);
        }

        index++;
    }

    while ((node.widgets || []).length > index) {
        removeWidget(node, index);
        widgetStructureChanged = true;
    }

    if (widgetStructureChanged) {
        const computedSize = node.computeSize();
        node.setSize?.([
            Math.max(previousSize[0], computedSize[0], 260),
            Math.max(previousSize[1], computedSize[1]),
        ]);
    }
}

function scheduleRefresh(ms = 500) {
    if (refreshTimer || !fastGroupNodes.size) {
        return;
    }

    refreshTimer = setTimeout(() => {
        refreshTimer = null;
        for (const node of fastGroupNodes) {
            refreshWidgets(node);
        }
        scheduleRefresh();
    }, ms);
}

function stopRefreshIfIdle() {
    if (!fastGroupNodes.size && refreshTimer) {
        clearTimeout(refreshTimer);
        refreshTimer = null;
    }
}

function handleAction(node, action) {
    const widgets = (node.widgets || []).filter((widget) => typeof widget.doModeChange === "function");

    if (action === "Bypass all" || action === "Mute all") {
        const alwaysOne = node.properties?.[PROPERTY_RESTRICTION] === "always one";
        widgets.forEach((widget, index) => widget.doModeChange(alwaysOne && index === 0, true));
    } else if (action === "Enable all") {
        const onlyOne = String(node.properties?.[PROPERTY_RESTRICTION] || "").includes(" one");
        widgets.forEach((widget, index) => widget.doModeChange(!(onlyOne && index > 0), true));
    } else if (action === "Toggle all") {
        const onlyOne = String(node.properties?.[PROPERTY_RESTRICTION] || "").includes(" one");
        let foundOne = false;
        widgets.forEach((widget) => {
            const nextValue = onlyOne && foundOne ? false : !widget.toggled;
            foundOne = foundOne || nextValue;
            widget.doModeChange(nextValue, true);
        });

        if (!foundOne && node.properties?.[PROPERTY_RESTRICTION] === "always one") {
            widgets[widgets.length - 1]?.doModeChange(true, true);
        }
    }
}

function setupNode(node, config = TARGET_NODE_CONFIG[node.type]) {
    if (!config) {
        return;
    }

    node.isVirtualNode = true;
    node.serialize_widgets = false;
    node.category = CATEGORY;
    node.__cpFastGroupsModeOff = config.modeOff;
    ensureProperties(node);
    setTimeout(() => refreshWidgets(node), 1);
}

app.registerExtension({
    name: EXTENSION_NAME,

    beforeRegisterNodeDef(nodeType, nodeData) {
        const config = nodeData && TARGET_NODE_CONFIG[nodeData.name];
        if (!config) {
            return;
        }

        nodeType.category = CATEGORY;
        nodeType.exposedActions = [config.actionOff, "Enable all", "Toggle all"];
        nodeType["@matchColors"] = { type: "string" };
        nodeType["@matchTitle"] = { type: "string" };
        nodeType["@showNav"] = { type: "boolean" };
        nodeType["@showAllGraphs"] = { type: "boolean" };
        nodeType["@sort"] = {
            type: "combo",
            values: ["position", "alphanumeric", "custom alphabet"],
        };
        nodeType["@customSortAlphabet"] = { type: "string" };
        nodeType["@toggleRestriction"] = {
            type: "combo",
            values: ["default", "max one", "always one"],
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            setupNode(this, config);
            return result;
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (...args) {
            const result = originalConfigure?.apply(this, args);
            setupNode(this, config);
            return result;
        };

        const originalOnAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function (...args) {
            const result = originalOnAdded?.apply(this, args);
            fastGroupNodes.add(this);
            scheduleRefresh(8);
            return result;
        };

        const originalOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            fastGroupNodes.delete(this);
            stopRefreshIfIdle();
            return originalOnRemoved?.apply(this, args);
        };

        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const result = originalGetExtraMenuOptions?.apply(this, [canvas, options]) || options || [];
            result.push(null);
            for (const action of nodeType.exposedActions) {
                result.push({
                    content: action,
                    callback: () => handleAction(this, action),
                });
            }
            return result;
        };

        nodeType.prototype.handleAction = function (action) {
            handleAction(this, action);
        };
    },

    loadedGraphNode(node) {
        const config = node && TARGET_NODE_CONFIG[node.type];
        if (config) {
            setupNode(node, config);
            fastGroupNodes.add(node);
            scheduleRefresh(8);
        }
    },
});
