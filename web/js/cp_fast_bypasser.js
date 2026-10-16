import { app } from "/scripts/app.js";

const EXTENSION_NAME = "ComfyUI.CustomPack.RgthreeFastBypasser";
const TARGET_NODE_NAME = "Fast Bypasser (rgthree)";
const CATEGORY = "Custom pack/Imported/Rgthree";
const MODE_ON = LiteGraph.ALWAYS ?? 0;
const MODE_OFF = 4;
const REFRESH_INTERVAL = 500;

const fastBypasserNodes = new Set();
let refreshTimer = null;

function markDirty(node, foreground = true, background = false) {
    if (typeof node?.setDirtyCanvas === "function") {
        node.setDirtyCanvas(foreground, background);
    } else {
        node?.graph?.setDirtyCanvas?.(foreground, background);
        app.graph?.setDirtyCanvas?.(foreground, background);
    }
}

function ensureProperties(node) {
    node.properties = node.properties || {};
    if (node.properties.toggleRestriction === undefined) {
        node.properties.toggleRestriction = "default";
    }

    if (!node.__cpFastBypasserPropertiesAdded) {
        node.__cpFastBypasserPropertiesAdded = true;
        try {
            node.addProperty?.(
                "toggleRestriction",
                node.properties.toggleRestriction,
                "combo",
                { values: ["default", "max one", "always one"] },
            );
        } catch (_error) {
            // Older frontend builds can throw when a serialized property already exists.
        }
    }
}

function getGraph(node) {
    return node?.graph || app.canvas?.getCurrentGraph?.() || app.graph;
}

function getLink(graph, linkId) {
    if (linkId && typeof linkId === "object") {
        return linkId;
    }
    return (
        graph?.links?.[linkId]
        || graph?.links?.get?.(linkId)
        || graph?._links?.[linkId]
        || graph?._links?.get?.(linkId)
        || null
    );
}

function getNodeById(graph, nodeId) {
    if (typeof graph?.getNodeById === "function") {
        return graph.getNodeById(nodeId);
    }
    return (graph?._nodes || graph?.nodes || []).find((node) => node.id === nodeId) || null;
}

function getInputOriginNode(node, input) {
    if (input?.link === null || input?.link === undefined) {
        return null;
    }

    const graph = getGraph(node);
    const link = getLink(graph, input.link);
    return link ? getNodeById(graph, link.origin_id) : null;
}

function isPassThroughNode(node) {
    const type = String(node?.type || node?.constructor?.type || "");
    return (
        type.includes("Reroute")
        || type.includes("Node Combiner")
        || type.includes("Node Collector")
    );
}

function collectUpstreamNodes(node, seen = new Set()) {
    if (!node || seen.has(node)) {
        return [];
    }
    seen.add(node);

    if (!isPassThroughNode(node)) {
        return [node];
    }

    const linkedNodes = [];
    for (const input of node.inputs || []) {
        const originNode = getInputOriginNode(node, input);
        linkedNodes.push(...collectUpstreamNodes(originNode, seen));
    }
    return linkedNodes;
}

function getNodesForInput(node, input) {
    return collectUpstreamNodes(getInputOriginNode(node, input));
}

function getLinkedNodes(node) {
    const linkedNodes = [];
    const seen = new Set();

    for (const input of node.inputs || []) {
        for (const linkedNode of getNodesForInput(node, input)) {
            if (linkedNode !== node && !seen.has(linkedNode)) {
                seen.add(linkedNode);
                linkedNodes.push(linkedNode);
            }
        }
    }
    return linkedNodes;
}

function changeModeOfNodes(nodeOrNodes, mode) {
    const stack = Array.isArray(nodeOrNodes) ? [...nodeOrNodes] : [nodeOrNodes];
    const seen = new Set();

    while (stack.length) {
        const node = stack.pop();
        if (!node || seen.has(node)) {
            continue;
        }
        seen.add(node);
        node.mode = mode;

        if (node.isSubgraphNode?.() && node.subgraph?.nodes) {
            stack.push(...node.subgraph.nodes);
        }
    }
}

function removeWidget(node, widget) {
    const index = (node.widgets || []).indexOf(widget);
    if (index >= 0) {
        node.widgets.splice(index, 1);
        widget.onRemove?.();
    }
}

function getToggleWidgets(node) {
    return (node.widgets || []).filter((widget) => widget.__cpFastBypasserWidget);
}

function configureWidget(widget, node, linkedNode) {
    widget.__cpFastBypasserWidget = true;
    widget.__cpLinkedNode = linkedNode;
    widget.name = `Enable ${linkedNode.title || linkedNode.type || "Node"}`;
    widget.options = { on: "yes", off: "no" };

    widget.doModeChange = (forceValue = null, skipOtherNodeCheck = false) => {
        let nextValue = forceValue === null || forceValue === undefined
            ? linkedNode.mode === MODE_OFF
            : !!forceValue;
        const restriction = node.properties?.toggleRestriction || "default";

        if (!skipOtherNodeCheck) {
            if (nextValue && restriction.includes(" one")) {
                for (const otherWidget of getToggleWidgets(node)) {
                    if (otherWidget !== widget) {
                        otherWidget.doModeChange(false, true);
                    }
                }
            } else if (!nextValue && restriction === "always one") {
                const anotherEnabled = getToggleWidgets(node).some(
                    (otherWidget) => otherWidget !== widget && otherWidget.value,
                );
                if (!anotherEnabled) {
                    nextValue = true;
                }
            }
        }

        changeModeOfNodes(linkedNode, nextValue ? MODE_ON : MODE_OFF);
        widget.value = nextValue;
        linkedNode.graph?.setDirtyCanvas?.(true, false);
        markDirty(node, true, false);
    };

    widget.callback = () => widget.doModeChange();
}

function createWidget(node, linkedNode) {
    let widget = null;
    widget = node.addWidget("toggle", "", false, "", { on: "yes", off: "no" });
    configureWidget(widget, node, linkedNode);
    return widget;
}

function stabilizeInputs(node) {
    node.inputs = node.inputs || [];
    let changed = false;
    const lastInput = node.inputs[node.inputs.length - 1];

    if (!lastInput || (lastInput.link !== null && lastInput.link !== undefined)) {
        node.addInput("", "*");
        changed = true;
    }

    for (let index = node.inputs.length - 2; index >= 0; index--) {
        const input = node.inputs[index];
        if (input.link === null || input.link === undefined) {
            node.removeInput(index);
            changed = true;
            continue;
        }

        const linkedNode = getNodesForInput(node, input)[0];
        const inputName = linkedNode?.title || linkedNode?.type || "";
        if (input.name !== inputName) {
            input.name = inputName;
            changed = true;
        }
    }

    return changed;
}

function stabilizeWidgets(node) {
    const linkedNodes = getLinkedNodes(node);
    const desiredWidgets = [];
    let changed = false;

    for (const linkedNode of linkedNodes) {
        let widget = getToggleWidgets(node).find(
            (candidate) => candidate.__cpLinkedNode === linkedNode,
        );
        if (!widget) {
            widget = createWidget(node, linkedNode);
            changed = true;
        } else {
            const previousName = widget.name;
            configureWidget(widget, node, linkedNode);
            changed = changed || previousName !== widget.name;
        }

        const enabled = linkedNode.mode === MODE_ON;
        if (widget.value !== enabled) {
            widget.value = enabled;
            changed = true;
        }
        desiredWidgets.push(widget);
    }

    for (const widget of getToggleWidgets(node)) {
        if (!desiredWidgets.includes(widget)) {
            removeWidget(node, widget);
            changed = true;
        }
    }

    for (let index = 0; index < desiredWidgets.length; index++) {
        const widget = desiredWidgets[index];
        const currentIndex = node.widgets.indexOf(widget);
        if (currentIndex !== index) {
            node.widgets.splice(index, 0, node.widgets.splice(currentIndex, 1)[0]);
            changed = true;
        }
    }

    return changed;
}

function refreshNode(node) {
    if (!node?.graph) {
        return;
    }

    const oldWidth = node.size?.[0] || 260;
    const inputsChanged = stabilizeInputs(node);
    const widgetsChanged = stabilizeWidgets(node);
    const structureChanged = inputsChanged || widgetsChanged;
    if (structureChanged) {
        const size = node.computeSize?.() || node.size || [260, 80];
        size[0] = Math.max(oldWidth, size[0], 260);
        node.setSize?.(size);
        markDirty(node, true, true);
    }
}

function scheduleRefresh(delay = REFRESH_INTERVAL) {
    if (refreshTimer || !fastBypasserNodes.size) {
        return;
    }

    refreshTimer = setTimeout(() => {
        refreshTimer = null;
        for (const node of fastBypasserNodes) {
            refreshNode(node);
        }
        scheduleRefresh();
    }, delay);
}

function stopRefreshIfIdle() {
    if (!fastBypasserNodes.size && refreshTimer) {
        clearTimeout(refreshTimer);
        refreshTimer = null;
    }
}

function setupNode(node) {
    node.isVirtualNode = true;
    node.serialize_widgets = false;
    node.category = CATEGORY;
    ensureProperties(node);

    if (!(node.inputs || []).length) {
        node.addInput("", "*");
    }
    if (!(node.outputs || []).length) {
        node.addOutput("OPT_CONNECTION", "*");
    }

    setTimeout(() => refreshNode(node), 32);
}

function handleAction(node, action) {
    const widgets = getToggleWidgets(node);
    if (action === "Bypass all") {
        widgets.forEach((widget) => widget.doModeChange(false, true));
    } else if (action === "Enable all") {
        widgets.forEach((widget) => widget.doModeChange(true, true));
    } else if (action === "Toggle all") {
        widgets.forEach((widget) => widget.doModeChange(!widget.value, true));
    }
}

app.registerExtension({
    name: EXTENSION_NAME,

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET_NODE_NAME) {
            return;
        }

        nodeType.category = CATEGORY;
        nodeType.exposedActions = ["Bypass all", "Enable all", "Toggle all"];
        nodeType["@toggleRestriction"] = {
            type: "combo",
            values: ["default", "max one", "always one"],
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            setupNode(this);
            return result;
        };

        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (...args) {
            const result = configure?.apply(this, args);
            setupNode(this);
            return result;
        };

        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function (...args) {
            const result = onAdded?.apply(this, args);
            fastBypasserNodes.add(this);
            scheduleRefresh(8);
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            fastBypasserNodes.delete(this);
            stopRefreshIfIdle();
            return onRemoved?.apply(this, args);
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const result = onConnectionsChange?.apply(this, args);
            setTimeout(() => refreshNode(this), 16);
            return result;
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const menu = getExtraMenuOptions?.apply(this, [canvas, options]) || options || [];
            menu.push(null);
            for (const action of nodeType.exposedActions) {
                menu.push({
                    content: action,
                    callback: () => handleAction(this, action),
                });
            }
            return menu;
        };

        nodeType.prototype.handleAction = function (action) {
            handleAction(this, action);
        };
    },

    loadedGraphNode(node) {
        if (node?.type === TARGET_NODE_NAME) {
            setupNode(node);
            fastBypasserNodes.add(node);
            scheduleRefresh(8);
        }
    },
});
